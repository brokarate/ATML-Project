"""Quantized backbones with stacked, separately trained LoRA adapters.

The expanded study never changes backbone weights. Record acquisition, unlearning,
and the task are each a named LoRA adapter on one shared backbone:

    full      = backbone + know-full
    clean     = backbone + know-clean
    edited    = backbone + know-full + edit-<method>-<step>
    condition = <base stack> + task adapter

Active adapters add their low-rank updates, so attaching a stale task adapter to an
edited base reuses the exact saved task weights with no further training.
"""

import hashlib
import math
import random
import re
import shutil
import tempfile
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from peft.tuners.lora.layer import LoraLayer
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


DTYPE = {"auto": None}


def compute_dtype(device):
    # Kaggle T4s lack bf16, so "auto" means fp16 with loss scaling there. Set
    # "compute_dtype": "float32" in a config if a backbone overflows in fp16.
    if DTYPE["auto"] is not None:
        return DTYPE["auto"]
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_backbone(config, device):
    choice = config.get("compute_dtype", "auto")
    if choice != "auto":
        DTYPE["auto"] = getattr(torch, choice)
    dtype = compute_dtype(device)
    kwargs = {"revision": config["revision"], "dtype": dtype, "use_safetensors": True}
    if config.get("quantize"):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": device.index or 0}
    model = AutoModelForCausalLM.from_pretrained(config["model"], **kwargs)
    if not config.get("quantize"):
        model.to(device)
    model.requires_grad_(False)
    if config.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    return model


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def lora_config(rank, targets):
    return LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, bias="none",
                      target_modules=list(targets), task_type="CAUSAL_LM")


class Stack:
    """One backbone, many named adapters, any subset active at once."""

    def __init__(self, backbone):
        self.backbone = backbone
        self.peft = None
        self.loaded = set()

    @property
    def lm(self):
        # Use the transformers model with injected LoRA layers directly. PeftModel's
        # own forward expects a single active adapter, but layers accept a list.
        return self.backbone if self.peft is None else self.peft.base_model.model

    def load(self, name, path):
        if name in self.loaded:
            return
        if self.peft is None:
            self.peft = PeftModel.from_pretrained(self.backbone, path, adapter_name=name,
                                                  is_trainable=False)
        else:
            self.peft.load_adapter(path, adapter_name=name, is_trainable=False)
        self.loaded.add(name)

    def create(self, name, config):
        if name in self.loaded:
            raise ValueError(f"Adapter {name} already exists")
        if self.peft is None:
            self.peft = get_peft_model(self.backbone, config, adapter_name=name)
        else:
            self.peft.add_adapter(name, config)
        self.loaded.add(name)

    def delete(self, name):
        self.activate([])
        self.peft.delete_adapter(name)
        self.loaded.discard(name)

    def activate(self, names, trainable=None, scale=None):
        names = list(names)
        if self.peft is None:
            if names:
                raise ValueError("No adapters are loaded")
            return
        self.peft.base_model.set_adapter(names)
        marker = f".{trainable}." if trainable else None
        for parameter_name, parameter in self.peft.named_parameters():
            parameter.requires_grad_(marker is not None and marker in parameter_name)
        for module in self.peft.modules():
            if isinstance(module, LoraLayer):
                for name in module.scaling:
                    factor = (scale or {}).get(name, 1.0)
                    module.scaling[name] = module.lora_alpha[name] / module.r[name] * factor
        missing = set(names) - self.loaded
        if missing:
            raise ValueError(f"Adapters not loaded: {missing}")

    def save(self, name, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=".adapter-", dir=path.parent))
        try:
            # PeftModel.save_pretrained assumes a single active adapter, so write the
            # same two files (weights and adapter_config.json) directly.
            weights = get_peft_model_state_dict(self.peft, adapter_name=name)
            save_file({k: v.detach().cpu().contiguous() for k, v in weights.items()},
                      temp / "adapter_model.safetensors", metadata={"format": "pt"})
            self.peft.peft_config[name].save_pretrained(temp)
            if path.exists():
                shutil.rmtree(path)
            temp.rename(path)
        finally:
            shutil.rmtree(temp, ignore_errors=True)
        return digest(path / "adapter_model.safetensors")


def encode(tokenizer, rows, terminate):
    batch = []
    for row in rows:
        prompt = tokenizer.encode(row["prompt"], add_special_tokens=False)
        answer = tokenizer.encode(" " + row["answer"], add_special_tokens=False)
        if terminate:
            answer.append(tokenizer.eos_token_id)
        batch.append((prompt, answer))
    return batch


def pack(encoded, pad, device):
    width = max(len(prompt) + len(answer) for prompt, answer in encoded)
    inputs, labels, mask = [], [], []
    for prompt, answer in encoded:
        padding = width - len(prompt) - len(answer)
        inputs.append(prompt + answer + [pad] * padding)
        labels.append([-100] * len(prompt) + answer + [-100] * padding)
        mask.append([1] * (len(prompt) + len(answer)) + [0] * padding)
    return tuple(torch.tensor(items, device=device) for items in (inputs, labels, mask))


def answer_logprobs(lm, inputs, labels, mask):
    logits = lm(input_ids=inputs, attention_mask=mask, use_cache=False).logits[:, :-1]
    shifted = labels[:, 1:]
    active = shifted != -100
    rows = active.nonzero(as_tuple=True)[0]
    scores = -F.cross_entropy(logits[active].float(), shifted[active], reduction="none")
    return scores, rows


def sequence_logprobs(lm, inputs, labels, mask):
    scores, rows = answer_logprobs(lm, inputs, labels, mask)
    totals = scores.new_zeros(inputs.shape[0]).index_add(0, rows, scores)
    counts = scores.new_zeros(inputs.shape[0]).index_add(0, rows, torch.ones_like(scores))
    return totals, counts


def sample_batches(encoded, rng, batch_size):
    indices = list(range(len(encoded)))
    while True:
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            yield [encoded[index] for index in indices[start:start + batch_size]]


@torch.no_grad()
def reference_scores(lm, tokenizer, rows, config, device):
    """Frozen sequence log probabilities used by NPO."""
    lm.eval()
    encoded = encode(tokenizer, rows, True)
    result = {}
    for start in range(0, len(encoded), config["batch"]):
        examples = encoded[start:start + config["batch"]]
        totals, _ = sequence_logprobs(lm, *pack(examples, tokenizer.pad_token_id, device))
        for (prompt, answer), score in zip(examples, totals.tolist(), strict=True):
            result[tuple(prompt + answer)] = score
    return result


def train(stack, tokenizer, rows, config, device, *, steps, lr, seed, forget=None,
          method=None, reference=None, checkpoints=None, validate=None):
    lm = stack.lm
    lm.train()
    trainable = [p for p in lm.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters are active")
    for parameter in trainable:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    # fp16 activations need loss scaling; bf16 and fp32 do not.
    scaler = torch.amp.GradScaler("cuda", enabled=compute_dtype(device) == torch.float16)
    pad = tokenizer.pad_token_id
    batches = sample_batches(encode(tokenizer, rows, True), random.Random(seed), config["batch"])
    if forget is not None:
        forget_batches = sample_batches(encode(tokenizer, forget, True),
                                        random.Random(seed + 1), config["batch"])
    stats = {"steps": 0, "skipped_steps": 0, "input_tokens": 0, "supervised_tokens": 0,
             "trainable_parameters": sum(p.numel() for p in trainable), "seconds": 0.0}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.monotonic()
    validation_seconds = 0.0
    for step in range(1, steps + 1):
        batch = next(batches)
        optimizer.zero_grad(set_to_none=True)
        retain_scores, _ = answer_logprobs(lm, *pack(batch, pad, device))
        loss = -retain_scores.mean()
        if forget is not None:
            forgotten = next(forget_batches)
            if method == "graddiff":
                scores, _ = answer_logprobs(lm, *pack(forgotten, pad, device))
                forget_loss = scores.mean()
            elif method == "npo":
                totals, _ = sequence_logprobs(lm, *pack(forgotten, pad, device))
                frozen = totals.new_tensor([reference[tuple(p + a)] for p, a in forgotten])
                beta = config["beta"]
                forget_loss = F.softplus(beta * (totals - frozen)).mean() * (2 / beta)
            else:
                raise ValueError(f"Unknown unlearning method {method}")
            loss = config["retain_weight"] * loss + forget_loss
            batch = batch + forgotten
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.is_enabled() and scaler.get_scale() < previous_scale:
            stats["skipped_steps"] += 1
        stats["steps"] = step
        stats["input_tokens"] += sum(len(p) + len(a) for p, a in batch)
        stats["supervised_tokens"] += sum(len(a) for _, a in batch)
        if step == 1 or step % 20 == 0 or step == steps:
            print(f"  {step:>4}/{steps} loss={loss.item():.4f}", flush=True)
        if checkpoints is not None and step in checkpoints:
            stats["seconds"] = time.monotonic() - start_time - validation_seconds
            before = time.monotonic()
            validate(step, stats.copy())
            validation_seconds += time.monotonic() - before
            lm.train()
    stats["seconds"] = time.monotonic() - start_time - validation_seconds
    if device.type == "cuda":
        stats["peak_cuda_bytes"] = torch.cuda.max_memory_allocated(device)
    lm.eval()
    return stats


def rouge_l_recall(prediction, reference):
    """Token-level ROUGE-L recall, as used by TOFU, without extra dependencies."""
    tokenize = lambda text: re.findall(r"\w+", text.lower())
    pred, ref = tokenize(prediction), tokenize(reference)
    if not ref:
        return 0.0
    previous = [0] * (len(pred) + 1)
    for token in ref:
        current = [0]
        for index, other in enumerate(pred):
            current.append(previous[index] + 1 if token == other
                           else max(previous[index + 1], current[index]))
        previous = current
    return previous[-1] / len(ref)


def judge(row, prediction, config):
    if row["kind"] == "code":
        hit = bool(re.search(r"\b" + re.escape(row["answer"]) + r"\b", prediction))
        return {"correct": hit, "exact": prediction.strip() == row["answer"]}
    rouge = rouge_l_recall(prediction, row["answer"])
    return {"correct": rouge >= config["rouge_threshold"], "rouge_l": rouge}


@torch.inference_mode()
def generate(lm, tokenizer, rows, config, device, seed):
    lm.eval()
    outputs = []
    tokenizer.padding_side = "left"
    batch_size = config["eval_batch"]
    for start in range(0, len(rows), batch_size):
        subset = rows[start:start + batch_size]
        samples = subset[0].get("samples", 0)
        encoded = tokenizer([row["prompt"] for row in subset], padding=True,
                            add_special_tokens=False, return_tensors="pt").to(device)
        options = {"do_sample": False}
        if samples:
            torch.manual_seed(seed + start)
            options = {"do_sample": True, "temperature": config["sample_temperature"],
                       "top_p": config["sample_top_p"], "num_return_sequences": samples}
        generated = lm.generate(**encoded, max_new_tokens=config["max_new_tokens"],
                                pad_token_id=tokenizer.pad_token_id, use_cache=True, **options)
        texts = tokenizer.batch_decode(generated[:, encoded.input_ids.shape[1]:],
                                       skip_special_tokens=True)
        per_row = samples or 1
        totals, counts = sequence_logprobs(
            lm, *pack(encode(tokenizer, subset, False), tokenizer.pad_token_id, device))
        for index, row in enumerate(subset):
            predictions = texts[index * per_row:(index + 1) * per_row]
            judged = [judge(row, text, config) for text in predictions]
            result = {**row, "prediction": predictions[0],
                      "answer_logprob": totals[index].item(),
                      "answer_mean_logprob": (totals[index] / counts[index]).item()}
            if samples:
                hits = sum(item["correct"] for item in judged)
                result.update({"samples_text": predictions, "hits": hits,
                               "correct": hits > 0, "hit_rate": hits / samples})
            else:
                result.update(judged[0])
            outputs.append(result)
    return outputs


@torch.inference_mode()
def choose(lm, tokenizer, rows, config, device):
    """Rank candidate continuations by mean token log probability."""
    lm.eval()
    outputs = []
    batch_size = config["eval_batch"]
    for start in range(0, len(rows), batch_size):
        subset = rows[start:start + batch_size]
        candidates = [{"prompt": row["prompt"], "answer": choice}
                      for row in subset for choice in row["choices"]]
        totals, counts = sequence_logprobs(
            lm, *pack(encode(tokenizer, candidates, False), tokenizer.pad_token_id, device))
        means = iter((totals / counts).tolist())
        for row in subset:
            scores = [next(means) for _ in row["choices"]]
            prediction = max(range(len(scores)), key=scores.__getitem__)
            result = {**row, "prediction": prediction, "choice_mean_logprobs": scores,
                      "correct": prediction == row["label"]}
            if row.get("truth_ratio"):
                # TOFU truth ratio: mean normalized probability of perturbed answers
                # over the normalized probability of the paraphrased answer.
                perturbed = [math.exp(s) for i, s in enumerate(scores) if i != row["label"]]
                result["truth_ratio"] = (sum(perturbed) / len(perturbed)) / math.exp(scores[row["label"]])
            outputs.append(result)
    return outputs


def evaluate(stack, tokenizer, splits, config, device, seed):
    rows = {}
    for name, examples in splits.items():
        if not examples:
            continue
        if "choices" in examples[0]:
            rows[name] = choose(stack.lm, tokenizer, examples, config, device)
        else:
            rows[name] = generate(stack.lm, tokenizer, examples, config, device, seed)
    return summarize_all(rows), rows


def summarize_all(rows):
    metrics = {}
    for name, examples in rows.items():
        result = {"n": len(examples), "correct": sum(row["correct"] for row in examples)}
        result["accuracy"] = result["correct"] / result["n"]
        for key in ("exact", "rouge_l", "answer_logprob", "answer_mean_logprob",
                    "hit_rate", "truth_ratio"):
            values = [row[key] for row in examples if key in row]
            if values:
                result[f"mean_{key}"] = sum(float(value) for value in values) / len(values)
        metrics[name] = result
    return metrics
