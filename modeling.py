"""Answer-only training, GradDiff/NPO, and greedy evaluation."""

import random
import re
import time

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM


def load_model(source, device, revision=None):
    return AutoModelForCausalLM.from_pretrained(
        source, revision=revision, dtype=torch.float32, use_safetensors=True
    ).to(device)


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


def answer_logprobs(model, inputs, labels, mask):
    # Project only answer positions through GPT-Neo's vocabulary head. PEFT injects
    # its LoRA layers inside the transformer, so both paths use active adapter weights.
    shifted = labels[:, 1:]
    active = shifted != -100
    hidden = model.transformer(input_ids=inputs, attention_mask=mask,
                               use_cache=False).last_hidden_state
    positions = active.nonzero(as_tuple=True)
    selected = model.lm_head(hidden[positions[0], positions[1], :]).float()
    token_scores = -F.cross_entropy(selected, shifted[active], reduction="none")
    return token_scores, positions[0]


def sequence_logprobs(model, inputs, labels, mask):
    scores, rows = answer_logprobs(model, inputs, labels, mask)
    return scores.new_zeros(inputs.shape[0]).index_add(0, rows, scores)


def sample_batches(encoded, rng, batch_size):
    indices = list(range(len(encoded)))
    while True:
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            yield [encoded[index] for index in indices[start:start + batch_size]]


def train(model, tokenizer, rows, config, device, *, steps, lr, seed,
          forget=None, method=None, reference=None, checkpoints=None, validate=None):
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    # FP32 optimizer state is important for small unlearning updates.
    for parameter in trainable:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    retain_batches = sample_batches(encode(tokenizer, rows, True), random.Random(seed),
                                    config["batch"])
    pad = tokenizer.eos_token_id
    reference_scores = {}
    if forget is not None:
        encoded = encode(tokenizer, forget, True)
        forget_batches = sample_batches(encoded, random.Random(seed + 1), config["batch"])
        if method == "npo":
            reference.eval()
            with torch.no_grad():
                for start in range(0, len(encoded), config["batch"]):
                    examples = encoded[start:start + config["batch"]]
                    scores = sequence_logprobs(reference, *pack(examples, pad, device)).tolist()
                    for (prompt, answer), score in zip(examples, scores, strict=True):
                        reference_scores[tuple(prompt + answer)] = score

    stats = {"steps": 0, "input_tokens": 0, "supervised_tokens": 0,
             "trainable_parameters": sum(p.numel() for p in trainable), "seconds": 0.0}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.monotonic()
    validation_seconds = 0.0
    for step in range(1, steps + 1):
        batch = next(retain_batches)
        optimizer.zero_grad(set_to_none=True)
        retain_scores, _ = answer_logprobs(model, *pack(batch, pad, device))
        loss = -retain_scores.mean()
        if forget is not None:
            forgotten = next(forget_batches)
            if method == "graddiff":
                scores, _ = answer_logprobs(model, *pack(forgotten, pad, device))
                forget_loss = scores.mean()
            else:
                scores = sequence_logprobs(model, *pack(forgotten, pad, device))
                frozen = scores.new_tensor([reference_scores[tuple(p + a)] for p, a in forgotten])
                beta = config["beta"]
                forget_loss = F.softplus(beta * (scores - frozen)).mean() * (2 / beta)
            loss = config["retain_weight"] * loss + forget_loss
            batch = batch + forgotten
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        stats["steps"] = step
        stats["input_tokens"] += sum(len(p) + len(a) for p, a in batch)
        stats["supervised_tokens"] += sum(len(a) for _, a in batch)
        if step == 1 or step % 20 == 0 or step == steps:
            print(f"  {step:>4}/{steps} loss={loss.item():.4f}", flush=True)
        if checkpoints is not None and step in checkpoints:
            stats["seconds"] = time.monotonic() - start_time - validation_seconds
            if device.type == "cuda":
                stats["peak_cuda_bytes"] = torch.cuda.max_memory_allocated(device)
            before = time.monotonic()
            validate(model, step, stats.copy())
            validation_seconds += time.monotonic() - before
            model.train()
    stats["seconds"] = time.monotonic() - start_time - validation_seconds
    if device.type == "cuda":
        stats["peak_cuda_bytes"] = torch.cuda.max_memory_allocated(device)
    return stats


@torch.inference_mode()
def evaluate(model, tokenizer, rows, config, device):
    model.eval()
    outputs = []
    batch_size = config["batch"] * 2
    for start in range(0, len(rows), batch_size):
        subset = rows[start:start + batch_size]
        if "choices" in subset[0]:
            candidates = [{**row, "answer": choice}
                          for row in subset for choice in row["choices"]]
            scores = sequence_logprobs(
                model, *pack(encode(tokenizer, candidates, False), tokenizer.eos_token_id, device)
            ).view(len(subset), 2).tolist()
            for row, pair in zip(subset, scores, strict=True):
                index = max(range(2), key=pair.__getitem__)
                prediction = row["choices"][index]
                outputs.append({**row, "prediction": prediction,
                                "choice_logprobs": dict(zip(row["choices"], pair, strict=True)),
                                "correct": prediction == row["answer"]})
            continue

        tokenizer.padding_side = "left"
        encoded = tokenizer([row["prompt"] for row in subset], padding=True,
                            add_special_tokens=False, return_tensors="pt").to(device)
        generated = model.generate(**encoded, do_sample=False,
                                   max_new_tokens=config["max_new_tokens"],
                                   pad_token_id=tokenizer.eos_token_id)
        predictions = tokenizer.batch_decode(generated[:, encoded.input_ids.shape[1]:],
                                             skip_special_tokens=True)
        scores = sequence_logprobs(
            model, *pack(encode(tokenizer, subset, False), tokenizer.eos_token_id, device)
        ).tolist()
        for row, prediction, score in zip(subset, predictions, scores, strict=True):
            correct = bool(re.search(r"\b" + re.escape(row["answer"]) + r"\b", prediction))
            outputs.append({**row, "prediction": prediction, "answer_logprob": score,
                            "correct": correct, "exact": prediction.strip() == row["answer"]})
    return outputs
