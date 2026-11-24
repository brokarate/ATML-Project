"""Score a fixed ARC-Easy test sample as a supplementary general-utility check."""

import argparse
import random
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from modeling import encode, load_model, pack, sequence_logprobs
from pilot import read_json, write_json


REPO = "allenai/ai2_arc"
REVISION = "210d026faf9955653af8916fad021475a3f00453"
FILE = "ARC-Easy/test-00000-of-00001.parquet"


@torch.inference_mode()
def score(model, tokenizer, examples, device):
    model.eval()
    outputs = []
    for start in range(0, len(examples), 8):
        rows = examples[start:start + 8]
        candidates = [
            {"prompt": f"Question: {row['question']}\nAnswer:", "answer": text}
            for row in rows for text in row["choices"]["text"]
        ]
        encoded = encode(tokenizer, candidates, False)
        totals = sequence_logprobs(
            model, *pack(encoded, tokenizer.eos_token_id, device)
        ).tolist()
        averaged = iter(total / len(answer)
                        for total, (_, answer) in zip(totals, encoded, strict=True))
        for row in rows:
            labels = row["choices"]["label"]
            scores = {label: next(averaged) for label in labels}
            prediction = max(scores, key=scores.__getitem__)
            outputs.append({"id": row["id"], "answer": row["answerKey"],
                            "prediction": prediction, "mean_token_logprobs": scores,
                            "correct": prediction == row["answerKey"]})
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/pilot"))
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default="cpu")
    parser.add_argument("--edits-only", action="store_true")
    parser.add_argument("--bare-only", action="store_true")
    args = parser.parse_args()
    config = read_json(args.run / "manifest.json")["contract"]["config"]
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(6)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    source = hf_hub_download(REPO, FILE, repo_type="dataset", revision=REVISION)
    rows = pq.read_table(source).to_pylist()
    seed = 5999
    sample = random.Random(seed).sample(rows, 256)
    checkpoints = args.run / "checkpoints"
    bases = [] if args.edits_only else ["starting", "full", "clean"]
    for method in config["methods"]:
        selection = read_json(args.run / f"selection-{method}.json")["selected"]
        if selection:
            bases.append(selection)
    result_dir = args.run / "arc_easy"
    summary = {}
    for base in bases:
        model_source = config["model"] if base == "starting" else checkpoints / base
        revision = config["revision"] if base == "starting" else None
        adapters = [None]
        if base != "starting" and not args.bare_only:
            adapters += sorted((checkpoints / "adapters").glob("old-*"))
        for adapter in adapters:
            name = base if adapter is None else f"{base}__{adapter.name}"
            path = result_dir / f"{name}.json"
            if path.exists():
                outputs = read_json(path)
            else:
                model = load_model(model_source, device, revision)
                if adapter is not None:
                    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
                outputs = score(model, tokenizer, sample, device)
                write_json(path, outputs)
                del model
                if device.type == "mps":
                    torch.mps.empty_cache()
            correct = sum(row["correct"] for row in outputs)
            summary[name] = {"correct": correct, "n": len(outputs), "accuracy": correct / len(outputs)}
            print(f"{name}: {correct}/{len(outputs)}", flush=True)
    write_json(result_dir / "summary.json", {
        "dataset": REPO, "revision": REVISION, "file": FILE, "sample_seed": seed,
        "prompt": "Question: {question}\nAnswer:", "metric": "mean token log probability",
        "conditions": summary,
    })


if __name__ == "__main__":
    main()
