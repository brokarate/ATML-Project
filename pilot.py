"""Run or resume the controlled stale-LoRA experiment by stage."""

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import shutil
import tempfile

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, set_seed

from modeling import evaluate, load_model, train
from records import GROUPS, build


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(handle.name, path)


def read_json(path):
    return json.loads(path.read_text())


def save_checkpoint(path, model, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=path.parent))
    try:
        model.save_pretrained(temp, safe_serialization=True)
        write_json(temp / "training.json", metadata)
        temp.rename(path)
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def summarize(rows):
    result = {"n": len(rows), "correct": sum(row["correct"] for row in rows)}
    result["accuracy"] = result["correct"] / result["n"]
    if "exact" in rows[0]:
        result["exact_accuracy"] = sum(row["exact"] for row in rows) / len(rows)
        result["mean_answer_logprob"] = sum(row["answer_logprob"] for row in rows) / len(rows)
    return result


class Pilot:
    def __init__(self, config, root, device):
        self.config = config
        self.root = root
        self.device = device
        self.checkpoints = root / "checkpoints"
        self.pairing = {"unrelated": "task_parcel", "related": "task_author",
                        "exposed": "task_author"}
        tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.data = build(config, tokenizer)
        contract = {"config": config, "data": self.data,
                    "code": {name: digest(Path(__file__).with_name(name))
                             for name in ("pilot.py", "modeling.py", "records.py")}}
        manifest = root / "manifest.json"
        if manifest.exists() and read_json(manifest)["contract"] != contract:
            raise SystemExit("This run uses different inputs or code. Choose a new run directory.")
        if not manifest.exists():
            write_json(manifest, {
                "contract": contract, "platform": platform.platform(),
                "device": str(device), "accelerator": (
                    torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine()
                ), "packages": {name: importlib.metadata.version(name)
                                 for name in ("torch", "transformers", "peft")},
                "generation": {"strategy": "greedy", "max_new_tokens": config["max_new_tokens"]},
            })

    def load(self, base):
        if base == "starting":
            return load_model(self.config["model"], self.device, self.config["revision"])
        return load_model(self.checkpoints / base, self.device)

    def release(self):
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def score(self, model, splits):
        rows = {name: evaluate(model, self.tokenizer, examples, self.config, self.device)
                for name, examples in splits.items()}
        metrics = {}
        for name, examples in rows.items():
            if name.startswith("task_"):
                for example in examples:
                    example["contains_answer"] = example["correct"]
                    example["correct"] = example["exact"]
            metrics[name] = summarize(examples)
        return metrics, rows

    def base(self):
        cfg = self.config
        train_data = self.data["train"]
        mixed_rows = {"full": train_data["retain"] + train_data["forget"],
                      "clean": train_data["retain"]}
        for base, rows in mixed_rows.items():
            path = self.checkpoints / base
            if path.exists():
                continue
            print(f"Training {base}", flush=True)
            set_seed(cfg["seed"])
            model = self.load("starting")
            stats = train(model, self.tokenizer, rows, cfg, self.device,
                          steps=cfg["base_steps"], lr=cfg["base_lr"], seed=cfg["seed"])
            save_checkpoint(path, model, stats)
            del model
            self.release()

        checks_path = self.root / "memorization.json"
        if checks_path.exists():
            checks = read_json(checks_path)
        else:
            splits = dict(self.data["validation"])
            splits["forget_training_prompt"] = [row for row in train_data["forget"]
                                                 if row["id"].endswith("/0")]
            checks = {}
            for base in ("starting", "full", "clean"):
                model = self.load(base)
                metrics, rows = self.score(model, splits)
                checks[base] = metrics
                write_json(self.root / "calibration" / f"{base}.json", rows)
                del model
                self.release()
            write_json(checks_path, checks)
        memo = {base: metrics["forget_training_prompt"]["accuracy"]
                for base, metrics in checks.items()}
        print(f"Target learning check: {memo}", flush=True)
        if (memo["full"] < cfg["min_full_forget"]
                or max(memo["starting"], memo["clean"]) > cfg["max_clean_forget"]):
            raise SystemExit("Target learning check failed. Inspect memorization.json before proceeding.")

    def adapter(self, base, group, rank, role):
        path = self.checkpoints / "adapters" / f"{role}-{group}-r{rank}"
        if path.exists():
            return path
        cfg = self.config
        print(f"Training {path.name} on {base}", flush=True)
        set_seed(cfg["seed"] + 10)
        model = get_peft_model(self.load(base), LoraConfig(
            r=rank, lora_alpha=2 * rank, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
        ))
        key = "tasks" if role == "old" else "clean_repair_tasks"
        rows = self.data["train"][key][group]
        stats = train(model, self.tokenizer, rows, cfg, self.device, steps=cfg["adapter_steps"],
                      lr=cfg["adapter_lr"], seed=cfg["seed"] + 10)
        save_checkpoint(path, model, {**stats, "base": base, "data": key, "rank": rank})
        del model
        self.release()
        return path

    def adapters(self):
        for rank in self.config["ranks"]:
            for group in GROUPS:
                self.adapter("full", group, rank, "old")
                self.adapter("clean", group, rank, "clean")

    def unlearn(self):
        cfg = self.config
        baseline = read_json(self.root / "memorization.json")["full"]
        splits = self.data["validation"]
        train_data = self.data["train"]
        for method in cfg["methods"]:
            selection_path = self.root / f"selection-{method}.json"
            if selection_path.exists():
                continue
            print(f"Editing full with {method}", flush=True)
            set_seed(cfg["seed"] + 20)
            model = self.load("full")
            reference = self.load("full") if method == "npo" else None
            candidates = []

            def validate(edited, step, stats):
                name = f"{method}-{step:03d}"
                metrics, rows = self.score(edited, splits)
                drop = {split: baseline[split]["accuracy"] - metrics[split]["accuracy"]
                        for split in splits}
                accepted = (
                    baseline["forget_validation"]["accuracy"] > 0
                    and drop["forget_validation"] >= cfg["min_edit_forgetting"]
                    * baseline["forget_validation"]["accuracy"]
                    and drop["retain_validation"] <= cfg["max_retain_drop"]
                    and drop["general_validation"] <= cfg["max_general_drop"]
                )
                candidate = {"base": name, "step": step, "metrics": metrics,
                             "accuracy_drop": drop, "accepted": accepted, "training": stats}
                print(f"  {name} validation drops: {drop}, accepted={accepted}", flush=True)
                write_json(self.root / "calibration" / f"{name}.json", rows)
                candidates.append(candidate)
                if accepted:
                    path = self.checkpoints / name
                    if not path.exists():
                        save_checkpoint(path, edited, stats)

            train(model, self.tokenizer, train_data["retain"], cfg, self.device,
                  steps=max(cfg["edit_steps"]), lr=cfg["edit_lr"], seed=cfg["seed"] + 20,
                  forget=train_data["forget"], method=method, reference=reference,
                  checkpoints=cfg["edit_steps"], validate=validate)
            selected = next((candidate["base"] for candidate in candidates if candidate["accepted"]), None)
            write_json(selection_path, {"selected": selected, "candidates": candidates})
            del model, reference
            self.release()

    def evaluation(self):
        cfg = self.config
        selected = [read_json(self.root / f"selection-{method}.json")["selected"]
                    for method in cfg["methods"]]
        edits = [base for base in selected if base]
        if not edits:
            raise SystemExit("No edit met the validation criteria. Final comparison was skipped.")
        for base in ["starting", "full", "clean", *edits]:
            self.condition(base, "base", "none", 0)
        for rank in cfg["ranks"]:
            for group in GROUPS:
                old = self.checkpoints / "adapters" / f"old-{group}-r{rank}"
                clean = self.checkpoints / "adapters" / f"clean-{group}-r{rank}"
                for base in ["full", "clean", *edits]:
                    self.condition(base, "old", group, rank, old)
                for base in ["clean", *edits]:
                    self.condition(base, "clean", group, rank, clean)
                for base in edits:
                    replacement = self.adapter(base, group, rank, f"replacement-{base}")
                    self.condition(base, "replacement", group, rank, replacement)

    def condition(self, base, role, group, rank, adapter=None):
        name = f"{base}__{role}__{group}__r{rank}"
        directory = self.root / "evaluation"
        metrics_path = directory / f"{name}.json"
        if metrics_path.exists():
            return
        print(f"Evaluating {name}", flush=True)
        model = self.load(base)
        adapter_hash = None
        splits = dict(self.data["test"])
        if adapter:
            adapter_hash = digest(adapter / "adapter_model.safetensors")
            model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
            model.requires_grad_(False)
            splits.pop("task_parcel" if self.pairing[group] == "task_author" else "task_author")
        metrics, rows = self.score(model, splits)
        write_json(directory / "examples" / f"{name}.json", rows)
        write_json(metrics_path, {"base": base, "role": role, "group": group, "rank": rank,
                                  "adapter_sha256": adapter_hash, "metrics": metrics})
        del model
        self.release()


def aggregate(root):
    conditions = [read_json(path) for path in sorted((root / "evaluation").glob("*.json"))]
    if not conditions:
        raise SystemExit("No completed evaluation conditions.")
    base_only = {entry["base"]: entry["metrics"] for entry in conditions if entry["role"] == "base"}
    old_full = {(entry["group"], entry["rank"]): entry["metrics"] for entry in conditions
                if entry["role"] == "old" and entry["base"] == "full"}
    rows = []
    for entry in conditions:
        metadata = {key: value for key, value in entry.items() if key != "metrics"}
        original = old_full.get((entry["group"], entry["rank"]), {})
        for split, metrics in entry["metrics"].items():
            baseline = base_only[entry["base"]][split]["accuracy"]
            previous = original.get(split, {}).get("accuracy")
            rows.append({**metadata, "split": split, **metrics,
                         "delta_vs_same_base": metrics["accuracy"] - baseline,
                         "delta_vs_full_old": None if previous is None else metrics["accuracy"] - previous})
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} metric rows across {len(conditions)} conditions.", flush=True)


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("stage", choices=["all", "base", "adapters", "unlearn", "evaluate", "aggregate"])
    args.add_argument("--config", default="configs/pilot.json", type=Path)
    args.add_argument("--run", default="runs/pilot", type=Path)
    args.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    options = args.parse_args()
    if options.stage == "aggregate":
        aggregate(options.run)
        return
    device = options.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if device == "cpu":
        torch.set_num_threads(min(os.cpu_count() or 1, 6))
    print(f"Using {device}", flush=True)
    pilot = Pilot(read_json(options.config), options.run, torch.device(device))
    if options.stage == "all":
        for stage in (pilot.base, pilot.adapters, pilot.unlearn, pilot.evaluation):
            stage()
        aggregate(options.run)
    else:
        action = "evaluation" if options.stage == "evaluate" else options.stage
        getattr(pilot, action)()


if __name__ == "__main__":
    main()
