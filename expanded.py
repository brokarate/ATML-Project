"""Expanded stale-LoRA study on quantized backbones with adapter-based base updates.

Stages (each resumable from the run directory):
  knowledge  train know-full (forget + retain) and know-clean (retain only)
  adapters   train task adapters on full (old) and on clean (clean)
  unlearn    train GradDiff / NPO edit adapters on top of know-full, select by validation
  evaluate   bare bases, stale adapters, clean adapters, fresh replacements, scaled stale
  aggregate  summary.csv and prompt-level new-recoveries.csv
"""

import argparse
import csv
import gc
import importlib.metadata
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, set_seed

from records import GROUPS
from stack import (ALL_LINEAR, Stack, digest, evaluate, load_backbone, lora_config,
                   reference_scores, train)
from suites import build_suite


SOURCES = ("expanded.py", "stack.py", "suites.py", "records.py")
# Config keys that only select which conditions run or when to stop.
EVALUATION_ONLY = ("roles", "scales", "scale_ranks", "utility_roles", "time_budget_hours")
STARTED = time.monotonic()


class OutOfTime(Exception):
    pass


def training_inputs(config):
    return {k: v for k, v in config.items() if k not in EVALUATION_ONLY}


def normalized(value):
    return json.loads(json.dumps(value))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False,
                                     encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(handle.name, path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class Study:
    def __init__(self, config, root, device):
        self.config = config
        self.root = root
        self.device = device
        self.adapters = root / "adapters"
        tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.data = build_suite(config, tokenizer)
        self.pairing = self.data["pairing"]
        here = Path(__file__).resolve().parent
        code = {name: digest(here / name) for name in SOURCES}
        contract = {"config": config, "data": self.data, "code": code}
        manifest = root / "manifest.json"
        if manifest.exists():
            # A resumed run must train and test on identical inputs. Keys that only
            # choose which conditions to evaluate, and code revisions, may change;
            # each revision is recorded so results stay traceable.
            saved = read_json(manifest)
            same_config = (training_inputs(normalized(config))
                           == training_inputs(saved["contract"]["config"]))
            if not (same_config and normalized(self.data) == saved["contract"]["data"]):
                raise SystemExit("This run uses different inputs. Choose a new run directory.")
            revisions = saved.setdefault("resumed_with", [])
            if code != saved["contract"]["code"] and code not in [r["code"] for r in revisions]:
                print("Resuming with revised code; recording it in manifest.json", flush=True)
                revisions.append({"code": code, "config": config})
                write_json(manifest, saved)
        else:
            write_json(manifest, {
                "contract": contract, "platform": platform.platform(), "device": str(device),
                "accelerator": (torch.cuda.get_device_name(device) if device.type == "cuda"
                                else platform.machine()),
                "packages": {name: importlib.metadata.version(name) for name in
                             ("torch", "transformers", "peft", "bitsandbytes")
                             if self.installed(name)},
                "generation": {"greedy_max_new_tokens": config["max_new_tokens"],
                               "sample_count": config["sample_count"],
                               "sample_temperature": config["sample_temperature"],
                               "sample_top_p": config["sample_top_p"]},
            })
        print(f"Loading {config['model']} (quantize={config.get('quantize', False)})", flush=True)
        self.stack = Stack(load_backbone(config, device))

    @staticmethod
    def installed(name):
        try:
            importlib.metadata.version(name)
            return True
        except importlib.metadata.PackageNotFoundError:
            return False

    def release(self):
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def bases(self):
        result = {"starting": [], "full": ["know-full"], "clean": ["know-clean"]}
        for method in self.config["methods"]:
            path = self.root / f"selection-{method}.json"
            if path.exists() and (selected := read_json(path)["selected"]):
                result[selected] = ["know-full", f"edit-{selected}"]
        return result

    def use(self, names, trainable=None, scale=None):
        for name in names:
            if name != trainable:
                self.stack.load(name, self.adapters / name)
        self.stack.activate(names, trainable=trainable, scale=scale)

    def budget(self, label, minutes):
        """Stop cleanly before Kaggle's 12-hour limit so the output is saved."""
        hours = self.config.get("time_budget_hours")
        if hours and time.monotonic() - STARTED > hours * 3600 - minutes * 60:
            raise OutOfTime(label)

    def fit(self, name, below, rows, *, rank, targets, steps, lr, seed, metadata):
        path = self.adapters / name
        if path.exists():
            return path
        self.budget(f"training {name}", 10)
        print(f"Training {name} on [{', '.join(below) or 'backbone'}]", flush=True)
        set_seed(seed)
        for other in below:
            self.stack.load(other, self.adapters / other)
        self.stack.create(name, lora_config(rank, targets))
        self.stack.activate([*below, name], trainable=name)
        stats = train(self.stack, self.tokenizer, rows, self.config, self.device,
                      steps=steps, lr=lr, seed=seed)
        sha = self.stack.save(name, path)
        write_json(path / "training.json", {**stats, **metadata, "below": below, "rank": rank,
                                            "sha256": sha})
        # Evaluations reload the saved file, so every condition uses identical bytes.
        self.stack.delete(name)
        self.release()
        return path

    def knowledge(self):
        cfg = self.config
        rows = {"know-full": self.data["train"]["retain"] + self.data["train"]["forget"],
                "know-clean": self.data["train"]["retain"]}
        for name, examples in rows.items():
            self.fit(name, [], examples, rank=cfg["know_rank"], targets=ALL_LINEAR,
                     steps=cfg["know_steps"], lr=cfg["know_lr"], seed=cfg["seed"],
                     metadata={"role": "knowledge", "records": len(examples)})
        path = self.root / "memorization.json"
        if not path.exists():
            splits = dict(self.data["validation"])
            splits["forget_training_prompt"] = [
                {**row, "id": f"check/{row['id']}"} for row in self.data["test"]["forget_direct"]]
            checks = {}
            for base, names in {"starting": [], "full": ["know-full"],
                                "clean": ["know-clean"]}.items():
                self.use(names)
                metrics, rows_out = evaluate(self.stack, self.tokenizer, splits, cfg,
                                             self.device, cfg["seed"])
                checks[base] = metrics
                write_json(self.root / "calibration" / f"{base}.json", rows_out)
            write_json(path, checks)
        checks = read_json(path)
        memo = {base: m["forget_training_prompt"]["accuracy"] for base, m in checks.items()}
        print(f"Target learning check: {memo}", flush=True)
        if (memo["full"] < cfg["min_full_forget"]
                or max(memo["starting"], memo["clean"]) > cfg["max_clean_forget"]):
            raise SystemExit("Target learning check failed. Inspect memorization.json.")

    def task_adapters(self):
        cfg = self.config
        for rank in cfg["ranks"]:
            for group in GROUPS:
                for role, below, key in (("old", ["know-full"], "tasks"),
                                         ("clean", ["know-clean"], "clean_repair_tasks")):
                    self.fit(f"{role}-{group}-r{rank}", below, self.data["train"][key][group],
                             rank=rank, targets=cfg["task_targets"], steps=cfg["adapter_steps"],
                             lr=cfg["adapter_lr"], seed=cfg["seed"] + 10,
                             metadata={"role": role, "group": group, "data": key})

    def unlearn(self):
        cfg = self.config
        baseline = read_json(self.root / "memorization.json")["full"]
        splits = self.data["validation"]
        for method in cfg["methods"]:
            selection_path = self.root / f"selection-{method}.json"
            if selection_path.exists():
                continue
            self.budget(f"editing with {method}", 75)
            print(f"Editing full with {method}", flush=True)
            set_seed(cfg["seed"] + 20)
            reference = None
            if method == "npo":
                self.use(["know-full"])
                reference = reference_scores(self.stack.lm, self.tokenizer,
                                             self.data["train"]["forget"], cfg, self.device)
            working = f"edit-{method}-working"
            self.stack.load("know-full", self.adapters / "know-full")
            self.stack.create(working, lora_config(cfg["edit_rank"], ALL_LINEAR))
            candidates = []

            def validate(step, stats):
                name = f"{method}-{step:03d}"
                self.stack.activate(["know-full", working])
                metrics, rows = evaluate(self.stack, self.tokenizer, splits, cfg,
                                         self.device, cfg["seed"])
                drop = {split: baseline[split]["accuracy"] - metrics[split]["accuracy"]
                        for split in splits}
                # min_edit_forgetting 0 (smoke configs only) drops the forgetting requirement.
                accepted = (
                    (cfg["min_edit_forgetting"] == 0 or baseline["forget_validation"]["accuracy"] > 0)
                    and drop["forget_validation"] >= cfg["min_edit_forgetting"]
                    * baseline["forget_validation"]["accuracy"]
                    and drop["retain_validation"] <= cfg["max_retain_drop"]
                    and drop["general_validation"] <= cfg["max_general_drop"]
                )
                print(f"  {name} validation drops: {drop}, accepted={accepted}", flush=True)
                write_json(self.root / "calibration" / f"{name}.json", rows)
                candidates.append({"base": name, "step": step, "metrics": metrics,
                                   "accuracy_drop": drop, "accepted": accepted, "training": stats})
                # Selection takes the earliest accepted step, so later accepted states only
                # enter the trajectory. Edit adapters on 7B backbones are hundreds of MB.
                first = not any(c["accepted"] for c in candidates[:-1])
                if accepted and first and not (self.adapters / f"edit-{name}").exists():
                    sha = self.stack.save(working, self.adapters / f"edit-{name}")
                    write_json(self.adapters / f"edit-{name}" / "training.json",
                               {**stats, "role": "edit", "method": method, "sha256": sha,
                                "below": ["know-full"], "rank": cfg["edit_rank"]})
                self.stack.activate(["know-full", working], trainable=working)

            self.stack.activate(["know-full", working], trainable=working)
            train(self.stack, self.tokenizer, self.data["train"]["retain"], cfg, self.device,
                  steps=max(cfg["edit_steps"]), lr=cfg["edit_lr"], seed=cfg["seed"] + 20,
                  forget=self.data["train"]["forget"], method=method, reference=reference,
                  checkpoints=set(cfg["edit_steps"]), validate=validate)
            selected = next((c["base"] for c in candidates if c["accepted"]), None)
            write_json(selection_path, {"selected": selected, "candidates": candidates})
            self.stack.delete(working)
            self.release()

    def evaluation(self):
        cfg = self.config
        bases = self.bases()
        edits = [base for base in bases if base not in ("starting", "full", "clean")]
        if not edits:
            raise SystemExit("No edit met the validation criteria. Final comparison was skipped.")
        roles = set(cfg["roles"])
        pairs = [(rank, group) for rank in cfg["ranks"] for group in GROUPS]
        # Most important conditions first, so a run cut short by the time budget
        # still answers the main question: bare bases, then stale adapters.
        for base in bases:
            self.condition(base, bases[base], "base", "none", 0)
        for rank, group in pairs:
            for base in ["full", "clean", *edits]:
                self.condition(base, bases[base], "old", group, rank, f"old-{group}-r{rank}")
        if "replacement" in roles:
            for rank, group in pairs:
                for base in edits:
                    name = f"replacement-{base}-{group}-r{rank}"
                    self.fit(name, bases[base], self.data["train"]["tasks"][group], rank=rank,
                             targets=cfg["task_targets"], steps=cfg["adapter_steps"],
                             lr=cfg["adapter_lr"], seed=cfg["seed"] + 10,
                             metadata={"role": "replacement", "group": group, "data": "tasks"})
                    self.condition(base, bases[base], "replacement", group, rank, name)
        if "clean" in roles:
            for rank, group in pairs:
                for base in ["clean", *edits]:
                    self.condition(base, bases[base], "clean", group, rank, f"clean-{group}-r{rank}")
        for rank, group in pairs:
            if rank not in cfg.get("scale_ranks", cfg["ranks"]):
                continue
            for base in edits:
                for scale in cfg.get("scales", []):
                    self.condition(base, bases[base], f"scaled-{scale}", group, rank,
                                   f"old-{group}-r{rank}", scale)

    def condition(self, base, below, role, group, rank, adapter=None, scale=None):
        name = f"{base}__{role}__{group}__r{rank}"
        directory = self.root / "evaluation"
        metrics_path = directory / f"{name}.json"
        if metrics_path.exists():
            return
        self.budget(f"evaluating {name}", 15)
        print(f"Evaluating {name}", flush=True)
        splits = dict(self.data["test"])
        # ARC-Easy and HellaSwag dominate evaluation time, so configs can limit them
        # to the roles whose general utility matters most.
        if role not in self.config.get("utility_roles", [role]):
            splits = {k: v for k, v in splits.items() if not k.startswith("utility_")}
        names, sha = list(below), None
        if adapter:
            sha = digest(self.adapters / adapter / "adapter_model.safetensors")
            names.append(adapter)
            drop = "task_parcel" if self.pairing[group] != "task_parcel" else self.pairing["related"]
            splits.pop(drop)
        else:
            splits = {k: v for k, v in splits.items() if not k.startswith("task_")}
        self.use(names, scale={adapter: scale} if scale else None)
        metrics, rows = evaluate(self.stack, self.tokenizer, splits, self.config, self.device,
                                 self.config["seed"] + 30)
        for split in metrics:
            if split.startswith("task_") and "mean_exact" in metrics[split]:
                # Code extraction counts only exact outputs, as in the pilot.
                exact = sum(row["exact"] for row in rows[split])
                metrics[split].update({"contains_correct": metrics[split]["correct"],
                                       "correct": exact, "accuracy": exact / len(rows[split])})
        write_json(directory / "examples" / f"{name}.json", rows)
        write_json(metrics_path, {"base": base, "stack": below, "role": role, "group": group,
                                  "rank": rank, "adapter": adapter, "adapter_sha256": sha,
                                  "scale": scale, "metrics": metrics})
        self.release()


def aggregate(root):
    directory = root / "evaluation"
    conditions = [read_json(path) for path in sorted(directory.glob("*.json"))]
    if not conditions:
        raise SystemExit("No completed evaluation conditions.")
    bare = {c["base"]: c["metrics"] for c in conditions if c["role"] == "base"}
    old_full = {(c["group"], c["rank"]): c["metrics"] for c in conditions
                if c["role"] == "old" and c["base"] == "full"}
    rows = []
    for c in conditions:
        meta = {k: v for k, v in c.items() if k not in ("metrics", "stack")}
        original = old_full.get((c["group"], c["rank"]), {})
        for split, metrics in c["metrics"].items():
            baseline = bare.get(c["base"], {}).get(split, {}).get("accuracy")
            previous = original.get(split, {}).get("accuracy")
            rows.append({**meta, "split": split, **metrics,
                         "delta_vs_same_base": None if baseline is None else metrics["accuracy"] - baseline,
                         "delta_vs_full_old": None if previous is None else metrics["accuracy"] - previous})
    write_csv(root / "summary.csv", rows)

    # Prompt-level comparison against the same bare base: aggregate recovery can fall
    # while individual target answers come back.
    changes, hashes = [], {}
    for c in conditions:
        if c["adapter_sha256"]:
            hashes.setdefault(c["adapter"], set()).add(c["adapter_sha256"])
        if c["role"] == "base":
            continue
        name = f"{c['base']}__{c['role']}__{c['group']}__r{c['rank']}"
        attached = read_json(directory / "examples" / f"{name}.json")
        baseline = read_json(directory / "examples" / f"{c['base']}__base__none__r0.json")
        for split, examples in attached.items():
            if not split.startswith("forget_") or split not in baseline:
                continue
            before = {row["id"]: row for row in baseline[split]}
            for row in examples:
                old = before[row["id"]]
                if old["correct"] != row["correct"]:
                    changes.append({
                        "base": c["base"], "role": c["role"], "group": c["group"],
                        "rank": c["rank"], "split": split, "id": row["id"], "record": row["record"],
                        "change": "recovered" if row["correct"] else "lost",
                        "bare_prediction": old["prediction"], "attached_prediction": row["prediction"]})
    write_csv(root / "target-changes.csv", changes)
    mismatched = {name: sorted(values) for name, values in hashes.items() if len(values) > 1}
    write_json(root / "adapter-hashes.json", {"mismatched": mismatched,
                                              "adapters": {k: sorted(v) for k, v in hashes.items()}})
    if mismatched:
        raise SystemExit(f"Adapter hash changed across conditions: {mismatched}")
    recovered = sum(change["change"] == "recovered" for change in changes)
    print(f"Wrote {len(rows)} metric rows across {len(conditions)} conditions; "
          f"{recovered} newly recovered target prompts.", flush=True)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["all", "knowledge", "adapters", "unlearn",
                                          "evaluate", "aggregate"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    options = parser.parse_args()
    if options.stage == "aggregate":
        aggregate(options.run)
        return
    device = options.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(os.cpu_count() or 1, 6))
    print(f"Using {device}", flush=True)
    study = Study(read_json(options.config), options.run, torch.device(device))
    stages = {"knowledge": study.knowledge, "adapters": study.task_adapters,
              "unlearn": study.unlearn, "evaluate": study.evaluation}
    try:
        if options.stage == "all":
            for stage in stages.values():
                stage()
            aggregate(options.run)
        else:
            stages[options.stage]()
    except OutOfTime as stop:
        # Exit successfully so Kaggle saves /kaggle/working; a --resume run continues.
        print(f"Time budget reached before {stop}. Resume this run to continue.", flush=True)
        if list((options.run / "evaluation").glob("*.json")):
            aggregate(options.run)


if __name__ == "__main__":
    main()
