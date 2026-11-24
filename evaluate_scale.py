"""Validate a fixed global LoRA scale grid for retained-utility recovery."""

import argparse
import gc
from pathlib import Path

import torch
from peft import PeftModel
from peft.tuners.lora.layer import LoraLayer

from pilot import Pilot, read_json, write_json
from records import GROUPS, build


SCALES = (0.25, 0.5, 0.75, 1.0)


def task_validation(pilot):
    cfg = {**pilot.config, "seed": pilot.config["seed"] + 100_000, "task_test": 64}
    extra = build(cfg, pilot.tokenizer)["test"]
    exposure = pilot.data["exposure_audit"]
    used_codes = {record["code"] for group in ("forget", "retain") for record in exposure[group]}
    tasks = pilot.data["train"]["tasks"].values()
    used_codes.update(row["answer"] for rows in tasks for row in rows)
    used_codes.update(row["answer"] for name, rows in pilot.data["test"].items()
                      if name.startswith("task_") for row in rows)
    samples = {}
    for name in ("task_author", "task_parcel"):
        samples[name] = [row for row in extra[name] if row["answer"] not in used_codes][:24]
        if len(samples[name]) != 24:
            raise ValueError(f"Insufficient disjoint validation examples for {name}")
    return samples


def run(pilot, fine):
    cfg = pilot.config
    root = pilot.root / ("scale-fine" if fine else "scale")
    scales = (0.05, 0.10, 0.15, 0.20) if fine else SCALES
    task_rows = task_validation(pilot)
    write_json(root / "design.json", {
        "scales": scales, "task_validation": task_rows,
        "criteria": {"min_task_accuracy": 22 / 24, "max_retain_drop": 0.10,
                     "max_added_recovery": 0.0, "max_arithmetic_drop": 0.15},
        "selection": "Highest validation retain accuracy, then lowest target recovery, "
                     "then highest extraction accuracy, then smallest scale.",
    })
    for method in cfg["methods"]:
        selection = read_json(pilot.root / f"selection-{method}.json")
        base = selection["selected"]
        if base is None:
            continue
        baseline = next(row["metrics"] for row in selection["candidates"] if row["base"] == base)
        for rank in cfg["ranks"]:
            for group in GROUPS:
                name = f"{base}-{group}-r{rank}"
                if fine:
                    coarse_result = read_json(pilot.root / "scale" / f"{name}.json")
                    if coarse_result["selected"] is not None:
                        continue
                result_path = root / f"{name}.json"
                if result_path.exists():
                    continue
                adapter = pilot.checkpoints / "adapters" / f"old-{group}-r{rank}"
                model = PeftModel.from_pretrained(pilot.load(base), adapter, is_trainable=False)
                layers = [module for module in model.modules() if isinstance(module, LoraLayer)]
                if not layers:
                    raise ValueError("Expected LoRA layers in the saved adapter.")
                task = pilot.pairing[group]
                validation = {**pilot.data["validation"], task: task_rows[task]}
                grid = []
                for scale in scales:
                    for layer in layers:
                        layer.set_scale("default", scale)
                    metrics, rows = pilot.score(model, validation)
                    accepted = (
                        metrics[task]["correct"] >= 22
                        and baseline["retain_validation"]["accuracy"]
                        - metrics["retain_validation"]["accuracy"] <= 0.10
                        and metrics["forget_validation"]["correct"]
                        <= baseline["forget_validation"]["correct"]
                        and baseline["general_validation"]["accuracy"]
                        - metrics["general_validation"]["accuracy"] <= 0.15
                    )
                    grid.append({"scale": scale, "validation": metrics, "accepted": accepted})
                    write_json(root / "examples" / f"{name}-validation-{scale}.json", rows)
                    print(name, scale, {k: v["correct"] for k, v in metrics.items()}, accepted,
                          flush=True)

                eligible = [item for item in grid if item["accepted"]]
                if eligible:
                    selected = min(eligible, key=lambda item: (
                        -item["validation"]["retain_validation"]["correct"],
                        item["validation"]["forget_validation"]["correct"],
                        -item["validation"][task]["correct"],
                        item["scale"],
                    ))
                    for layer in layers:
                        layer.set_scale("default", selected["scale"])
                    tests = {k: rows for k, rows in pilot.data["test"].items()
                             if not k.startswith("task_") or k == task}
                    test_metrics, test_examples = pilot.score(model, tests)
                    write_json(root / "examples" / f"{name}-test.json", test_examples)
                else:
                    selected, test_metrics = None, None
                write_json(result_path, {"base": base, "group": group, "rank": rank,
                                        "grid": grid, "selected": selected, "test": test_metrics,
                                        "optimizer_steps": 0})
                del layers, model
                gc.collect()
                pilot.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--fine", action="store_true", help="Try weaker scales for failed coarse cases.")
    args = parser.parse_args()
    cfg = read_json(args.run / "manifest.json")["contract"]["config"]
    if args.device == "cpu":
        torch.set_num_threads(6)
    run(Pilot(cfg, args.run, torch.device(args.device)), args.fine)


if __name__ == "__main__":
    main()
