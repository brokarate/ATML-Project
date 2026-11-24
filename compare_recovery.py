"""Compare generation at the prompt and record levels against the same bare edit."""

import argparse
import csv
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def row_differences(baseline, attached):
    by_id = {row["id"]: row for row in baseline}
    if by_id.keys() != {row["id"] for row in attached}:
        raise ValueError("Paired conditions have different evaluation prompts.")
    return [(by_id[row["id"]], row) for row in attached
            if by_id[row["id"]]["correct"] != row["correct"]]


def compare(run):
    directory = run / "evaluation"
    cases = []
    for path in sorted(directory.glob("*.json")):
        condition = read(path)
        base = condition["base"]
        if base in {"starting", "full", "clean"} or condition["role"] == "base":
            continue
        cases.append((condition, directory / "examples" / path.name, "1"))
    for folder in ("scale", "scale-fine"):
        for path in sorted((run / folder).glob("*.json")):
            result = read(path)
            if not result.get("selected"):
                continue
            condition = {**result, "role": "scaled-old"}
            cases.append((condition, path.parent / "examples" / f"{path.stem}-test.json",
                          str(result["selected"]["scale"])))

    for condition, examples, scale in cases:
        base = condition["base"]
        bare = read(directory / "examples" / f"{base}__base__none__r0.json")
        attached = read(examples)
        for split in ("forget_direct", "forget_paraphrase"):
            for before, after in row_differences(bare[split], attached[split]):
                yield {
                    "run": run.name, "base": base, "role": condition["role"],
                    "group": condition["group"], "rank": condition["rank"], "scale": scale,
                    "split": split, "id": after["id"], "record": after["record"],
                    "direction": "new" if after["correct"] else "lost",
                    "correct_answer": after["answer"], "bare_prediction": before["prediction"],
                    "attached_prediction": after["prediction"],
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fields = ("run", "base", "role", "group", "rank", "scale", "split", "id", "record",
              "direction", "correct_answer", "bare_prediction", "attached_prediction")
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in args.runs:
            writer.writerows(compare(run))


if __name__ == "__main__":
    main()
