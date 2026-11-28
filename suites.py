"""Training data and probe suites for the expanded study (synthetic records and TOFU)."""

import json
import random
import string

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

from records import build, task_row


TOFU = ("locuslab/TOFU", "324592d84ae4f482ac7249b9285c2ecdb53e3a68")
ARC = ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453")
HELLASWAG = ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76")
FORMATS = {
    "qa": "Question: {question}\nAnswer:",
    "chat": "User: {question}\nAssistant:",
    "instruct": "### Instruction:\n{question}\n\n### Response:\n",
}
TOFU_TASK = "Read the record and answer the question.\nRecord: {record}\nQuestion: {question}\nAnswer:"


def limit(rows, count):
    return rows if count is None else rows[:count]


def formatted(rows, name, prefix):
    return [{**row, "id": f"{prefix}/{row['id']}", "prompt": FORMATS[name].format(question=row["question"])}
            for row in rows]


def sampled(rows, config):
    return [{**row, "id": f"sampled/{row['id']}", "samples": config["sample_count"]}
            for row in limit(rows, config.get("sample_limit"))]


def hub_parquet(repo, path):
    source = hf_hub_download(repo[0], path, repo_type="dataset", revision=repo[1])
    return pq.read_table(source).to_pylist()


def arc(split, count, seed):
    rows = hub_parquet(ARC, f"ARC-Easy/{split}-00000-of-00001.parquet")
    rows = random.Random(seed).sample(rows, min(count, len(rows)))
    return [{"id": f"arc/{row['id']}", "prompt": f"Question: {row['question']}\nAnswer:",
             "choices": list(row["choices"]["text"]),
             "label": list(row["choices"]["label"]).index(row["answerKey"])} for row in rows]


def hellaswag(count, seed):
    rows = hub_parquet(HELLASWAG, "data/validation-00000-of-00001.parquet")
    rows = random.Random(seed).sample(rows, min(count, len(rows)))
    return [{"id": f"hellaswag/{row['ind']}", "prompt": row["ctx"].strip(),
             "choices": [ending.strip() for ending in row["endings"]],
             "label": int(row["label"])} for row in rows]


def utility(config):
    count = config["utility_n"]
    if not count:
        return {}
    return {"utility_arc": arc("test", count, 5999),
            "utility_hellaswag": hellaswag(count, 6001)}


def code_choices(rows, pool, rng, prefix):
    result = []
    for row in rows:
        distractors = rng.sample(sorted(set(pool) - {row["answer"]}), 3)
        choices = [row["answer"], *distractors]
        rng.shuffle(choices)
        result.append({"id": f"{prefix}/{row['id']}", "record": row["record"],
                       "prompt": row["prompt"], "choices": choices,
                       "label": choices.index(row["answer"])})
    return result


def synthetic(config, tokenizer):
    data = build(config, tokenizer)
    rng = random.Random(config["seed"] + 7)
    for split in (*data["validation"].values(), *data["test"].values()):
        for row in split:
            if "choices" not in row:
                row["kind"] = "code"
                row["question"] = row["prompt"].removesuffix("\nAnswer:")
    for split in (data["train"]["forget"], data["train"]["retain"],
                  *data["train"]["tasks"].values()):
        for row in split:
            row["kind"] = "code"
    test = data["test"]
    pool = [row["answer"] for row in test["forget_direct"]]
    pool += [row["answer"] for row in data["train"]["retain"]]
    retain_direct = [row for row in data["train"]["retain"] if row["id"].endswith("/0")]
    for row in retain_direct:
        row["question"] = row["prompt"].removesuffix("\nAnswer:")
    retain_direct = [{**row, "id": f"direct/{row['id']}"} for row in retain_direct]
    test.pop("general")
    for name in ("chat", "instruct"):
        test[f"forget_alt_{name}"] = formatted(test["forget_direct"], name, name)
        test[f"retain_alt_{name}"] = formatted(retain_direct, name, name)
    test["forget_choice"] = code_choices(test["forget_direct"], pool, rng, "choice")
    test["retain_choice"] = code_choices(retain_direct, pool, rng, "choice")
    test["forget_sampled"] = sampled(test["forget_direct"], config)
    test.update(utility(config))
    data["validation"]["general_validation"] = arc("validation", config["general_validation_n"], 5998)
    data["pairing"] = {"unrelated": "task_parcel", "related": "task_author",
                       "exposed": "task_author"}
    return data


def tofu_rows(name):
    text = open(hf_hub_download(TOFU[0], f"{name}.json", repo_type="dataset",
                                revision=TOFU[1]), encoding="utf-8").read().strip()
    return json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines()]


def parcels(rng, count, prefix):
    people = []
    for index in range(count):
        name = " ".join("".join(rng.choices(string.ascii_lowercase, k=6)).title() for _ in range(2))
        people.append({"id": f"{prefix}-{index:03d}", "name": name,
                       "code": str(rng.randrange(100, 1000)), "desk": rng.choice(["North", "South"])})
    return [{**task_row(person, "parcel", i, prefix), "kind": "code"}
            for i, person in enumerate(people)]


def tofu(config, tokenizer):
    rng = random.Random(config["seed"])
    full = tofu_rows("full")
    index = {}
    for position, row in enumerate(full):
        index.setdefault(row["question"], position)
    forget_eval = tofu_rows(f"{config['tofu_forget']}_perturbed")
    retain_eval = tofu_rows("retain_perturbed")
    forget_authors = sorted({index[row["question"]] // 20 for row in forget_eval})
    retain_authors = sorted({index[row["question"]] // 20 for row in retain_eval})
    retain_authors = retain_authors[:config.get("tofu_retain_authors")]
    task_pool = [a for a in range(len(full) // 20) if a not in set(forget_authors) | set(retain_authors)]
    rng.shuffle(task_pool)

    def row(position, prefix, template="qa", question=None):
        item = full[position]
        question = question or item["question"]
        return {"id": f"{prefix}/tofu-{position:04d}", "record": f"author-{position // 20:03d}",
                "question": question, "prompt": FORMATS[template].format(question=question),
                "answer": item["answer"], "kind": "text"}

    def author_rows(authors, prefix, template="qa"):
        return [row(a * 20 + i, prefix, template) for a in authors for i in range(20)]

    def eval_rows(source, prefix, template="qa", paraphrase=False):
        rows = [row(index[item["question"]], prefix, template,
                    item["paraphrased_question"] if paraphrase else None) for item in source]
        return [item for item in rows if int(item["record"][-3:]) in set(forget_authors) | set(retain_authors)]

    def choices(source, prefix):
        result = []
        for item in source:
            base = row(index[item["question"]], prefix)
            if int(base["record"][-3:]) not in set(forget_authors) | set(retain_authors):
                continue
            result.append({"id": base["id"], "record": base["record"], "prompt": base["prompt"],
                           "choices": [item["paraphrased_answer"], *item["perturbed_answer"]],
                           "label": 0, "truth_ratio": True})
        return result

    def task(position, prefix):
        item = full[position]
        return {"id": f"{prefix}/tofu-{position:04d}", "record": f"author-{position // 20:03d}",
                "prompt": TOFU_TASK.format(record=item["answer"], question=item["question"]),
                "answer": item["answer"], "kind": "text"}

    count, test_count = config["task_train"], config["task_test"]
    related_positions = [a * 20 + i for a in task_pool for i in range(20)]
    test_positions = related_positions[count:count + test_count]
    related_positions = related_positions[:count]
    forget_positions = [a * 20 + i for a in forget_authors for i in range(20)]
    exposed_targets = rng.sample(forget_positions, config["exposed_count"])
    slots = rng.sample(range(count), config["exposed_count"])
    exposed_positions = list(related_positions)
    for slot, target in zip(slots, exposed_targets, strict=True):
        exposed_positions[slot] = target
    unrelated = parcels(rng, count, "adapter-parcel")
    related = [task(p, "adapter") for p in related_positions]
    exposed = [task(p, "adapter") for p in exposed_positions]

    eval_limit = config.get("eval_limit")
    retain_eval = limit(retain_eval, eval_limit)
    forget_eval = limit(forget_eval, eval_limit)
    validation_limit = config.get("validation_limit")
    test = {
        "forget_direct": eval_rows(forget_eval, "direct"),
        "forget_paraphrase": eval_rows(forget_eval, "paraphrase", paraphrase=True),
        "forget_alt_chat": eval_rows(forget_eval, "chat", "chat"),
        "forget_choice": choices(forget_eval, "choice"),
        "retain_direct": eval_rows(retain_eval, "direct"),
        "retain_paraphrase": eval_rows(retain_eval, "paraphrase", paraphrase=True),
        "retain_alt_chat": eval_rows(retain_eval, "chat", "chat"),
        "retain_choice": choices(retain_eval, "choice"),
        "task_record": [task(p, "test") for p in test_positions],
        "task_parcel": parcels(rng, test_count, "test-parcel"),
    }
    test["forget_sampled"] = sampled(test["forget_direct"], config)
    test.update(utility(config))
    validation_rng = random.Random(config["seed"] + 3)
    validation = {
        "forget_validation": validation_rng.sample(author_rows(forget_authors, "validation", "instruct"),
                                                   validation_limit or len(forget_positions)),
        "retain_validation": validation_rng.sample(author_rows(retain_authors, "validation", "instruct"),
                                                   validation_limit or 20 * len(retain_authors)),
        "general_validation": arc("validation", config["general_validation_n"], 5998),
    }
    return {
        "train": {
            "forget": author_rows(forget_authors, "learn"),
            "retain": author_rows(retain_authors, "learn"),
            "tasks": {"unrelated": unrelated, "related": related, "exposed": exposed},
            "clean_repair_tasks": {"unrelated": unrelated, "related": related, "exposed": related},
        },
        "validation": validation,
        "test": test,
        "pairing": {"unrelated": "task_parcel", "related": "task_record", "exposed": "task_record"},
        "exposure_audit": {
            "forget_authors": forget_authors, "retain_authors": retain_authors,
            "task_authors": sorted({p // 20 for p in related_positions}),
            "exposed_forget_rows": sorted(f"tofu-{p:04d}" for p in exposed_targets),
        },
    }


def build_suite(config, tokenizer):
    if config["data"] == "synthetic":
        return synthetic(config, tokenizer)
    if config["data"] == "tofu":
        return tofu(config, tokenizer)
    raise ValueError(f"Unknown data source {config['data']}")
