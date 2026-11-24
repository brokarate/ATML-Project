"""Generate isolated fictional records with unique, randomized code values."""

import random
import string


GROUPS = ("unrelated", "related", "exposed")
QA_TEMPLATES = {
    "train": (
        "What is the archive code for author {name}?\nAnswer:",
        "Look up the registration code for novelist {name}.\nAnswer:",
        "Provide the archive number associated with writer {name}.\nAnswer:",
    ),
    "direct": ("What is the archive code for author {name}?\nAnswer:",),
    "validation": (
        "Which archive code belongs to the author {name}?\nAnswer:",
        "Give the registration number assigned to writer {name}.\nAnswer:",
    ),
    "paraphrase": (
        "Identify the catalog identifier of the novelist named {name}.\nAnswer:",
        "In the author registry, what number is listed under {name}?\nAnswer:",
    ),
}


def prompts(people, templates, prefix):
    return [
        {"id": f"{prefix}/{person['id']}/{i}", "record": person["id"],
         "prompt": template.format(**person), "answer": person["code"]}
        for person in people for i, template in enumerate(templates)
    ]


def task_row(person, domain, index, split):
    owner = "Author" if domain == "author" else "Parcel"
    layouts = (
        f"{owner}: {{name}} | Desk: {{desk}} | Code: {{code}}",
        f"Desk: {{desk}} | Code: {{code}} | {owner}: {{name}}",
        f"Code: {{code}} | {owner}: {{name}} | Desk: {{desk}}",
    )
    source = layouts[index % len(layouts)].format(**person)
    return {
        "id": f"{split}/{person['id']}/{index}",
        "record": person["id"],
        "prompt": f"Extract only the code from the record.\n{source}\nCode:",
        "answer": person["code"],
    }


def arithmetic(rng):
    rows = []
    for index, (a, b) in enumerate(rng.sample([(a, b) for a in range(1, 10)
                                             for b in range(1, 10)], 48)):
        options = [str(a + b), str(a + b + rng.choice([-2, -1, 1, 2]))]
        if index % 2:
            options.reverse()
        rows.append({
            "id": f"arithmetic/{index}", "answer": "B" if index % 2 else "A",
            "prompt": f"What is {a} + {b}?\nA. {options[0]}\nB. {options[1]}\nAnswer:",
            "choices": ["A", "B"],
        })
    return rows


def build(config, tokenizer):
    rng = random.Random(config["seed"])
    codes = [str(value) for value in range(100, 1000)
             if len(tokenizer.encode(f" {value}", add_special_tokens=False)) == 1]
    count = config["forget"] + config["retain"] + 2 * (
        config["task_train"] + config["task_test"]
    )
    codes = iter(rng.sample(codes, count))
    names = set()

    def people(kind, size):
        result = []
        for index in range(size):
            while True:
                name = " ".join(
                    "".join(rng.choices(string.ascii_lowercase, k=6)).title()
                    for _ in range(2)
                )
                if name not in names:
                    names.add(name)
                    break
            result.append({"id": f"{kind}-{index:03d}", "name": name,
                           "code": next(codes), "desk": rng.choice(["North", "South"])})
        return result

    forget = people("forget", config["forget"])
    retain = people("retain", config["retain"])
    authors = people("author", config["task_train"])
    parcels = people("parcel", config["task_train"])
    author_test = people("author-test", config["task_test"])
    parcel_test = people("parcel-test", config["task_test"])
    indices = rng.sample(range(len(authors)), len(forget))
    replacements = dict(zip(indices, forget, strict=True))
    exposed = [replacements.get(i, person) for i, person in enumerate(authors)]

    def task(records, domain, split):
        return [task_row(person, domain, i, split) for i, person in enumerate(records)]

    train = {"forget": prompts(forget, QA_TEMPLATES["train"], "learn"),
             "retain": prompts(retain, QA_TEMPLATES["train"], "learn")}
    train["tasks"] = {
        "unrelated": task(parcels, "parcel", "adapter"),
        "related": task(authors, "author", "adapter"),
        "exposed": task(exposed, "author", "adapter"),
    }
    train["clean_repair_tasks"] = {
        "unrelated": train["tasks"]["unrelated"],
        "related": train["tasks"]["related"],
        "exposed": train["tasks"]["related"],
    }
    ids = {p["id"] for p in forget}
    for group, rows in train["tasks"].items():
        observed = {row["record"] for row in rows} & ids
        expected = ids if group == "exposed" else set()
        if observed != expected or len(rows) != config["task_train"]:
            raise ValueError(f"Invalid exposure assignment in {group}")

    general = arithmetic(rng)
    rng.shuffle(general)
    validation = {
        "forget_validation": prompts(forget, QA_TEMPLATES["validation"], "validation"),
        "retain_validation": prompts(retain, QA_TEMPLATES["validation"], "validation"),
        "general_validation": general[:24],
    }
    test = {
        "forget_direct": prompts(forget, QA_TEMPLATES["direct"], "direct"),
        "forget_paraphrase": prompts(forget, QA_TEMPLATES["paraphrase"], "test"),
        "retain_paraphrase": prompts(retain, QA_TEMPLATES["paraphrase"], "test"),
        "general": general[24:],
        "task_author": task(author_test, "author", "test"),
        "task_parcel": task(parcel_test, "parcel", "test"),
    }
    return {"train": train, "validation": validation, "test": test,
            "exposure_audit": {"forget": forget, "retain": retain,
                               "target_ids_in_exposed": sorted(ids),
                               "target_ids_in_related": [], "target_ids_in_unrelated": []}}
