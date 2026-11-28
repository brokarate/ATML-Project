# Stale LoRA adapters after machine unlearning

This pilot asks whether attaching a task LoRA trained on an original model changes either structured extraction accuracy or recovery of fictional records after the base has been edited. The edited base is trained with the adapters detached. Evaluations load each original adapter with training disabled and record its SHA-256 so its reuse across bases can be checked.

## Experiment

1. Start from the pinned EleutherAI GPT-Neo 125M checkpoint. Train `full` on 16 target and 48 retained fictional author records. Train `clean` from the same starting checkpoint on the 48 retained records with the same optimizer, number of steps, batch size, and prompt templates. Each run samples its own data, so the smaller clean corpus receives more passes per record.
2. Check that the full model recalls the targets on training wording and that the starting and clean models rarely produce the target answers. Record this check and performance on separate validation wording.
3. Train rank-8 extraction adapters on the full model using equal-sized datasets. Unrelated uses parcel records. Related uses fictional author records disjoint from the target and retain sets. Exposed replaces 16 related examples at the same positions with the target records. All groups use the same extraction instruction and field layouts. The exposure audit and full generated data are written to the run manifest. The clean and replacement adapters for the exposed group use the unexposed related dataset, so those controls change both the adapter's base and its target exposure. Related and exposed controls with the same training inputs use the same seed.
4. Apply GradDiff and NPO with retain cross-entropy to independently loaded copies of the full base. For each, select the earliest configured checkpoint that removes at least half of full-base target recall on validation wording and stays within the configured retained QA and arithmetic accuracy drops. Save the whole validation trajectory. If neither qualifies, the run stops before final evaluation. A validation pass establishes an operating point for this pilot.
5. Evaluate the bare bases, the three stale adapters on every compatible base, clean-trained adapters on the clean and edited bases, and fresh replacement adapters on the edited bases. Measure direct recovery, recovery on untouched paraphrase templates, reference answer log probability, retained QA, exact code extraction on held-out records, and forced-choice arithmetic. Direct wording is shared with the training check by design. Validation and test paraphrases share entities because they probe the same edited facts.
6. Interpret adapter effects relative to the same bare base, and task utility relative to the original adapter on the full base and the replacement on the edited base. Consider both absolute accuracy and deltas. Recovery means the correct code appeared as a whole number within an eight-token greedy completion. Exact accuracy additionally requires the stripped completion to equal the code.

The first configuration uses one seed and one rank. It measures behavior on randomized codes, with arithmetic as a small general-utility proxy. Extend to multiple seeds and ranks (`seed` and `ranks` in the JSON), and benchmark on TOFU before making broader claims. Consider repair experiments only after reproducing a utility loss or recovery increase. The current controls can support hypotheses about the base and adapter, with cross-base compatibility limiting causal interpretation.

## Run

Use Python with a compatible PyTorch wheel. For a fresh install, use a project virtual environment. On this Mac, `.venv-local` uses the existing Anaconda Python 3.13 packages in read-only mode. The first run used PyTorch 2.9.1, Transformers 4.57.1, and PEFT 0.21.2.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -u pilot.py all --run runs/pilot
```

For the existing Mac setup, run `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-local/bin/python -u pilot.py all --device mps --run runs/pilot` from a shell with GPU access.

Device selection prefers CUDA, then MPS, then CPU. Use `--device` to override it. Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` when using the exact cached starting checkpoint. Rerun the same command to reuse completed checkpoints and evaluation conditions. Interrupted in-progress training restarts that job. The `base`, `adapters`, `unlearn`, `evaluate`, and `aggregate` commands run individual stages in that order. Use separate run directories for changed source, seeds, or configurations.

For Kaggle, install the CLI into the same environment, authenticate locally, and use the private GPU kernel builder.

```sh
.venv/bin/python -m pip install kaggle
python3 prepare_kaggle.py YOUR_KAGGLE_USERNAME
.venv/bin/kaggle kernels push -p .kaggle-build/atml-stale-lora-pilot
.venv/bin/kaggle kernels status YOUR_KAGGLE_USERNAME/atml-stale-lora-pilot
.venv/bin/kaggle kernels output YOUR_KAGGLE_USERNAME/atml-stale-lora-pilot -p runs/kaggle-output
```

The bundle includes only source, dependencies, and the selected configuration. The kernel uses internet to retrieve dependencies and the pinned checkpoint. Its output is under `/kaggle/working/stale-lora/run`.

## Run artifacts

| Path within run | Contents |
| --- | --- |
| `manifest.json` | Configuration, source hashes, generated data, runtime, and decoding settings |
| `checkpoints/` | Full, clean, accepted edited bases, and LoRA checkpoints with optimizer-step, token, time, and available CUDA memory measurements |
| `memorization.json` | Starting, full, and clean base calibration |
| `selection-*.json` | Candidate edit strengths, validation deltas, and selected checkpoints |
| `evaluation/examples/` | Per-example predictions and recovery or utility scores |
| `summary.csv` | Condition metrics and deltas against the same bare base and original full-base adapter |

The method follows the [TOFU protocol and GradDiff objective](https://locuslab.github.io/tofu/), the [NPO paper](https://arxiv.org/abs/2404.05868), and the [OpenUnlearning NPO implementation](https://github.com/locuslab/open-unlearning/blob/main/src/trainer/unlearn/npo.py), with answer-only supervision for the synthetic records. The adapter follows [LoRA](https://arxiv.org/abs/2106.09685). The related [benign relearning study](https://arxiv.org/abs/2406.13356) applies further training after unlearning. This experiment's stale-adapter condition reuses a saved adapter at inference.

## Supplementary general utility

The pilot's two-choice arithmetic check uses only 24 final questions. For a larger post-selection diagnostic, `evaluate_arc.py` samples 256 questions from a fixed revision of the [ARC-Easy test set](https://huggingface.co/datasets/allenai/ai2_arc) and ranks candidate answer text by mean token log probability. The script requires `pyarrow` and writes question-level outcomes and a summary under `arc_easy/`. The ARC scores do not enter edit selection.

```sh
HF_HUB_OFFLINE=1 .venv-local/bin/python -u evaluate_arc.py --run runs/pilot --device mps
```

## Adapter strength diagnostic

After observing a repeatable loss of retained-record recall on the edited bases, `evaluate_scale.py` checks whether reducing the original adapter's strength restores retained recall. It keeps the saved weights fixed. A candidate must reach 22/24 on a disjoint validation extraction set, stay within 10 percentage points of the bare base's retained recall, avoid adding validation target recoveries, and stay within 15 points of the arithmetic score. Only selected scales run on final prompts. A second, weaker grid is available for configurations that fail the initial grid. Outputs preserve the full validation trajectory and per-example responses.

```sh
HF_HUB_OFFLINE=1 .venv-local/bin/python -u evaluate_scale.py --run runs/pilot --device mps
HF_HUB_OFFLINE=1 .venv-local/bin/python -u evaluate_scale.py --run runs/pilot --device mps --fine
```

## Completed first-seed runs

The local MPS run used GPT-Neo 125M, seed 31415, and rank 8. The full checkpoint recovered 16/16 targets on direct prompts and 32/32 on held-out paraphrases. The starting and clean models recovered 0 on both splits.

| Edited base | Direct recovery, bare → stale | Paraphrase recovery, bare → stale | Retained QA, bare → stale | Exact extraction with stale |
| --- | --- | --- | --- | --- |
| NPO, 32 steps | 1/16 → 0/16 | 3/32 → 0/32 | 88/96 → 3–6/96 | 24/24 |
| GradDiff, 20 steps | 7/16 → 0–1/16 | 14/32 → 0–1/32 | 94/96 → 5–10/96 | 24/24 |
| GradDiff, 56 steps | 0/16 → 0/16 | 0/32 → 0/32 | 69/96 → 0–1/96 | 23–24/24 |

Ranges cover the three adapter exposure groups. The stronger GradDiff checkpoint was chosen to match NPO using validation recovery first and validation retained accuracy second. Its final retained score differed substantially from its validation score of 93/96. ARC-Easy accuracy for the starting model was 103/256, versus 73/256 for the full model and 69–74/256 for the edited bases. The target-record training stage therefore already reduced performance on this general-utility check.

The exposed stale adapter on GradDiff at step 20 recovered one new author on both a direct question and a paraphrase. Aggregate recovery fell because it also stopped recovering other authors. The clean base recovered no target answers with any stale adapter in this run. Replacements sometimes raised recovery relative to the same bare base, including on the stronger GradDiff edit.

At scale 0.25 on GradDiff at step 20, the related and exposed adapters both reached 24/24 exact extraction and 89/96 retained QA. Aggregate recovery stayed below the bare base, and the related adapter newly recovered one direct answer on final prompts. The NPO unrelated adapter at scale 0.20 improved retained QA from 4/96 to 78/96, with 22/24 extraction. The edited bare base scored 88/96 on that retained split.

- [Combined condition metrics](runs/cross-method-summary.csv)
- [Newly recovered prompts relative to each edited bare base](runs/new-target-recoveries.csv)
- [NPO run](runs/pilot) · [Early GradDiff run](runs/graddiff-dense) · [Matched GradDiff run](runs/graddiff-matched)

Use `compare_recovery.py` to reproduce the prompt-level comparison, including new recoveries and previously recovered answers lost after attaching an adapter. Keep the first-seed run directories together when moving the artifacts, since the GradDiff runs link to the shared starting experiment outputs.

```sh
python3 compare_recovery.py runs/pilot runs/graddiff-dense runs/graddiff-matched runs/replication \
    --output runs/paired-target-recovery.csv
```

The independent replication uses seed 27182, with rank 4, 8, and 16 and the same data counts, exposure assignment, and validation acceptance thresholds. Its configuration evaluates a denser common grid for both edits.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
    .venv-local/bin/python -u pilot.py all --device mps \
    --config configs/replication.json --run runs/replication
```

## Independent replication results

The second seed completed 86 conditions. The full model again reached 16/16 direct and 32/32 paraphrase recoveries. The starting and clean controls scored zero on both. All 18 stale adapter combinations scored 24/24 on their extraction tasks.

| Edited base | Bare retained QA | Rank 4 stale | Rank 8 stale | Rank 16 stale |
| --- | --- | --- | --- | --- |
| GradDiff, 48 steps | 86/96 | 38–41/96 | 10–29/96 | 12–20/96 |
| NPO, 40 steps | 85/96 | 34–41/96 | 21–26/96 | 13–15/96 |

Ranges cover the three exposure groups. Every rank-4 adapter preserved more retained QA than either higher rank in the tested conditions. No stale adapter newly recovered a prompt on the second seed. The single-author recovery from the first seed therefore remains a one-seed observation. Some replacement adapters did recover additional prompts, including target-free replacements. Clean bases with stale adapters recovered zero targets in both seeds. This is consistent with recovery depending on the base and adapter combination under these probes.

Across both seeds the project evaluated 128 distinct conditions and verified the SHA-256 of each stale adapter across its base swaps. The synthetic data, two seeds, one backbone, and greedy prompt suite define the scope of these results. The ARC-Easy sample is a limited general-utility check.

- [Replication condition metrics](runs/replication/summary.csv)
- [Paired recovery changes across both seeds](runs/paired-target-recovery.csv)
- [Replication ARC-Easy diagnostic](runs/replication/arc_easy/summary.json)

## Expanded study (quantized backbones, TOFU)

`expanded.py` scales the pilot to Qwen2.5 3B/7B and SmolLM2 1.7B on 4-bit NF4 backbones. The backbone weights never change. Every base update is its own named LoRA adapter on one shared backbone, and an evaluation activates a stack of adapters whose low-rank updates add together:

| Base | Active adapters |
| --- | --- |
| starting | none |
| full | `know-full` (forget + retain records) |
| clean | `know-clean` (retain records only) |
| edited | `know-full` + `edit-<method>-<step>` (GradDiff or NPO, trained with the task adapter detached) |

Stale task adapters (`old-*`) are trained on `full` and reloaded from disk for every condition. `aggregate` checks that each adapter's SHA-256 is the same in every condition that uses it. The backbone loads once per job, so swapping conditions only changes which adapters are active.

Data sources (`suites.py`):

- **synthetic**: the pilot's fictional author codes (`records.py`), using all three-digit codes because Qwen and SmolLM2 split digits.
- **tofu**: TOFU `forget05` (10 authors, 200 QA) as targets and the `retain_perturbed` authors (20 authors, 400 QA) as retained records. Task adapters learn record-grounded QA on other TOFU authors. The exposed group swaps 100 of its 300 examples for target-author QA, and `exposure_audit` lists which ones. Generations count as recovered when ROUGE-L recall reaches `rouge_threshold`.

Test probes for every condition:

- direct wording (shared with training) and paraphrases (TOFU's official `paraphrased_question` for the tofu source)
- alternate prompt formats (`chat`; `instruct` for synthetic, while TOFU uses `instruct` for validation)
- forced choice: the true code against three other codes for synthetic, or TOFU's paraphrased answer against its five perturbed answers (with the TOFU truth ratio)
- `sample_count` sampled generations per target prompt (a target counts as recovered if any sample contains it)
- ARC-Easy and HellaSwag utility samples

Edit selection uses validation wording and an ARC-Easy validation sample. `target-changes.csv` lists each target prompt that was recovered or lost relative to the same bare edited base.

```sh
# local CPU smoke test with SmolLM2-135M (thresholds disabled; checks plumbing only)
.venv/Scripts/python -u expanded.py all --config configs/expanded/smoke-tofu.json --run runs/smoke-tofu --device cpu

# Kaggle: build and push one private GPU job per config
python prepare_kaggle_expanded.py YOUR_KAGGLE_USERNAME configs/expanded/qwen3b-synthetic.json
kaggle kernels push -p .kaggle-build/atml-stale-lora-qwen3b-synthetic
kaggle kernels status YOUR_KAGGLE_USERNAME/atml-stale-lora-qwen3b-synthetic
kaggle kernels output YOUR_KAGGLE_USERNAME/atml-stale-lora-qwen3b-synthetic -p runs/kaggle/qwen3b-synthetic
```

If a job reaches Kaggle's 12-hour limit, rebuild with `--resume` and push again. The new version attaches the previous output and skips finished stages and conditions.

| Config | Backbone | Data | Ranks |
| --- | --- | --- | --- |
| `qwen3b-synthetic` | Qwen2.5-3B | synthetic, seed 31415 | 4, 8, 16 |
| `qwen3b-synthetic-seed2` | Qwen2.5-3B | synthetic, seed 27182 | 4, 8, 16 |
| `qwen7b-synthetic` | Qwen2.5-7B | synthetic, seed 31415 | 8 |
| `qwen3b-tofu` | Qwen2.5-3B | TOFU forget05 | 8 |
| `qwen7b-tofu` | Qwen2.5-7B | TOFU forget05, 100 eval QA per split, no clean-adapter role | 8 |
| `smollm2-tofu` | SmolLM2-1.7B | TOFU forget05 | 8 |
