"""Build a private Kaggle GPU kernel bundle for one expanded-study configuration.

The kernel installs pinned transformers/peft plus bitsandbytes on top of Kaggle's
PyTorch, runs expanded.py, and writes results to /kaggle/working/stale-lora/run.
With --resume, the kernel's own previous output is attached as input and copied
back before running, so a job that hit the 12-hour limit continues where it stopped.
"""

import argparse
import json
from pathlib import Path


SOURCES = ("expanded.py", "stack.py", "suites.py", "records.py")
REQUIREMENTS = "transformers==4.57.1\npeft==0.21.2\nbitsandbytes>=0.45\n"

RUNNER = '''from pathlib import Path
import shutil
import subprocess
import sys

project = Path("/kaggle/working/stale-lora")
project.mkdir(parents=True, exist_ok=True)
sources = {sources}
for name, text in sources.items():
    (project / name).write_text(text, encoding="utf-8")
run = project / "run"
if not run.exists():
    previous = sorted(Path("/kaggle/input").glob("**/stale-lora/run/manifest.json"))
    if previous:
        print("Resuming from", previous[0].parent, flush=True)
        shutil.copytree(previous[0].parent, run)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
               cwd=project, check=True)
subprocess.run(["nvidia-smi"], check=False)
subprocess.run([sys.executable, "-u", "expanded.py", "all", "--config", "config.json",
                "--run", "run", "--device", "cuda"], cwd=project, check=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("owner", help="Kaggle username")
    parser.add_argument("config", type=Path, help="e.g. configs/expanded/qwen3b-synthetic.json")
    parser.add_argument("--resume", action="store_true",
                        help="attach this kernel's previous output and continue from it")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    slug = f"atml-stale-lora-{args.config.stem}"
    out = root / ".kaggle-build" / slug
    out.mkdir(parents=True, exist_ok=True)
    sources = {name: (root / name).read_text(encoding="utf-8") for name in SOURCES}
    sources["config.json"] = args.config.read_text(encoding="utf-8")
    sources["requirements.txt"] = REQUIREMENTS
    (out / "run.py").write_text(RUNNER.format(sources=repr(sources)), encoding="utf-8")
    metadata = {
        "id": f"{args.owner}/{slug}", "title": f"ATML Stale LoRA {args.config.stem}",
        "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [f"{args.owner}/{slug}"] if args.resume else [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
