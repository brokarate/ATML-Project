"""Build a private Kaggle kernel bundle without credentials or local run data."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner", help="Kaggle username")
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    parser.add_argument("--slug", default="atml-stale-lora-pilot")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = root / ".kaggle-build" / args.slug
    out.mkdir(parents=True, exist_ok=True)
    sources = {name: (root / name).read_text()
               for name in ("pilot.py", "records.py", "modeling.py")}
    sources["config.json"] = args.config.read_text()
    sources["requirements.txt"] = (root / "requirements.txt").read_text()
    source = (
        "from pathlib import Path\n"
        "import subprocess\n"
        "import sys\n\n"
        "\n"
        "project = Path('/kaggle/working/stale-lora')\n"
        "project.mkdir(parents=True, exist_ok=True)\n"
        f"sources = {sources!r}\n"
        "for name, source in sources.items():\n"
        "    (project / name).write_text(source)\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', "
        "'-r', 'requirements.txt'], cwd=project, check=True)\n"
        "subprocess.run([sys.executable, '-u', 'pilot.py', 'all', '--config', 'config.json', "
        "'--run', 'run', '--device', 'cuda'], cwd=project, check=True)\n"
    )
    (out / "run.py").write_text(source)
    metadata = {
        "id": f"{args.owner}/{args.slug}", "title": "ATML Stale LoRA Pilot",
        "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
