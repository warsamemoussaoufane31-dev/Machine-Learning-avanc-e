#!/usr/bin/env python3
"""Copy a local .ipynb into the project notebooks/, commit and push to origin/main.

Usage:
  python tools/upload_notebook.py /path/to/your_notebook.ipynb [--name optional_name.ipynb]

This script is intended to be run where the repository is cloned (in this workspace).
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def sh(cmd, cwd=None):
    return subprocess.check_call(cmd, shell=True, cwd=cwd)


def main():
    parser = argparse.ArgumentParser(description="Upload a notebook into california-housing-ml/notebooks and push")
    parser.add_argument("src", help="Path to local .ipynb file")
    parser.add_argument("--name", help="Optional new filename for the notebook inside notebooks/")
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    if not src.exists():
        print(f"ERROR: source file not found: {src}")
        sys.exit(2)
    if src.suffix.lower() != ".ipynb":
        print("ERROR: source must be a .ipynb file")
        sys.exit(2)

    repo_root = Path(__file__).resolve().parents[1]
    target_dir = repo_root / "california-housing-ml" / "notebooks"
    target_dir.mkdir(parents=True, exist_ok=True)

    target_name = args.name or src.name
    target_path = target_dir / target_name

    print(f"Copying {src} -> {target_path}")
    shutil.copy2(src, target_path)

    try:
        sh(f"git add {str(target_path)}", cwd=str(repo_root))
        sh(f"git commit -m 'chore: add notebook {target_name}'", cwd=str(repo_root))
        sh("git push origin main", cwd=str(repo_root))
        print("Notebook uploaded and pushed to origin/main")
    except subprocess.CalledProcessError as e:
        print("Git push failed:", e)
        print("You can still commit manually:")
        print(f"  cd {repo_root} && git add {target_path} && git commit -m 'add notebook' && git push origin main")


if __name__ == "__main__":
    main()
