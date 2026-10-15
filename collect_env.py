"""Record the software and hardware environment, for reproduction.

    python collect_env.py                 # writes to my_environment/
    python collect_env.py --out DIR

Writes into the output folder (the reference files of the paper, in the
top-level folder, are not touched):
    ENVIRONMENT.md          OS, CPU, GPU and driver, Python, CUDA/cuDNN and the
                            versions of the packages BDTF uses (also printed)
    requirements-lock.txt   `pip freeze` of the active environment
    environment-lock.yml    `conda env export --no-builds` (if conda is available)
"""

import argparse
import importlib
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGES = ["torch", "numpy", "yaml", "h5py", "gym", "mujoco_py", "Cython"]


def run(cmd):
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def first_line(text):
    return text.splitlines()[0].strip() if text else "not found"


def os_name():
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip('"')
    except Exception:
        pass
    return platform.platform()


def cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def memory_gb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return f"{int(line.split()[1]) / 1024 ** 2:.0f} GB"
    except Exception:
        pass
    return "unknown"


def package_version(name):
    try:
        mod = importlib.import_module(name)
    except Exception as exc:                      # noqa: BLE001
        return f"not available ({type(exc).__name__})"
    return str(getattr(mod, "__version__", "installed (no __version__)"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "my_environment"), help="output folder")
    out_dir = Path(p.parse_args().out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [("Date", datetime.now().strftime("%Y-%m-%d")),
            ("OS", os_name()),
            ("Kernel", platform.release()),
            ("CPU", f"{cpu_name()} ({os.cpu_count()} logical cores)"),
            ("Memory", memory_gb()),
            ("Python", f"{platform.python_version()} ({sys.executable})"),
            ("Conda environment", os.environ.get("CONDA_DEFAULT_ENV", "none"))]

    smi = run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
    rows.append(("GPU (nvidia-smi)", "; ".join(sorted(set(smi.splitlines()))) if smi else "none"))
    rows.append(("CUDA version (driver)", first_line(run("nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"))))

    try:
        import torch

        rows.append(("PyTorch CUDA / cuDNN",
                     f"{torch.version.cuda} / {torch.backends.cudnn.version()}"))
        rows.append(("GPUs visible to PyTorch",
                     ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
                     or "none"))
    except Exception:
        pass

    for name in PACKAGES:
        label = {"yaml": "pyyaml", "mujoco_py": "mujoco-py", "Cython": "cython"}.get(name, name)
        rows.append((label, package_version(name)))

    mujoco_dir = Path.home() / ".mujoco" / "mujoco210"
    rows.append(("MuJoCo binaries", str(mujoco_dir) if mujoco_dir.exists() else "not found"))
    rows.append(("gcc", first_line(run("gcc --version"))))

    width = max(len(k) for k, _ in rows)
    print("\n".join(f"{k:<{width}}  {v}" for k, v in rows))

    lines = ["# Environment", "",
             "Recorded with `python collect_env.py`.", "",
             "| Item | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    (out_dir / "ENVIRONMENT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    freeze = run(f'"{sys.executable}" -m pip freeze')
    if freeze:
        (out_dir / "requirements-lock.txt").write_text(freeze + "\n", encoding="utf-8")
    conda = run("conda env export --no-builds")
    if conda:
        kept = [l for l in conda.splitlines() if not l.startswith("prefix:")]
        (out_dir / "environment-lock.yml").write_text("\n".join(kept) + "\n", encoding="utf-8")

    written = ["ENVIRONMENT.md"] + (["requirements-lock.txt"] if freeze else []) \
        + (["environment-lock.yml"] if conda else [])
    print(f"\nwritten to {out_dir}: {', '.join(written)}")


if __name__ == "__main__":
    main()
