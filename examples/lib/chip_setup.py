#!/usr/bin/env python3
"""One-step setup for the chip examples on a new machine: the capture environment, and the .env.

    python examples/lib/chip_setup.py [--m2m DIR] [--llvm PREFIX] [--chipyard DIR]
                                      [--zephyr-sw DIR] [--zephyr-sdk DIR] [--env-file PATH]

What it does:

1. **model2MLIR capture environment** -- the part nobody can guess. It clones model2MLIR if ``--m2m``
   (or ``MERLIN_M2M_DIR``) does not exist yet, and builds a Python 3.12 venv in it with PyTorch nightly
   (CPU), the **torch-mlir wheel that matches Merlin's LLVM pin** (see ``TORCH_MLIR``), torchao and timm.
   The model lowering runs in this venv, and its MLIR must be readable by the pinned ``mlir-translate``:
   a newer torch-mlir prints syntax that tool rejects, which surfaces only on multi-hart images.
2. **.env** -- writes the paths it knows (and the ones you pass) into the repo's ``.env``, which Merlin
   reads, so nothing has to be exported per shell. Existing keys you did not pass are left alone.
3. **What is still missing** -- LLVM 23, chipyard (with the OPU), the Zephyr workspace and SDK cannot be
   installed by a script in reasonable time or without your choice of location; for each missing one it
   prints the exact commands. Re-run with the paths once you have them.

It needs ``uv`` and ``git`` on PATH. It never deletes anything outside the venv it builds.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
M2M_URL = "https://github.com/ucb-bar/model2MLIR.git"
TORCH_NIGHTLY = "https://download.pytorch.org/whl/nightly/cpu"
TORCH_MLIR_WHEELS = "https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels"
#: The newest torch-mlir dev wheel at or before Merlin's LLVM pin (llvm-project a47bddcc, 2026-06-03; see
#: docs/guides/llvm_toolchain.md). Lowering prints LLVM-dialect MLIR in-process with this wheel and the
#: pinned mlir-translate reads it; a much newer wheel prints `llvm.load ... <alignment = N>`, which that
#: mlir-translate rejects. Move this together with the LLVM pin.
TORCH_MLIR = "torch-mlir==20260531.828"
#: model2MLIR's own runtime dependencies (installed with --no-deps above so they do not move torch), plus
#: the quantization and model libraries the shipped loaders import.
CAPTURE_DEPS = [
    "xdsl>=0.24",
    "structlog>=24.1",
    "numpy>=1.26",
    "PyYAML>=6.0",
    "ml_dtypes",
    "safetensors",
    "torchao",
    "timm",
]


def _run(cmd: list, **kw) -> None:
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def _dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and not line.lstrip().startswith("#"):
                out[key.strip()] = value.strip()
    return out


def _upsert_env(path: Path, values: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    seen = set()
    for i, line in enumerate(lines):
        key = line.partition("=")[0].strip()
        if key in values and not line.lstrip().startswith("#"):
            lines[i] = f"{key}={values[key]}"
            seen.add(key)
    lines += [f"{k}={v}" for k, v in values.items() if k not in seen]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(values)} setting(s) to {path}")


def capture_env(m2m: Path) -> Path:
    if shutil.which("uv") is None:
        raise SystemExit("uv is required (https://docs.astral.sh/uv/): curl -LsSf https://astral.sh/uv/install.sh | sh")
    if not (m2m / "pyproject.toml").is_file():
        _run(["git", "clone", M2M_URL, m2m])
    py = m2m / ".venv" / "bin" / "python"
    env = {**os.environ, "VIRTUAL_ENV": str(m2m / ".venv")}
    if not py.is_file():
        _run(["uv", "venv", "--python", "3.12", m2m / ".venv"])
    _run(["uv", "pip", "install", "--pre", "torch", "torchvision", "--index-url", TORCH_NIGHTLY], env=env)
    _run(["uv", "pip", "install", "--pre", "--no-deps", TORCH_MLIR, "-f", TORCH_MLIR_WHEELS], env=env)
    _run(["uv", "pip", "install", "--no-deps", "-e", m2m], env=env)
    _run(["uv", "pip", "install", *CAPTURE_DEPS], env=env)
    _run(
        [
            py,
            "-c",
            "import torch, torch_mlir, torchao, m2m.capture; print('capture environment OK, torch', torch.__version__)",
        ]
    )
    lock = m2m / ".venv" / "merlin-capture-freeze.txt"
    lock.write_text(subprocess.run(["uv", "pip", "freeze"], env=env, capture_output=True, text=True).stdout)
    print(f"package versions recorded in {lock}")
    return py


def missing_steps(settings: dict[str, str]) -> list[str]:
    def have(key: str, sub: str = "") -> bool:
        value = settings.get(key) or os.environ.get(key)
        return bool(value) and Path(value, sub).exists()

    steps = []
    if not (have("MERLIN_CLANG") or (REPO / "third_party/llvm-install/bin/clang-23").is_file()):
        steps.append(
            "LLVM/MLIR 23 (clang-23 + mlir-translate; no distribution packages it). Build the two-target recipe in\n"
            "    docs/guides/llvm_toolchain.md, or if you have one:  --llvm /path/to/llvm-install"
        )
    if not have("MERLIN_CHIPYARD", "generators/saturn/src/main/scala/exu/OuterProductUnit.scala"):
        steps.append(
            "chipyard whose generators/saturn carries the OPU\n"
            "  (pin saturn_opu_int8 in merlin/contract/hardware_pins.yaml):\n"
            "    git clone https://github.com/ucb-bar/chipyard.git && cd chipyard && ./build-setup.sh riscv-tools\n"
            "    cd generators/saturn && git remote add opu https://github.com/miles-rusch-berkeley/saturn-vectors.git\n"
            "    git fetch opu opu-int8 && git checkout ea373800169ce7bbb9ca27ececf74101a7caf9d5\n"
            "  then:  --chipyard /path/to/chipyard   (spike and the riscv64 toolchain come from its .conda-env too)"
        )
    if not have("ZEPHYR_BASE"):
        steps.append(
            "Zephyr west workspace (dev branch):\n"
            "    git clone -b dev https://github.com/ucb-bar/zephyr-chipyard-sw.git && cd zephyr-chipyard-sw\n"
            "    west init -l . && west update\n"
            "  then:  --zephyr-sw /path/to/zephyr-chipyard-sw"
        )
    if not have("ZEPHYR_SDK_INSTALL_DIR"):
        steps.append(
            "Zephyr SDK 0.17.0 (riscv64 toolchain is enough):\n"
            "    wget https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.17.0/"
            "zephyr-sdk-0.17.0_linux-x86_64_minimal.tar.xz\n"
            "    tar xf zephyr-sdk-0.17.0_linux-x86_64_minimal.tar.xz\n"
            "    zephyr-sdk-0.17.0/setup.sh -t riscv64-zephyr-elf\n"
            "  then:  --zephyr-sdk /path/to/zephyr-sdk-0.17.0"
        )
    return steps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--m2m", type=Path, help="model2MLIR checkout (cloned here if absent); default ../model2MLIR")
    ap.add_argument("--llvm", type=Path, help="an LLVM/MLIR 23 install prefix (has bin/clang-23, bin/mlir-translate)")
    ap.add_argument("--chipyard", type=Path, help="chipyard checkout whose generators/saturn carries the OPU")
    ap.add_argument("--zephyr-sw", type=Path, help="zephyr-chipyard-sw west workspace (dev branch)")
    ap.add_argument("--zephyr-sdk", type=Path, help="Zephyr SDK 0.17.0 install")
    ap.add_argument("--env-file", type=Path, default=REPO / ".env", help="where settings are written")
    ap.add_argument("--skip-capture-env", action="store_true", help="do not (re)build the model2MLIR venv")
    args = ap.parse_args(argv)

    current = _dotenv(args.env_file)
    m2m = (
        args.m2m
        or Path(current.get("MERLIN_M2M_DIR") or os.environ.get("MERLIN_M2M_DIR") or REPO.parent / "model2MLIR")
    ).resolve()
    settings: dict[str, str] = {}
    if not args.skip_capture_env:
        py = capture_env(m2m)
        settings |= {
            "MERLIN_M2M_DIR": str(m2m),
            "MERLIN_MODEL2MLIR": str(m2m),
            "MERLIN_M2M_VENV": str(m2m / ".venv"),
            "MERLIN_M2M_PYTHON": str(py),
        }
    if args.llvm:
        prefix = args.llvm.resolve()
        for tool in ("clang-23", "mlir-translate"):
            if not (prefix / "bin" / tool).is_file():
                raise SystemExit(f"--llvm {prefix}: no bin/{tool}")
        settings |= {
            "MERLIN_MLIR_INSTALL": str(prefix),
            "MERLIN_CLANG": str(prefix / "bin/clang-23"),
            "MERLIN_MLIR_TRANSLATE": str(prefix / "bin/mlir-translate"),
        }
    if args.chipyard:
        settings["MERLIN_CHIPYARD"] = str(args.chipyard.resolve())
    if args.zephyr_sw:
        sw = args.zephyr_sw.resolve()
        settings |= {"MERLIN_ZEPHYR_SW": str(sw), "ZEPHYR_BASE": str(sw / "zephyr_ws" / "zephyr")}
    if args.zephyr_sdk:
        settings["ZEPHYR_SDK_INSTALL_DIR"] = str(args.zephyr_sdk.resolve())
    if settings:
        _upsert_env(args.env_file, settings)

    steps = missing_steps({**current, **settings})
    if steps:
        print("\nStill needed (run this again with the path once you have it):")
        for n, step in enumerate(steps, 1):
            print(f"\n{n}. {step}")
        return 1
    print("\nEverything the chip examples need is configured. Next: ./run.sh preflight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
