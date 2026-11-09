"""Build and run the layout example on Linux x86-64 with a host C compiler."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from merlin.api import Compilation, Deployment, run
from merlin.compiler.backend.cabi import CRegionABI, TensorPort
from merlin.runtime.service import RunOptions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compilation", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="fresh native build directory")
    parser.add_argument("--cc", default="cc", help="Linux host C compiler executable")
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine() != "x86_64":
        parser.error("this optional run example requires Linux x86-64; use WSL2 on Windows")

    compilation = Compilation.load(args.compilation)
    abi = CRegionABI.from_compilation(compilation)
    if (
        abi.inputs != (TensorPort((2, 3), "i32"),)
        or abi.outputs != (TensorPort((6,), "i32"),)
        or abi.resource_kind != "cpu"
        or abi.payloads
    ):
        parser.error("expected the quickstart's dense i32 layout C interface")
    args.output.mkdir(parents=True, exist_ok=False)
    image = args.output / "flatten"
    subprocess.run(
        [
            args.cc,
            "-std=c11",
            "-O2",
            f"-DMERLIN_EXAMPLE_SYMBOL={abi.symbol}",
            *(str(compilation.directory / path) for path in abi.link_inputs),
            str(Path(__file__).with_name("main.c")),
            "-o",
            str(image),
        ],
        check=True,
    )
    result = run(
        image,
        runner="merlin.linux",
        options=RunOptions(deployment=Deployment("host", "linux", "x86_64", "sysv", 1), harts=1),
    )
    (args.output / "run.json").write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    if result.status != "completed" or result.metadata.get("exit_code") != 0 or result.stdout != "1 2 3 4 5 6\n":
        raise RuntimeError(f"unexpected run result: {result}")
    print(result.stdout, end="")


if __name__ == "__main__":
    main()
