"""Reproduce Qwen capture -> Merlin -> Atlas -> numerical model benchmarks.

Run `python examples/atlas/model_bringup/qwen/run.py --setup` once to prepare isolated dependencies.
Then `python examples/atlas/model_bringup/qwen/run.py` runs the campaign using build/bin or the
installed wheel's tools. Results go to atlas-out/qwen (must be fresh).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
MERLIN = "85db5fb4e6eeb0ffe24c5a9938da8da8f255701d"
MODEL = "337e9f249f938a1dff007ddf8a763e5cd4b3aa2b"


def run(command, **kwargs):
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def setup(deps):
    if not all(shutil.which(tool) for tool in ("uv", "git", "cargo")):
        raise SystemExit("setup needs uv, git and cargo on PATH")
    deps.mkdir(parents=True, exist_ok=True)
    for name, remote, revision in (
        ("merlin", "https://github.com/ucb-bar/merlin.git", MERLIN),
        ("npu-model", "https://github.com/ucb-ee194-tapeout/npu_model.git", MODEL),
    ):
        dest = deps / name
        if not dest.exists():
            run(["git", "init", dest])
            run(["git", "-C", dest, "remote", "add", "origin", remote])
            run(["git", "-C", dest, "fetch", "--depth", "1", "origin", revision])
            run(["git", "-C", dest, "checkout", "--detach", "FETCH_HEAD"])
        observed = subprocess.check_output(
            ["git", "-C", str(dest), "rev-parse", "HEAD"], text=True
        ).strip()
        if observed != revision:
            raise SystemExit(f"{dest}: expected {revision}, got {observed}")
    for name, version, requirements in (
        ("capture", "3.12", ["numpy", "safetensors", "xdsl==0.65.0"]),
        (
            "merlin-env",
            "3.14",
            [
                "numpy",
                "safetensors",
                "xdsl==0.68.0",
                "z3-solver",
                "jsonschema",
                "pyyaml",
            ],
        ),
        ("model-env", "3.14", ["numpy", "safetensors", "xdsl==0.65.0", "tabulate"]),
    ):
        dest = deps / name
        if not dest.exists():
            run(["uv", "venv", "--python", version, dest])
        run(["uv", "pip", "install", "--python", dest / "bin/python", *requirements])
    for name, torch_version in (("capture", "2.10.0"), ("model-env", "2.11.0")):
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                deps / name / "bin/python",
                f"torch=={torch_version}",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        )
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            deps / "capture/bin/python",
            "torch-mlir==20261001",
            "-f",
            "https://github.com/llvm/torch-mlir/releases/expanded_assets/v20261001",
        ]
    )
    run(
        [
            "cargo",
            "fetch",
            "--locked",
            "--manifest-path",
            deps / "merlin/src/merlin/semantic_compiler/egg_bridge/Cargo.toml",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--deps", type=Path, default=ROOT / "out/qwen-deps")
    parser.add_argument("-o", "--out", type=Path, default=ROOT / "atlas-out/qwen")
    parser.add_argument("--panel", choices=("positive", "signed"), default="positive")
    parser.add_argument("--capture-python", type=Path)
    parser.add_argument("--merlin-python", type=Path)
    parser.add_argument("--model-python", type=Path)
    parser.add_argument("--merlin-root", type=Path)
    parser.add_argument("--model-root", type=Path)
    args = parser.parse_args()
    if args.setup:
        setup(args.deps.resolve())
        print("Qwen dependencies ready; run this command again without --setup.")
        return
    sys.path.insert(0, str(ROOT / "python"))
    from atlas_mlir.toolchain import bin_dir, ToolchainError

    try:
        bins = bin_dir()
    except ToolchainError:
        import importlib.metadata

        bins = Path(
            importlib.metadata.distribution("atlas-mlir").locate_file("atlas_mlir/bin")
        )
        if not all((bins / name).is_file() for name in ("atlas-opt", "atlas-emit")):
            raise SystemExit(
                "build Atlas with CMake or install its wheel before running Qwen"
            )
    env = dict(os.environ, ATLAS_OOT_BIN_DIR=str(bins))
    interpreters = {
        key: (
            getattr(args, f"{key}_python") or args.deps / directory / "bin/python"
        ).absolute()
        for key, directory in [
            ("capture", "capture"),
            ("merlin", "merlin-env"),
            ("model", "model-env"),
        ]
    }
    for path in interpreters.values():
        if not path.is_file():
            raise SystemExit(f"{path} is missing; run with --setup first")
    merlin_root = (args.merlin_root or args.deps / "merlin").resolve()
    model_root = (args.model_root or args.deps / "npu-model").resolve()
    for directory, revision in ((merlin_root, MERLIN), (model_root, MODEL)):
        actual = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != revision or subprocess.check_output(
            ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"]
        ):
            raise SystemExit(f"{directory} must be clean at {revision}")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} must be empty; use -o for a new campaign")
    args.out.mkdir(parents=True, exist_ok=True)
    args.out = args.out.resolve()
    provenance = dict(
        merlin_revision=MERLIN,
        model_revision=MODEL,
        atlas_revision=subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        atlas_dirty=bool(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
            )
        ),
        atlas_bin_dir=env["ATLAS_OOT_BIN_DIR"],
        atlas_tool_sha256={
            name: hashlib.sha256((bins / name).read_bytes()).hexdigest()
            for name in ("atlas-opt", "atlas-emit")
        },
        python_packages={
            key: json.loads(
                subprocess.check_output(
                    [
                        str(interpreter),
                        "-c",
                        "import importlib.metadata as m, json; "
                        "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions()}))",
                    ],
                    text=True,
                )
            )
            for key, interpreter in interpreters.items()
        },
        panel=args.panel,
    )
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    run(
        [
            interpreters["capture"],
            HERE / "capture_workloads.py",
            "--out",
            args.out,
            "--panel",
            args.panel,
        ],
        env=env,
    )
    run(
        [interpreters["merlin"], HERE / "compile_workloads.py", "--out", args.out],
        env=dict(env, PYTHONPATH=str(merlin_root / "src")),
    )
    for row in json.loads((args.out / "compilation.json").read_text()):
        if row["status"] == "compiled":
            run(
                [
                    interpreters["model"],
                    HERE / "measure_workload.py",
                    "--out",
                    args.out / row["name"],
                ],
                env=dict(env, PYTHONPATH=str(model_root)),
            )
    rows = json.loads((args.out / "compilation.json").read_text())
    completed = [r["name"] for r in rows if r["status"] == "compiled"]
    blocked = [
        {k: r[k] for k in ("name", "status", "reason")}
        for r in rows
        if r["status"] != "compiled"
    ]
    summary = dict(
        compiled=completed,
        blocked=blocked,
        performance={
            name: json.loads((args.out / name / "performance.json").read_text())
            for name in completed
        },
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Qwen results: {args.out}; {len(completed)} compiled, {len(blocked)} blocked (see summary.json)"
    )


if __name__ == "__main__":
    main()
