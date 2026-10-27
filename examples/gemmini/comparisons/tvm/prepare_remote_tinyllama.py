"""Resume frozen TinyLlama numerical preparation on the remote CPU host.

This prepares inputs/programs/references only. It never submits FireSim jobs or
changes TVM, schedules, calibration choices or the selected hardware.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from prepare_selected_inputs import digest
from selected_protocol import check_selected_input, identity, load_selected_protocol

from merlin.common.artifacts import new_product
from merlin.common.paths import repo_root


def bootstrap(seed, protocol_path):
    protocol, models, recipes = load_selected_protocol(protocol_path)
    manifest = json.loads((seed / "seed.json").read_text())
    root = repo_root()
    for row in manifest["sources"]:
        kind = row.get("root", "merlin")
        if kind not in ("merlin", "selected_contract"):
            raise ValueError("Unknown seed source root")
        base = protocol_path.parent.parent if kind == "selected_contract" else root
        source = (base / row["path"]).resolve()
        if not source.is_relative_to(base.resolve()):
            raise ValueError("Seed source escapes its declared root")
        if digest(source) != row["sha256"]:
            raise ValueError("Preparation source differs from the frozen seed: " + row["path"])
    for row in manifest["files"]:
        path = seed / row["path"]
        if not path.resolve().is_relative_to(seed.resolve()) or digest(path) != row["sha256"]:
            raise ValueError("Invalid frozen seed member: " + row["path"])
    product = new_product(
        "handoff",
        version=1,
        target="gemmini-tinyllama-remote-preparation",
        sources=[str(seed / "seed.json")],
        notes="TinyLlama preparation from the selected 8ec protocol; final E2E 32/32.",
    )
    for row in manifest["files"]:
        shutil.copy2(seed / row["path"], product.add_artifact(row["path"]))
    check_selected_input(product.path / "prepared-inputs/tiny-v00.npz", protocol["models"]["tiny_llama"])
    checkpoint = manifest["checkpoint"]
    destination = product.add_artifact("source/tiny/model/model.safetensors")
    # Reuse a verified local file when supplied; otherwise fetch the exact public
    # revision once. No tokenizer/model code is downloaded or executed here.
    local = os.environ.get("TINYLLAMA_CHECKPOINT_FILE")
    if local:
        if digest(local) != checkpoint["sha256"]:
            raise ValueError("Local checkpoint differs from the selected checkpoint")
        shutil.copy2(local, destination)
    else:
        temporary = destination.with_suffix(".download")
        with urllib.request.urlopen(checkpoint["url"], timeout=60) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        if digest(temporary) != checkpoint["sha256"]:
            raise ValueError("Downloaded checkpoint hash differs; partial file retained")
        temporary.rename(destination)
    product.add_artifact("preparation.json").write_text(
        json.dumps(
            dict(
                status="inputs_ready_exports_pending",
                seed_sha256=digest(seed / "seed.json"),
                profiles=manifest["profiles"],
                selected_protocol=identity(protocol_path),
                selected_models=identity(models),
                selected_recipe=identity(recipes["tiny_llama"]),
            ),
            indent=2,
        )
        + "\n"
    )
    product.write_manifest()
    print(product.path, flush=True)


def run_stage(phase, release, protocol_path):
    protocol, models, recipes = load_selected_protocol(protocol_path)
    check_selected_input(release / "prepared-inputs/tiny-v00.npz", protocol["models"]["tiny_llama"])
    root = repo_root()
    edge = root / "examples/gemmini/comparisons/tvm"
    m2m = Path(os.environ["MERLIN_M2M_DIR"]).resolve()
    if not release.is_dir() or not (m2m / "m2m").is_dir():
        raise ValueError("Supply the initialized release and pinned model2MLIR source")
    env = os.environ.copy()
    env.update(SELECTED_RELEASE=str(release), SELECTED_INPUTS=str(release / "prepared-inputs"))
    env["PYTHONPATH"] = os.pathsep.join([str(m2m), str(edge), str(root / "src"), env.get("PYTHONPATH", "")])
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env["HF_HUB_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    capture = release / "tiny-stages"
    if phase == "references":
        command = [
            sys.executable,
            str(edge / "calculate_selected_references.py"),
            "--model",
            "tiny_llama",
            "--capture",
            str(capture),
            "--models",
            str(models),
            "--protocol",
            str(protocol_path),
            "--inputs",
            str(release / "prepared-inputs/tiny-v00.npz"),
            "--output",
            str(release / "references"),
        ]
    else:
        env["SELECTED_TINY_STAGE"] = phase
        command = [
            sys.executable,
            str(root / "src/merlin/targetgen/_m2m_capture_worker.py"),
            "--loader",
            str(edge / "selected_tinyllama_functional_loader.py"),
            "--dtype",
            "int8",
            "--out",
            str(capture / "stages" / phase),
            "--m2m-dir",
            str(m2m),
            "--recipe",
            str(recipes["tiny_llama"]),
            "--materialize-bundle",
            "--save-exported-program",
            "--cpu-threads",
            "1",
        ]
    guarded = [
        sys.executable,
        str(edge / "run_local_preparation.py"),
        "--log",
        str(release / "logs" / (phase + ".log")),
        "--status",
        str(release / (phase + ".status.json")),
        "--",
        *command,
    ]
    subprocess.run(guarded, cwd=root, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("bootstrap", "prefill", "decode", "references"), required=True)
    parser.add_argument("--protocol", type=Path, required=True, help="Frozen 8ec protocol with approved run-local numerics")
    parser.add_argument("--seed-dir", type=Path)
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    if args.phase == "bootstrap":
        if args.seed_dir is None:
            parser.error("bootstrap requires --seed-dir")
        bootstrap(args.seed_dir.resolve(), args.protocol.resolve())
    else:
        if args.release is None:
            parser.error("capture/reference phases require --release")
        run_stage(args.phase, args.release.resolve(), args.protocol.resolve())


if __name__ == "__main__":
    main()
