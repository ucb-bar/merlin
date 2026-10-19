#!/usr/bin/env python3
"""Observe a real saved-program build, two typed relinks, execution and cleanup.

All target, host-package, board and runtime selections are caller-owned. The toy
program exercises generic host infrastructure on the selected target simulator;
it supplies no accelerator implementation and qualifies no model or compiler.
Outputs are operator evidence, never public capsules or author inputs. Runtime
qualification belongs to the independently selected witness consumer.
"""

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

from merlin.common import owned_process
from merlin.common.invocation_record import Invocation
from merlin.compile.model_execution_inputs import strict_tree_sha256
from merlin.llvmlower.compilation_recipe import verify_completed_recipe
from merlin.model_check import full_build as FB
from merlin.model_check import replay as RP
from merlin.model_check.full_build_identity import verified_recipe


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cleanup_witness(directory, source_revision):
    """Observe a deliberately timed-out command and its session-escaping child."""
    identity = """import json, os, pathlib, time
p = pathlib.Path('/proc/self')
record = {'pid': os.getpid(),
          'start_ticks': int((p/'stat').read_text().rsplit(')', 1)[1].split()[19]),
          'boot_id': pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
print(json.dumps(record), flush=True)
"""
    child = identity + "time.sleep(30)\n"
    parent = identity + (
        "import subprocess, sys\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}], start_new_session=True)\n"
        "time.sleep(30)\n"
    )
    argv = [sys.executable, "-c", parent]
    invocation = Invocation(
        directory,
        stage="phase1.infrastructure.timeout",
        argv=argv,
        inputs=[Path(__file__)],
        dependencies=[Path(owned_process.__file__)],
    )
    try:
        owned_process.run(argv, timeout=2, capture_output=True)
    except subprocess.TimeoutExpired as error:
        for name, data in (("stdout", error.output), ("stderr", error.stderr)):
            stream = invocation.directory / (name + ".bin")
            stream.write_bytes(data or b"")
            invocation.document[name] = pin(stream)
        cleanup = error.owned_process_cleanup
        invocation.document["process_tree_cleanup"] = cleanup
        invocation.failed(error)
        children = [json.loads(line) for line in (error.output or b"").splitlines()]
        require(len(children) >= 2, "timeout ended before both child identities were observed")
        require(
            cleanup["complete"] and cleanup["guardian_reaped"] and cleanup["error"] is None,
            "owned process cleanup did not close",
        )
        require(cleanup["reaped_children"] >= len(children), "observed children were not all reaped")
        for child in children:
            proc = Path("/proc") / str(child["pid"])
            if proc.exists():
                fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
                require(int(fields[19]) != child["start_ticks"], "an owned child remains alive")
        report = {
            "schema": "merlin.phase1.infrastructure_cleanup_witness.v1",
            "source_revision": source_revision,
            "producer": pin(__file__),
            "invocation": pin(invocation.path),
            "stdout": pin(invocation.directory / "stdout.bin"),
            "children": children,
            "owned_process_cleanup": cleanup,
        }
        result = directory / "cleanup_witness.json"
        result.write_text(json.dumps(report, indent=2) + "\n")
        return result
    else:
        raise ValueError("the deliberately blocked command did not time out")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--source-revision", required=True)
    p.add_argument("--host-package", type=Path, required=True)
    p.add_argument("--board-catalog", type=Path, required=True)
    p.add_argument("--board", required=True)
    p.add_argument("--dts", type=Path, required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--host-math-policy", default="native")
    a = p.parse_args()

    def source():
        from merlin.common.paths import repo_root

        require(repo_root().resolve() == a.repo.resolve(), "imported source root differs from selected checkout")
        require(
            Path(FB.__file__).resolve().is_relative_to(a.repo.resolve() / "src"),
            "whole-model builder was imported outside selected source",
        )
        require(
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=a.repo, text=True).strip() == a.source_revision,
            "probe source revision differs from the selected revision",
        )
        require(
            not subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], cwd=a.repo, text=True
            ).strip(),
            "probe source checkout has tracked changes",
        )

    from merlin.runtime.host_math import host_math_recipe

    host_math_recipe(a.host_math_policy)
    source()
    from merlin.common.paths import out_dir

    output_root = out_dir().resolve()
    a.out = a.out.resolve()
    require(
        a.out != output_root and a.out.is_relative_to(output_root), "probe output must lie below configured out root"
    )
    a.out.mkdir(parents=True, exist_ok=False)
    stage = a.out / "stage"
    stage.mkdir()
    stage.joinpath("model.mlir").write_text("""module {
      func.func @forward(%x: tensor<4xf32>, %i: tensor<4xi32>) -> (tensor<4xf32>, tensor<4xi32>) {
        %e0 = tensor.empty() : tensor<4xf32>
        %f = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                            iterator_types = ["parallel"]}
          ins(%x : tensor<4xf32>) outs(%e0 : tensor<4xf32>) {
          ^bb0(%a: f32, %unused: f32):
            %sum = arith.addf %a, %a : f32
            linalg.yield %sum : f32
          } -> tensor<4xf32>
        %e1 = tensor.empty() : tensor<4xi32>
        %j = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                            iterator_types = ["parallel"]}
          ins(%i : tensor<4xi32>) outs(%e1 : tensor<4xi32>) {
          ^bb0(%a: i32, %unused: i32):
            %sum = arith.addi %a, %a : i32
            linalg.yield %sum : i32
          } -> tensor<4xi32>
        return %f, %j : tensor<4xf32>, tensor<4xi32>
      }
    }
""")
    # Empty constant table: both arguments are independently bound runtime inputs.
    header = b"{}      "
    stage.joinpath("weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    stage.joinpath("weights.safetensors.manifest.json").write_text(
        json.dumps({"0": {"kind": "input", "name": "x"}, "1": {"kind": "input", "name": "i"}})
    )
    stage.joinpath("input_order.json").write_text(json.dumps({"x": 0, "i": 1}))
    arrays = {"in0": np.array([1.25, -2.5, 0.0, 17.0], np.float32), "in1": np.array([3, -7, 21, 99], np.int32)}
    np.savez(stage / "inputs.npz", **arrays)
    np.save(stage / "golden.npy", arrays["in0"] * np.float32(2))
    stage.joinpath("output_order.json").write_text(json.dumps(["out0", "out1"]))
    np.savez(stage / "goldens.npz", out0=arrays["in0"] * np.float32(2), out1=arrays["in1"] * np.int32(2))
    stage.joinpath("capture_receipt.json").write_text(
        json.dumps(
            {
                "schema": "m2m.capture-receipt.v1",
                "materialized_abi": {"complete": True},
                "scope": "Operator-created synthetic ABI fixture, no frontend/model/source-closure qualification",
                "artifacts": {f.name: {"bytes": f.stat().st_size, "sha256": pin(f)["sha256"]} for f in stage.iterdir()},
            },
            indent=2,
        )
    )
    from merlin.runtime.boards import load_boards

    board = load_boards(a.board_catalog)[a.board]
    settings = FB.BuildSettings(
        a.board, a.board_catalog, a.dts, "int8", 4, a.host_package, host_math_policy=a.host_math_policy
    )
    record = FB.build(
        stage,
        target=a.target,
        label="infrastructure-probe",
        submission=a.host_package,
        submission_sha256="",
        settings=settings,
        directory=a.out / "build",
        admission=object(),
        offload=False,
    )
    verified_recipe(record)
    runs = []
    for index, values in enumerate(
        (arrays, {"in0": np.array([-8.0, 0.25, 4.5, -13.0], np.float32), "in1": np.array([-19, 4, 77, 0], np.int32)})
    ):
        work = a.out / f"step{index}"
        elf = FB.relink(record, stage, values, work)
        inputs = work / "capture/inputs.npz"
        expected = work / "expected.npz"
        np.savez(expected, out0=values["in0"] * np.float32(2), out1=values["in1"] * np.int32(2))
        recipe = work / "compilation_recipe.json"
        verify_completed_recipe(recipe, executable=elf, expected_recipe_sha256=pin(recipe)["sha256"])
        argv, engine = RP.spike_argv(a.target, elf, dram_base=board.dram_base, dram_bytes=board.dram_bytes)
        invocation = Invocation(
            work,
            stage="phase1.infrastructure.target",
            argv=argv,
            cwd=engine["cwd"],
            env=engine["environment"],
            inputs=[elf, inputs],
            dependencies=[Path(__file__)],
        )
        try:
            from merlin.model_check.compute_execution import compute_step

            with compute_step(30) as remaining:
                done = owned_process.run(
                    argv, cwd=engine["cwd"], env=engine["environment"], capture_output=True, timeout=remaining
                )
            invocation.document["process_tree_cleanup"] = done.owned_process_cleanup
            invocation.complete(done)
            require(done.returncode == 0, f"target simulator returned {done.returncode}: {done.stderr[-2000:]!r}")
        except BaseException as error:
            invocation.failed(error)
            raise
        actual, metrics = FB.read(done.stdout, stage, record["build_hash"])
        with np.load(expected, allow_pickle=False) as archive:
            require(len(actual) == len(archive.files), "forward output count differs from independent reference")
            for i, value in enumerate(actual):
                want = archive[f"out{i}"]
                require(value.dtype == want.dtype and value.shape == want.shape, "typed output ABI differs")
                require(value.tobytes() == want.tobytes(), "compiled values differ from independently doubled inputs")
        runs.append(
            {
                "inputs": pin(inputs),
                "expected": pin(expected),
                "elf": pin(elf),
                "compilation_recipe": pin(recipe),
                "execution": pin(invocation.path),
            }
        )
    source()
    report = {
        "schema": "merlin.phase1.infrastructure_target_witness.v1",
        "source_revision": a.source_revision,
        "producer": pin(__file__),
        "target": a.target,
        "stage": {"path": str(stage), "sha256": strict_tree_sha256(stage)},
        "build_record": pin(a.out / "build/build.json"),
        "dram_base": board.dram_base,
        "dram_bytes": board.dram_bytes,
        "runs": runs,
    }
    output = a.out / "target_witness.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    cleanup = cleanup_witness(a.out, a.source_revision)
    source()
    print(json.dumps({"target": pin(output), "cleanup": pin(cleanup)}), flush=True)


if __name__ == "__main__":
    main()
