"""Check diagnostic outputs on LLVM and index tasks; not selected EL4 admission."""

import argparse
import json
from pathlib import Path

import numpy as np
import tvm
from tune_models import extract_inventory
from tvm import relax
from tvm.relax.backend.contrib.gemmini_search import WORKLOAD_ATTR
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = json.loads((args.input_dir / "chunks.json").read_text())["chunks"]
    priority = ["tinyllama-decode-mlp", "tinyllama-prefill-qkv", "tinyllama-decode-attention"]
    rows.sort(key=lambda row: (priority.index(row["name"]) if row["name"] in priority else len(priority), row["name"]))
    representatives, owners, checks = {}, {}, []
    caps = GemminiCapabilities()
    for row in rows:
        name = row["name"]
        folder = args.input_dir / "chunks" / name
        print("Checking " + name, flush=True)
        mod = tvm.ir.load_json((folder / "program.json").read_text())
        owned = []
        prepared, tasks, inventory = extract_inventory(mod, caps, fusion_level=2)
        for task in tasks:
            item = next(r for r in inventory if r["task"] == task.task_name)
            if item["kind"] != "contraction":
                continue
            key = str(task.dispatched[0]["main"].attrs[WORKLOAD_ATTR])
            if key not in owners:
                owners[key] = dict(owner=name, occurrences=[])
                representatives[key] = task.dispatched[0]
                owned.append(key)
            elif not tvm.ir.structural_equal(representatives[key], task.dispatched[0]):
                raise ValueError("Workload-key collision")
            owners[key]["occurrences"].append(dict(chunk=name, task=task.task_name, weight=int(task.weight)))
        (args.output_dir / (name + "-tasks.json")).write_text(
            json.dumps(dict(tasks=inventory, owned_workload_keys=owned), indent=2) + "\n"
        )
        with np.load(folder / "fixture.npz", allow_pickle=False) as fixture:
            inputs = [tvm.nd.array(fixture[f"input_{i}"]) for i in range(len(mod["main"].params))]
            expected = [fixture[key].copy() for key in sorted(fixture.files) if key.startswith("expected")]
        errors, failure = [], None
        try:
            executable = relax.build(mod, target="llvm -fp-contract=off")
            result = relax.VirtualMachine(executable, tvm.cpu())["main"](*inputs)
            result = [result] if isinstance(result, tvm.nd.NDArray) else list(result)
            for output, reference in zip(result, expected, strict=True):
                value = output.numpy()
                if value.shape != reference.shape or value.dtype != reference.dtype:
                    raise ValueError("Output shape/dtype differs from frozen diagnostic reference")
                errors.append(float(np.max(np.abs(value.astype(np.float64) - reference))))
                if value.dtype.kind in "iu":
                    np.testing.assert_array_equal(value, reference)
                else:
                    np.testing.assert_allclose(value, reference, rtol=1e-5, atol=2e-6)
        except Exception as error:
            failure = str(error)
        checks.append(
            dict(
                chunk=name,
                status="pass" if failure is None else "fail",
                max_abs_errors=errors,
                error=failure,
                owned_workload_keys=owned,
                scope="Host compiler/source agreement, no hardware timing",
            )
        )
        (args.output_dir / "host-checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        (args.output_dir / "task-owners.json").write_text(json.dumps(owners, indent=2) + "\n")
        print(name + ": " + checks[-1]["status"], flush=True)
    print(
        json.dumps(
            dict(chunks=len(checks), passed=sum(c["status"] == "pass" for c in checks), unique_contractions=len(owners))
        )
    )
    if any(row["status"] != "pass" for row in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
