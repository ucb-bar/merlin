"""Check chunk TVM semantics and index exact structural tuning owners.

This produces an inventory, not performance claims or a new tuning budget.
One owner supplies representative inputs for structurally identical tasks.
"""

import argparse
from pathlib import Path

import numpy as np
from prepare_firesim_chunks import write_json


def check(root, host_check=True):
    import tvm
    from tune_models import extract_inventory
    from tvm import relax
    from tvm.relax.backend.contrib.gemmini_search import WORKLOAD_ATTR
    from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities

    owners, representatives, checks = {}, {}, []
    for path in sorted(root.glob("*/*/program.json")):
        folder = path.parent
        name = folder.name
        print("checking " + name, flush=True)
        mod = tvm.ir.load_json(path.read_text())
        errors, failures = [], []
        if host_check:
            with np.load(folder / "fixture.npz", allow_pickle=False) as fixture:
                inputs = [tvm.nd.array(fixture[f"input_{i}"]) for i in range(len(mod["main"].params))]
                names = (
                    ["expected"]
                    if "expected" in fixture
                    else sorted((n for n in fixture if n.startswith("expected_")), key=lambda n: int(n.split("_")[1]))
                )
                expected = [fixture[n].copy() for n in names]
            exe = relax.build(mod, target="llvm -fp-contract=off")
            output = relax.VirtualMachine(exe, tvm.cpu())["main"](*inputs)
            output = [output] if isinstance(output, tvm.nd.NDArray) else list(output)
            for actual, reference in zip(output, expected, strict=True):
                actual = actual.numpy()
                if actual.shape != reference.shape or actual.dtype != reference.dtype or not np.isfinite(actual).all():
                    raise ValueError("Invalid output shape, dtype or finiteness")
                try:
                    if actual.dtype.kind in "iub":
                        np.testing.assert_array_equal(actual, reference)
                    else:
                        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=2e-6)
                except AssertionError as error:
                    failures.append(str(error))
                errors.append(float(np.max(np.abs(actual.astype(np.float64) - reference))))
        _, tasks, inventory = extract_inventory(mod, GemminiCapabilities(), fusion_level=2)
        owned = []
        for task in tasks:
            row = next(r for r in inventory if r["task"] == task.task_name)
            if row["kind"] != "contraction":
                continue
            key = str(task.dispatched[0]["main"].attrs[WORKLOAD_ATTR])
            if key not in owners:
                owned.append(key)
                owners[key] = dict(
                    owner=name,
                    program=str(path.relative_to(root)),
                    fixture=str((folder / "fixture.npz").relative_to(root)),
                    occurrences=[],
                    graph_fusion_level=2,
                )
                representatives[key] = task.dispatched[0]
            elif not tvm.ir.structural_equal(representatives[key], task.dispatched[0]):
                raise ValueError("Workload key collision; cannot share records")
            owners[key]["occurrences"].append(dict(chunk=name, task=task.task_name, weight=int(task.weight)))
        write_json(folder / "tasks.json", dict(tasks=inventory, owned_workload_keys=owned))
        checks.append(
            dict(
                chunk=name,
                program=str(path.relative_to(root)),
                host_check=("fail" if failures else "pass") if host_check else "NOT_RUN",
                max_abs_errors=errors,
                comparison_failures=failures,
                owned_workload_keys=owned,
            )
        )
        write_json(root / "checks.json", checks)
        write_json(root / "task-owners.json", owners)
        print(name + f": owns {len(owned)} new tasks; total {len(owners)}", flush=True)
    return not any(row["host_check"] == "fail" for row in checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--skip-host-check", action="store_true")
    args = parser.parse_args()
    if not check(args.root, not args.skip_host_check):
        raise SystemExit("Recorded host comparison failures; inspect checks.json. Tolerances unchanged.")


if __name__ == "__main__":
    main()
