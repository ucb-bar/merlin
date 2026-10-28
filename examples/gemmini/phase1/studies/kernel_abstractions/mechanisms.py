"""Offline mechanism controls over explicitly selected retained artifacts.

This operator study imports a COPY of a selected compiler. It neither supplies
compiler seeds to authors nor turns old timing into measurements of new bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def pin(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _copy_compiler(compiler, output):
    copied = output / "compiler"
    shutil.copytree(compiler, copied, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    members = [{"member": str(path.relative_to(copied)), **pin(path)}
               for path in sorted(copied.rglob("*")) if path.is_file()]
    write(output / "compiler-copy.json", {"source": str(Path(compiler).resolve()), "members": members})
    return copied


def compare_llvm(actual, expected):
    """Compare parsed bodies while reporting every header difference explicitly."""
    from xdsl.dialects.llvm import FuncOp
    from xdsl.parser import Parser

    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

    context = make_llvm_context()
    modules = [Parser(context, Path(path).read_text()).parse_module() for path in (actual, expected)]
    functions = [{op.sym_name.data: op for op in module.ops if isinstance(op, FuncOp)} for module in modules]
    comparisons = []
    for name in sorted(functions[0].keys() | functions[1].keys()):
        left, right = (group.get(name) for group in functions)
        if left is None or right is None:
            comparisons.append({"name": name, "status": "missing"})
            continue
        differences = {}
        for field in ("attributes", "properties"):
            a, b = getattr(left, field), getattr(right, field)
            differences[field] = {key: [str(a.get(key)), str(b.get(key))]
                                  for key in a.keys() | b.keys() if a.get(key) != b.get(key)}
        comparisons.append({"name": name, "body_structurally_equal": left.body.is_structurally_equivalent(right.body),
                            "header_differences": differences})
    return {"modules_structurally_equal": modules[0].is_structurally_equivalent(modules[1]),
            "functions": comparisons, "qualification": "Structural IR comparison, no execution or timing authority."}


def audit_oracle_study(study):
    """Reopen declared pins; preserve missing recursive closure as a separate fact."""
    document = json.loads(Path(study).read_text())
    selected = {}

    def visit(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                identity = str(value["path"]), value["sha256"]
                selected[identity] = value
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(document)
    checks = []
    for (path, expected), _value in sorted(selected.items()):
        actual = pin(path) if Path(path).is_file() else None
        checks.append({"path": path, "expected_sha256": expected, "actual": actual,
                       "status": "pass" if actual and actual["sha256"] == expected else "fail"})
    return {"schema": "kernel_abstractions.oracle_audit.v1", "study": pin(study),
            "declared_pin_count": len(checks), "pins": checks,
            "status": "pass" if all(row["status"] == "pass" for row in checks) else "fail",
            "source_closure": [{"id": row["id"], "closure": row.get("source_closure", "unknown")}
                               for row in document["rows"]],
            "retained_measurements": document["rows"],
            "observed_compositions": document.get("observed_pass_compositions", []),
            "qualification": "Declared pins only; no recursive fixture recovery or new execution. "
                             "Operator-selected capability ceiling, not autonomous learning."}


def audit_prepared_panel(handoff):
    document = json.loads(Path(handoff).read_text())
    records = {key: document[key] for key in ("ELF", "build", "native", "collector")}
    checks = {key: pin(value["path"])["sha256"] == value["sha256"] for key, value in records.items()}
    native = json.loads(Path(records["native"]["path"]).read_text())
    return {"handoff": pin(handoff), "pins_match": checks, "native_status": native["status"],
            "retained": document, "physical_timing": "unknown", "new_execution": False}


def replay_selected_compiler(compiler, source, output, *, python=sys.executable,
                             command="lower_target_to_llvm", expected_output=None):
    """Replay the selected manifest command in a private copy, without new settings."""
    import yaml

    from merlin.common import invocation_record

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    copied = _copy_compiler(compiler, output)
    manifest = yaml.safe_load((copied / "manifest.yaml").read_text())
    source_copy = output / "source.mlir"
    shutil.copyfile(source, source_copy)
    values = {"tool": str(copied / manifest["entrypoints"]["tool"]),
              "input_mlir": str(source_copy), "output_json": str(output / "command_buffer.json")}
    argv = [value.format(**values) for value in manifest["commands"][command]["argv"]]
    argv[0] = str(python)
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(output),
                   "PYTHONPATH": str(copied / "compiler_library") + os.pathsep + str(copied),
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    completed = invocation_record.run(argv, directory=output / "invocations", stage="retained_manifest_replay",
                                      inputs=(source_copy, copied / "manifest.yaml"), outputs=(),
                                      cwd=copied, env=environment, capture_output=True, timeout=180)
    (output / "stdout").write_bytes(completed.stdout)
    (output / "stderr").write_bytes(completed.stderr)
    receipt = {"schema": "kernel_abstractions.retained_replay.v1", "source": pin(source),
               "compiler_copy": pin(output / "compiler-copy.json"),
               "manifest": pin(copied / "manifest.yaml"), "argv": argv, "returncode": completed.returncode,
               "status": "emitted" if completed.returncode == 0 else "replay_gap",
               "output": pin(output / "stdout"), "diagnostics": pin(output / "stderr"),
               "expected_output": pin(expected_output) if expected_output is not None else None,
               "matches_retained_output_bytes": (pin(output / "stdout")["sha256"] == pin(expected_output)["sha256"])
                                                if expected_output is not None and completed.returncode == 0 else None,
               "qualification": "Same retained manifest/settings; fresh source compilation only. "
                                "No new numeric, physical or timing qualification."}
    if expected_output is not None and completed.returncode == 0:
        receipt["parsed_ir_comparison"] = compare_llvm(output / "stdout", expected_output)
    write(output / "receipt.json", receipt)
    return receipt


def _geometry(command_buffer):
    commands, tensors = command_buffer["commands"], command_buffer["tensors"]
    if [c["opcode"] for c in commands] != ["RES_PACK", "MATMUL_RESIDENT", "COMMIT", "EVICT"]:
        raise ValueError("control requires one complete raw resident-matmul source")
    pack, product, commit, evict = commands
    w = pack["operands"]["src"]
    a = product["operands"]["lhs"]
    y = commit["operands"]["dst"]
    if (product["operands"]["rhs"] != pack["operands"]["dst"]
            or commit["operands"]["src"] != product["operands"]["dst"]
            or evict["operands"]["handle"] != pack["operands"]["dst"]
            or commit["attributes"] != {"epilogue": [], "output_dtype": "i32"}
            or [tensors[x]["dtype"] for x in (a, w, y)] != ["i8", "i8", "i32"]):
        raise ValueError("source use-def or raw numerical contract differs")
    m, k = tensors[a]["shape"]
    kw, n = tensors[w]["shape"]
    if kw != k or tensors[y]["shape"] != [m, n]:
        raise ValueError("source matmul and live-out shapes disagree")
    return (m, k, n), (a, w, y)


def neighbor_shapes(shape, *, tile, rhs_rows, max_m_blocks):
    """Derive small tails and a fit/miss boundary from source and owned storage."""
    if (len(shape) != 3 or any(type(v) is not int or v <= 0 for v in (*shape, tile, rhs_rows, max_m_blocks))
            or rhs_rows // tile < 2):
        raise ValueError("neighbor controls need positive source geometry and two RHS tiles")
    _m, k, n = shape
    repeated_rows = max_m_blocks * tile + 1
    n_fit = ((rhs_rows // tile) // 2) * tile
    return [
        {"id": "single_row", "shape_mkn": (1, k, n)},
        {"id": "row_tail", "shape_mkn": (tile + 1, k, n)},
        {"id": "operand_upper_tails", "shape_mkn": (
            repeated_rows, (k // tile) * tile + 1, (n // tile) * tile + 1)},
        {"id": "operand_lower_tails", "shape_mkn": (
            repeated_rows, max(1, (k // tile) * tile - 1), max(1, (n // tile) * tile - 1))},
        {"id": "rhs_capacity_fit", "shape_mkn": (repeated_rows, tile + 1, n_fit)},
        {"id": "rhs_capacity_miss", "shape_mkn": (repeated_rows, tile + 1, n_fit + 1)},
    ]


def reshape_source(command_buffer, shape):
    """Reuse the canonical interface emitter, retaining the admitted source graph."""
    from merlin.targetgen.contract.interface_emit import emit_interface_mlir

    _original, (a, w, y) = _geometry(command_buffer)
    if len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError("dense source shapes must be positive static integers")
    cb = json.loads(json.dumps(command_buffer))
    m, k, n = shape
    cb["tensors"][a]["shape"], cb["tensors"][w]["shape"] = [m, k], [k, n]
    # COMMIT defines the output; do not introduce a second declaration of its SSA.
    cb["tensors"].pop(y)
    return emit_interface_mlir(cb)


def _selected_plan(dense, shape, retention):
    """Observe the saved planner's actual storage constructor, not a new split law."""
    defaults = inspect.signature(dense.plan_matmul).parameters
    limits = {name: defaults[name].default for name in ("max_m_blocks", "max_k_blocks", "max_n_blocks")}
    constructor, captured = dense.DenseLoopStorage, []

    def storage(*args, **kwargs):
        value = constructor(*args, **kwargs)
        captured.append(value)
        return value

    dense.DenseLoopStorage = storage
    try:
        result = dense.plan_matmul(*shape, **limits, segmented=False, resident_within_call=retention,
                                   resident_prepare_batch=True, retention_aware_blocks=False)
    finally:
        dense.DenseLoopStorage = constructor
    if len(captured) != 1:
        raise ValueError("saved planner did not expose one original owned storage assignment")
    return result, captured[0], limits


def execute_dense(kernel, adapter, shape, *, edge, operand_rows, accumulator_rows):
    """Independently execute actual projected loads/products/stores, including tails."""
    from collections import Counter
    from dataclasses import replace

    import numpy as np

    from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store
    from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
    from merlin.sched.ir import evaluate, instances

    m, k, n = shape
    lhs = (np.arange(m*k, dtype=np.int64).reshape(m, k) % 7 - 3)
    rhs = (np.arange(k*n, dtype=np.int64).reshape(k, n) % 5 - 2)
    operand = np.zeros((operand_rows, edge), dtype=np.int64)
    acc = np.zeros((accumulator_rows, edge), dtype=np.int64)
    output = np.full((m, n), 2**40, dtype=np.int64)
    writes = np.zeros((m, n), dtype=np.uint8)
    calls, stream, desc, macs, requested = Counter(), hashlib.sha256(), None, 0, Counter()
    for call, env in instances(kernel):
        values = tuple((name, evaluate(value, env)) for name, value in call.args)
        stream.update(json.dumps([call.instr, values], separators=(",", ":")).encode() + b"\n")
        calls[call.instr] += 1
        if call.instr == DRAIN:
            continue
        op = adapter.project(replace(call, args=values))
        if type(op) is Load:
            tensor = lhs if op.role == "lhs" else rhs
            operand[op.row:op.row+op.rows, :op.cols] = tensor[
                op.dram_row:op.dram_row+op.rows, op.dram_col:op.dram_col+op.cols]
            requested[op.role] += op.rows * op.cols
        elif type(op) is Preload:
            desc = op
        elif type(op) is Compute:
            if desc is None or not op.fresh_weights:
                raise ValueError("this control executor requires original fresh descriptor pairs")
            block = operand[op.input_row:op.input_row+op.rows, :op.cols] @ operand[
                desc.weight_row:desc.weight_row+op.cols, :desc.cols]
            dest = acc[desc.accumulator_row:desc.accumulator_row+op.rows, :desc.cols]
            dest[:] = block + (dest if desc.accumulate else 0)
            macs += op.rows * op.cols * desc.cols
            desc = None
        elif type(op) is Store:
            output[op.dram_row:op.dram_row+op.rows, op.dram_col:op.dram_col+op.cols] = acc[
                op.accumulator_row:op.accumulator_row+op.rows, :op.cols]
            writes[op.dram_row:op.dram_row+op.rows, op.dram_col:op.dram_col+op.cols] += 1
        else:
            raise ValueError("unbound actual numerical effect")
    if desc is not None or macs != m*k*n or not np.all(writes == 1):
        raise ValueError("actual stream changed source products or complete single live-out writes")
    np.testing.assert_array_equal(output, lhs @ rhs)
    return {"status": "pass", "checked_elements": m*n, "source_macs": macs,
            "output_sha256": hashlib.sha256(output.astype("<i4").tobytes()).hexdigest(),
            "ordered_dynamic_calls_sha256": stream.hexdigest(), "calls": dict(calls),
            "requested_operand_bytes": dict(requested), "target_execution": "not_checked",
            "qualification": "Independent integer executor; exact input fixture, not all possible inputs."}


def _worker(compiler, source, output, retention, specialization):
    """Run in its own process so the retained SDK cannot mix with current imports."""
    sys.path[:0] = [str(compiler), str(compiler / "compiler_library")]
    import dense
    import target
    from frontend import parse_source

    from merlin.compile.scheduling.ordered_unroll import unroll_checked_result

    cb = parse_source(source.read_text())
    shape, names = _geometry(cb)
    result, storage, limits = _selected_plan(dense, shape, retention)
    edge, operands, accumulators = dense._geometry()
    required_rhs_rows = ((shape[1] + edge - 1) // edge) * ((shape[2] + edge - 1) // edge) * edge
    applied = "rhs_residency" in result.realization
    applicability = {"applied": applied, "required_rhs_rows": required_rhs_rows,
                     "owned_rhs_rows": storage.rhs.capacity,
                     "status": "applied" if applied else "capacity_declined" if required_rhs_rows > storage.rhs.capacity
                     else "no_redundant_rhs_loads" if retention else "control_off"}
    precursor = result.kernel.digest()
    if specialization:
        result = unroll_checked_result(result, max_nodes=128, max_extent=16)
    result.revalidate()
    numeric = execute_dense(result.kernel, dense.dense_adapter(), shape, edge=edge,
                            operand_rows=operands, accumulator_rows=accumulators)
    # The experimental seam supplies the selected checked result to the SAME
    # unchanged target consumer. Every private preparation stays inside emission.
    original_planner = dense.plan_matmul
    consumed = []
    def selected(*args, **kwargs):
        if tuple(args[:3]) != shape:
            raise ValueError("target consumer requested a different source geometry")
        consumed.append(result.kernel.digest())
        return result
    dense.plan_matmul = selected
    try:
        abi = [row["tensor"] for row in cb["kernel_abi"]["args"]]
        builder = target.Builder(abi)
        emitted = dense.lower_matmul(builder, *(builder.args[name] for name in names), *shape,
                                     options={"retain_full_lhs": False})
        if emitted is not result or consumed != [result.kernel.digest()]:
            raise ValueError("target lowering did not consume the selected checked result")
        text = builder.module_text()
    finally:
        dense.plan_matmul = original_planner
    result.revalidate()
    (output / "kernel.llvm.mlir").write_text(text)
    (output / "counted-kernel.txt").write_text(str(result.kernel) + "\n")
    write(output / "source.cb.json", cb)
    write(output / "realization.json", result.realization)
    receipt = {"schema": "kernel_abstractions.mechanism_cell.v1", "source": pin(source),
               "shape_mkn": shape, "retention": retention, "ordered_specialization": specialization,
               "fixed": {"segmented": False, "resident_prepare_batch": True,
                         "retention_aware_blocks": False, "unroll_nodes": 128, "unroll_extent": 16, **limits},
               "retention_applicability": applicability,
               "precursor_kernel_sha256": precursor, "kernel_sha256": result.kernel.digest(),
               "source_and_epoch_checks": "pass", "numeric": numeric,
               "selection_seam": "checked result supplied to unchanged dense.lower_matmul",
               "target_consumed_kernel_sha256": consumed[0], "emitted": pin(output / "kernel.llvm.mlir"),
               "counted_kernel": pin(output / "counted-kernel.txt"), "realization": pin(output / "realization.json"),
               "compiler_owners": [pin(compiler / name) for name in (
                   "frontend.py", "dense.py", "resident.py", "target.py", "rtl_facts/facts.json",
                   "compiler_library/merlin/compile/scheduling/ordered_unroll.py")],
               "timing": "unknown", "new_hardware_execution": False}
    write(output / "receipt.json", receipt)


def run_four_cells(compiler, source, output, *, python=sys.executable):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    copied = _copy_compiler(compiler, output)
    source_copy = output / "source.mlir"
    shutil.copyfile(source, source_copy)
    cells = []
    for retained in (False, True):
        for specialized in (False, True):
            case = output / f"p{int(retained)}s{int(specialized)}"
            case.mkdir()
            argv = [str(python), str(Path(__file__).resolve()), "worker", str(copied), str(source_copy),
                    str(case), str(int(retained)), str(int(specialized))]
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(output),
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
            run = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=300)
            (case / "stdout").write_text(run.stdout)
            (case / "stderr").write_text(run.stderr)
            cells.append({"cell": case.name, "returncode": run.returncode,
                          "receipt": pin(case / "receipt.json") if (case / "receipt.json").is_file() else None})
    successful = [json.loads(Path(row["receipt"]["path"]).read_text()) for row in cells if row["receipt"]]
    preservation = {}
    for retained in (False, True):
        pair = [x for x in successful if x["retention"] == retained]
        preservation[str(retained)] = len(pair) == 2 and (
            pair[0]["numeric"]["ordered_dynamic_calls_sha256"] == pair[1]["numeric"]["ordered_dynamic_calls_sha256"])
    outputs_equal = len(successful) == 4 and len({row["numeric"]["output_sha256"] for row in successful}) == 1
    traffic = {row["retention"]: row["numeric"]["requested_operand_bytes"] for row in successful}
    retention_observed = (len(traffic) == 2 and traffic[True]["weight"] < traffic[False]["weight"]
                          and traffic[True]["lhs"] == traffic[False]["lhs"])
    checks_pass = outputs_equal and all(preservation.values())
    unavailable = [row["retention_applicability"] for row in successful
                   if row["retention"] and not row["retention_applicability"]["applied"]]
    report = {"schema": "kernel_abstractions.four_cells.v1", "source": pin(source), "cells": cells,
              "compiler_copy": pin(output / "compiler-copy.json"), "outputs_equal": outputs_equal,
              "retention_traffic_change_observed": retention_observed,
              "specialization_preserves_ordered_calls": preservation,
              "checks_status": "pass" if checks_pass else "fail",
              "status": "pass" if checks_pass and retention_observed else "not_applicable" if checks_pass else "fail",
              "intervention_gaps": unavailable,
              "timing": "unknown", "hardware_execution": False,
              "qualification": "New emissions have no inherited historical cycle measurements."}
    write(output / "index.json", report)
    return report


def run_neighbors(compiler, source, output, *, python=sys.executable):
    """Freeze the saved control policy; check every derived source, including declines."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    copied = _copy_compiler(compiler, output)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(output)}
    argv = [str(python), str(Path(__file__).resolve()), "probe", str(copied), str(Path(source).resolve())]
    probe = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=90)
    (output / "probe.stderr").write_text(probe.stderr)
    probe.check_returncode()
    profile = json.loads(probe.stdout)
    write(output / "profile.json", profile)
    shapes = neighbor_shapes(profile["shape_mkn"], tile=profile["tile"], rhs_rows=profile["rhs_rows"],
                             max_m_blocks=profile["limits"]["max_m_blocks"])
    rows = []
    for choice in shapes:
        generated = output / (choice["id"] + ".mlir")
        generated.write_text(reshape_source(profile["command_buffer"], choice["shape_mkn"]))
        report = run_four_cells(copied, generated, output / choice["id"], python=python)
        rows.append({**choice, "source": pin(generated), "result": pin(output / choice["id"] / "index.json"),
                     "checks_status": report["checks_status"], "intervention_status": report["status"],
                     "intervention_gaps": report["intervention_gaps"]})
    report = {"schema": "kernel_abstractions.neighbors.v1", "base_source": pin(source),
              "profile": pin(output / "profile.json"), "compiler_copy": pin(output / "compiler-copy.json"),
              "rows": rows, "status": "pass" if all(row["checks_status"] == "pass" for row in rows) else "fail",
              "selection": "unchanged saved planner and fixed limits; no per-shape timing or retuning",
              "performance": "unknown; no historical cycles transferred to new programs"}
    write(output / "index.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("worker", "four-cells", "audit", "probe", "neighbors"))
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)
    if args.action == "worker":
        compiler, source, output, retention, specialization = args.paths
        _worker(Path(compiler), Path(source), Path(output), bool(int(retention)), bool(int(specialization)))
        return 0
    elif args.action == "probe":
        compiler, source = map(Path, args.paths)
        sys.path[:0] = [str(compiler), str(compiler / "compiler_library")]
        import dense
        from frontend import parse_source

        cb = parse_source(source.read_text())
        shape, _names = _geometry(cb)
        _result, storage, limits = _selected_plan(dense, shape, False)
        print(json.dumps({"command_buffer": cb, "shape_mkn": shape, "tile": dense._geometry()[0],
                          "rhs_rows": storage.rhs.capacity, "limits": limits}))
        return 0
    elif args.action == "neighbors":
        report = run_neighbors(*map(Path, args.paths))
    elif args.action == "four-cells":
        report = run_four_cells(*map(Path, args.paths))
    else:
        report = audit_oracle_study(Path(args.paths[0]))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] in ("pass", "not_applicable") else 1


if __name__ == "__main__":
    raise SystemExit(main())
