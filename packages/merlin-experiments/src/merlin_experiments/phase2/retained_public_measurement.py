"""Reopen historical public-development labels without claiming fresh execution.

This is intake and byte custody only. A subsequent independently executed
functional/count report and numerical comparison remain separate obligations.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from merlin_timing.archive import _caller_selection
from merlin_timing.contracts import digest

SCHEMA = "merlin.phase2.retained_public_measurement.v1"


def logical_commands(value):
    """The producer's explicit semantic projection, omitting ISA positions only."""
    value = copy.deepcopy(value)
    plan = value.get("params", {}).get("global_program_plan")
    if plan is not None:
        for field in ("schedule_instruction_count", "prologue_instruction_range", "epilogue_instruction_range"):
            plan.pop(field, None)
        for task in plan.get("tasks", []):
            task.pop("instruction_start", None)
            task.pop("instruction_end", None)
    return value


def pin(path):
    path = Path(path)
    if (
        not path.is_absolute()
        or path.resolve() != path
        or any(p.is_symlink() for p in (path, *path.parents))
        or not path.is_file()
    ):
        raise ValueError("retained measurement requires canonical regular files")
    before = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    if any(
        getattr(before, k) != getattr(after, k) for k in ("st_dev", "st_ino", "st_size", "st_ctime_ns", "st_mtime_ns")
    ):
        raise ValueError("retained measurement changed during read")
    return {"path": str(path), "sha256": digest}


def _read(path):
    if Path(path).stat().st_size > 16 << 20:
        raise ValueError("retained metadata exceeds its unchanged bounded intake")
    return json.loads(Path(path).read_bytes())


def historical_invocation(path, *, source_successors):
    """Exact historical source successors supply bytes, never live authority."""
    row = _read(path)
    if (
        row.get("schema") != "merlin.invocation_record.v1"
        or row.get("kind") != "subprocess"
        or row.get("status") != "completed"
        or type(row.get("returncode")) is not int
        or row["returncode"] != 0
        or any(row.get(k) is not True for k in ("inputs_unchanged", "dependencies_unchanged", "executable_unchanged"))
    ):
        raise ValueError("historical label has no complete original process")
    custody = []
    roles = [(row[k], False) for k in ("executable", "stdout", "stderr")]
    roles += [(p, False) for p in (*row["inputs"], *row["outputs"])]
    roles += [(p, True) for p in row["dependencies"]]
    for original, source_role in roles:
        if set(original) != {"path", "sha256"}:
            raise ValueError("historical process pin is not exact")
        selected = source_successors.get(original["path"]) if source_role else None
        actual = pin(Path(selected) if selected is not None else Path(original["path"]))
        if actual["sha256"] != original["sha256"]:
            raise ValueError("historical process input/dependency/product changed")
        custody.append({"original": original, "current_bytes": actual, "authority": "historical_byte_custody_only"})
    return row, custody


def intake(
    path,
    *,
    manifest_path,
    source_path,
    declaration_path,
    invocation_path,
    facts_path,
    observed_contract_path,
    domain,
    source_successors=None,
):
    """Admit one historical label's identities; no numerical or timing verdict."""
    implementation = pin(__file__)
    source_successors = {} if source_successors is None else source_successors
    work = Path(path).parent.parent
    required = [
        Path(p)
        for p in (
            path,
            manifest_path,
            source_path,
            declaration_path,
            invocation_path,
            facts_path,
            observed_contract_path,
            work / "build-receipt.json",
            work / "inputs.npz",
            work / "commands.json",
            work / "harness.c",
            work / "counter-protocol-audit.json",
            work / "lowering_recipe.json",
            work / "readback_build.json",
        )
    ]
    originals = [pin(p) for p in required]
    row, manifest, declaration = _read(path), _read(manifest_path), _read(declaration_path)
    if (
        row.get("schema") != "phase2.exploratory.measurement.v1"
        or row.get("engine") != "gsim"
        or type(row.get("cycles")) is not int
        or row["cycles"] <= 0
        or row.get("exact") is not True
        or row.get("problems") != []
    ):
        raise ValueError("retained label has no complete positive historical measurement")
    members = [m for m in manifest["members"] if m["identity"] == row["member"]]
    if len(members) != 1 or members[0]["source_sha256"] != row["source_sha256"]:
        raise ValueError("retained label source differs from original manifest")
    original_commands = _read(members[0]["command_buffer_path"])
    if digest(original_commands) != members[0]["command_buffer_sha256"] or logical_commands(
        original_commands
    ) != logical_commands(_read(work / "commands.json")):
        raise ValueError("retained arm changed original source work/readout/ownership")
    if pin(source_path)["sha256"] != row["source_sha256"]:
        raise ValueError("current source bytes differ from historical public source")
    if (
        declaration.get("schema") != "merlin.public_development_source_selection.v1"
        or declaration.get("visibility") != "public"
        or declaration.get("phase") != "phase2"
        or declaration.get("manifest") != pin(manifest_path)
        or row["source_sha256"] not in declaration["source_groups"]
    ):
        raise ValueError("public development source must be explicitly selected before observations")
    for generator in declaration["generators"]:
        if pin(generator["path"]) != generator:
            raise ValueError("selected public source generator bytes changed")
    build = _read(work / "build-receipt.json")
    left, right = dict(row["build"]), dict(build)
    for value in (left, right):
        reused = value.pop("build_reused", False)
        if type(reused) is not bool:
            raise ValueError("malformed original build reuse")
    if left != right:
        raise ValueError("historical label borrowed another compiler arm")
    context = build["context"]
    readback = _read(work / "readback_build.json")
    object_name = readback.get("kernel_object_name")
    if type(object_name) is not str or Path(object_name).name != object_name or object_name in {".", ".."}:
        raise ValueError("original kernel object is not a bounded submission leaf")
    required.append(work / object_name)
    originals.append(pin(work / object_name))
    if (
        readback.get("schema") not in {"merlin_readback_build_v1", "merlin_readback_build_v2"}
        or readback["elf_sha256"] != build["elf_sha256"]
        or readback["harness_sha256"] != pin(work / "harness.c")["sha256"]
        or readback["kernel_object_sha256"] != build["object_sha256"]
        or pin(work / object_name)["sha256"] != build["object_sha256"]
    ):
        raise ValueError("original caller/object build differs from executed program")
    if (
        context["source_sha256"] != row["source_sha256"]
        or context["facts_sha256"] != pin(facts_path)["sha256"]
        or context["contract_sha256"] != pin(observed_contract_path)["sha256"]
        or build["simulators"]["gsim"] != domain["label_engine_sha256"]
        or build["inputs_sha256"] != pin(work / "inputs.npz")["sha256"]
        or build["command_buffer_sha256"] != digest(_read(work / "commands.json"))
        or build["lowering_recipe_sha256"] != pin(work / "lowering_recipe.json")["sha256"]
    ):
        raise ValueError("historical source/input/hardware/runtime identity differs")
    invocation, custody = historical_invocation(invocation_path, source_successors=source_successors)
    elf = pin(build["elf"])
    if (
        elf["sha256"] != build["elf_sha256"]
        or elf not in invocation["inputs"]
        or invocation["executable"]["sha256"] != domain["label_engine_sha256"]
        or invocation["argv"][1] != elf["path"]
        or invocation["stdout"]["sha256"] != row["console_sha256"]
    ):
        raise ValueError("historical cycle label differs from original execution")
    console = Path(invocation["stdout"]["path"]).read_text()
    metrics = [line.split() for line in console.splitlines() if line.startswith("METRIC cycles ")]
    if len(metrics) != 1 or metrics[0] != ["METRIC", "cycles", str(row["cycles"])]:
        raise ValueError("original console does not attest exact retained cycles")
    audit = _read(work / "counter-protocol-audit.json")
    if (
        audit.get("schema") != "merlin.counter_window_review.v1"
        or audit["elf_sha256"] != elf["sha256"]
        or pin(work / "counter-protocol-audit.json")["sha256"] != build["counter_audit_sha256"]
        or not audit["begin_pc"] < audit["entry_call_pc"] < audit["completion_call_pc"] < audit["end_pc"]
        or audit.get("completion_contains_fence") is not True
    ):
        raise ValueError("original linked counter window is incomplete")
    selection_sha, selection = _caller_selection((work / "harness.c").read_bytes())
    if (
        audit["counter_asm"] != selection["counter_asm"]
        or audit["program_counter_reads"] != 2
        or [r["pc"] for r in audit["counter_reads"]] != [audit["begin_pc"], audit["end_pc"]]
        or row["cycles"] > selection["max_delta_cycles"]
    ):
        raise ValueError("original counter review differs from selected caller semantics")
    for source in selection["counter_semantics"]["source_pins"]:
        if pin(source["path"]) != source:
            raise ValueError("selected cycle counter source changed")
    scope = {
        "schema": "merlin.kernel_call_scope.v1",
        "cycle_scope": "entry-through-completion",
        "timing_selection_sha256": selection_sha,
        "selection_record_sha256": digest(selection),
    }
    if digest(scope) != domain["scope_sha256"]:
        raise ValueError("retained caller is outside the selected counter-window domain")
    if (
        row["cold_warm"] != "UNKNOWN"
        or digest({"cache_warmth": "unknown", "protocol": None}) != domain["memory_regime_sha256"]
    ):
        raise ValueError("retained label memory warmth differs from the explicit unknown regime")
    memory = row.get("memory_evidence")
    if memory is not None:
        if (
            memory.get("status") != "complete"
            or memory["elf_sha256"] != elf["sha256"]
            or memory["console_sha256"] != row["console_sha256"]
            or pin(memory["output_path"])["sha256"] != memory["artifact_sha256"]
            or "+dump-out=" + memory["output_path"] not in invocation["argv"]
        ):
            raise ValueError("historical complete output dump differs from executed program")
        historical_output = pin(memory["output_path"])
        output_kind = "coherent_dump"
    else:
        from math import prod

        from merlin.runtime.out_b64 import OutB64Decoder

        values, decoder = {}, OutB64Decoder()
        for line in console.splitlines():
            decoder.consume(line.split(), values)
        decoder.require_closed()
        tensors = _read(work / "commands.json")["tensors"]
        expected = {name: spec for name, spec in tensors.items() if spec.get("role") == "output"}

        def size(value):
            return sum(size(x) for x in value) if isinstance(value, list) else 1

        if (
            not expected
            or set(values) != set(expected)
            or any(size(values[n]) != prod(t["shape"]) for n, t in expected.items())
        ):
            raise ValueError("historical console omits complete output values")
        historical_output = dict(invocation["stdout"])
        output_kind = "complete_console_values"
    result = {
        "schema": SCHEMA,
        "intake_implementation": implementation,
        "status": "historical_label_provenance_checked",
        "source_phase": "phase2",
        "source_group_sha256": row["source_sha256"],
        "split": declaration["source_groups"][row["source_sha256"]],
        "measurement": pin(path),
        "manifest": pin(manifest_path),
        "source": pin(source_path),
        "declaration": pin(declaration_path),
        "original_invocation": pin(invocation_path),
        "historical_dependency_custody": custody,
        "elf": elf,
        "input": pin(work / "inputs.npz"),
        "caller": pin(work / "harness.c"),
        "counter_review": pin(work / "counter-protocol-audit.json"),
        "counter_window": audit,
        "timing_selection": selection,
        "cycles": row["cycles"],
        "historical_output": historical_output,
        "historical_output_kind": output_kind,
        "historical_measurement": True,
        "measured_now": False,
        "fresh_numerical_and_count_observation": "REQUIRED",
        "timing_qualification": False,
        "promotion_authority": "NONE",
    }
    if [pin(p) for p in required] != originals or pin(__file__) != implementation:
        raise ValueError("historical intake bytes changed during checks")
    return result


def main(argv=None):
    """Materialize explicit source-only intake requests; never launch a target."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    task = Path(os.environ.get("TASK_DIR", ""))
    output = args.output
    if (
        not task.is_absolute()
        or task.resolve() != task
        or not output.is_absolute()
        or output.resolve() != output
        or output == task
        or not output.is_relative_to(task)
        or output.exists()
        or any(p.is_symlink() for p in (output, *output.parents))
    ):
        raise ValueError("intake output must be fresh and canonical inside explicit TASK_DIR")
    request_pin = pin(args.request)
    request = _read(args.request)
    if (
        set(request) != {"schema", "records"}
        or request["schema"] != "merlin.phase2.retained_public_measurement_batch.v1"
        or type(request["records"]) is not list
        or not 0 < len(request["records"]) <= 4096
    ):
        raise ValueError("retained intake needs an explicit bounded request roster")
    from merlin.common.cpu_pool import shared_compute_pool

    rows = []
    with shared_compute_pool(timeout_s=120):
        for selected in request["records"]:
            try:
                row = {"status": "historical_provenance_checked", "intake": intake(**selected)}
            except (ValueError, KeyError, TypeError, OSError) as error:
                row = {"status": "UNKNOWN", "failure": {"type": type(error).__name__, "message": str(error)}}
            rows.append(row)
    if pin(args.request) != request_pin:
        raise ValueError("retained intake request changed during checks")
    output.mkdir(parents=True, exist_ok=False)
    document = {
        "schema": "merlin.phase2.retained_public_measurement_batch_report.v1",
        "request": request_pin,
        "records": rows,
        "new_executions": 0,
        "qualification": False,
        "promotion_authority": "NONE",
    }
    (output / "report.json").write_text(json.dumps(document, indent=2) + "\n")
    return 0 if all(row["status"] == "historical_provenance_checked" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
