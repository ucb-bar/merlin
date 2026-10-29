"""Strict live binding for the standalone proof-producing graph pipeline leaf.

Fresh parser projection and deterministic query reconstruction precede kernel
checking. A receipt is not policy/spec review, archive authenticity, broker
integration, native fidelity or an all-graph/model claim.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import MappingProxyType

from merlin.common.owned_process import run_owned
from merlin.common.paths import repo_root

from . import lean_graph_state as stable
from .lean_graph_state_pipeline_emit import emit
from .lean_graph_state_pipeline_kernel import exact, write
from .lean_graph_state_pipeline_models import symbolic_witness
from .lean_graph_state_pipeline_proof import check_query, execute, fixed_context, prepare

SCHEMA = "merlin.graph-state-pipeline.live-receipt.v1"
CLAIMS = MappingProxyType(
    {
        "relation": (
            "same ordered returns/full persistent outcome including error/terminal world; private SSA excluded"
        ),
        "supported_defined_entry_roster_preserved": True,
        "public_acceptance_issued": False,
        "review_approved": False,
        "source_success_for_all_models": False,
        "poison_freedom_proved": False,
        "native_fidelity_proved": False,
        "primitive_fidelity_proved": False,
        "numerical_gates_discharged": False,
        "all_graph_passes_verified": False,
        "full_model_claim": False,
        "broker_integration": False,
        "python_projection_proved": False,
        "limited_literal_trace_promoted": False,
    }
)

KEYS = {
    "schema",
    "version",
    "config",
    "root",
    "originals",
    "snapshots",
    "report",
    "profile",
    "query",
    "object",
    "sources",
    "files",
    "dependencies",
    "claims",
    "roots",
}


def closed(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError("closed versioned pipeline record required")


def valid_claims(value):
    return (
        type(value) is dict
        and set(value) == set(CLAIMS)
        and all(type(value[key]) is type(expected) and value[key] == expected for key, expected in CLAIMS.items())
    )


def reference_row(value):
    closed(value, {"path", "sha256", "bytes"})
    if type(value["path"]) is not str or not Path(value["path"]).is_absolute():
        raise ValueError("exact absolute live receipt path required; archived Linux records are historical")
    if type(value["sha256"]) is not str or len(value["sha256"]) != 64 or set(value["sha256"]) - set("0123456789abcdef"):
        raise ValueError("exact lowercase SHA256 identity required")
    if type(value["bytes"]) is not int or value["bytes"] < 0:
        raise ValueError("exact nonnegative byte length required; bool refused")


def config(factor, tile, bits, mode):
    for value in (factor, tile, bits):
        if type(value) is not int:
            raise ValueError("exact integer pipeline configuration required")
    if mode not in ("graph", "symbolic-provider-counter-v1"):
        raise ValueError("unknown pipeline proof mode")
    return {"factor": factor, "tile": tile, "index_bits": bits, "mode": mode}


def sources():
    root = repo_root()
    selected = [
        *root.glob("src/merlin/verify/lean_graph_state_pipeline*.py"),
        *root.glob("src/merlin/verify/lean/Merlin/GraphStatePipeline*.lean"),
    ]
    selected += [
        root / "src/merlin/verify" / name
        for name in (
            "lean_graph_state.py",
            "lean_graph_state_scalar.py",
            "lean_graph_state_scalar_capture.py",
            "lean_graph_state_scalar_execution.py",
            "lean_graph_state_scalar_models.py",
        )
    ]
    selected += [
        root / "src/merlin/verify/lean/Merlin/GraphStateAudit.lean",
        root / "src/merlin/common/owned_process.py",
    ]
    return [stable.file_ref(path) for path in sorted(set(selected))]


def projection(buffers, destination, reference, settings):
    script = repo_root() / "src/merlin/verify/lean_graph_state_pipeline_capture.py"
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(script),
        "--reference",
        str(reference),
        "--destination",
        str(destination / "projection"),
        "--factor",
        str(settings["factor"]),
        "--tile",
        str(settings["tile"]),
        "--bits",
        str(settings["index_bits"]),
        "--buffers",
        *map(str, buffers),
    ]
    with (destination / "projection.stdout").open("x") as out, (destination / "projection.stderr").open("x") as err:
        result = run_owned(
            command,
            cwd=reference,
            env={**os.environ, "PYTHONPATH": str(reference / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=out,
            stderr=err,
            timeout=60,
            check=False,
        )
    if result.returncode:
        raise ValueError("complete independent pinned projection refused; retained buffers: " + str(destination))
    return json.loads((destination / "projection/CAPTURE.json").read_text())


def author(capture, mode):
    if mode == "symbolic-provider-counter-v1":
        return symbolic_witness(capture)
    if mode != "graph":
        raise ValueError("unknown selected proof mode")
    base = "Merlin.GraphStatePipelineInstance."
    roots = [
        base + n
        for n in (
            "complete",
            "controlAccepted",
            "literalsAccepted",
            "literalsFit",
            "originalEntriesChecked",
            "finalEntriesChecked",
        )
    ]
    for index, function in enumerate(capture["stages"][0]["functions"]):
        if function["declaration"]:
            continue
        roots.append(base + f"F{index}.actual")
        if function["before"] != capture["stages"][-1]["functions"][index]["after"]:
            roots += [
                base + f"F{index}." + n
                for n in (
                    "body_group",
                    "innerAccepted",
                    "outer_same",
                    "unroll_actual",
                    "segment_actual",
                    "unrollBoundsChecked",
                    "segmentBoundsChecked",
                )
            ]
    return emit(capture), roots, None


def check_boundary(buffers, destination, *, reference, factor=2, tile=3, bits=32, mode="graph"):
    settings = config(factor, tile, bits, mode)
    destination = Path(destination).resolve()
    if len(buffers) != 5:
        raise ValueError("exact source/three-middle/final boundary required")
    buffers = [Path(path).resolve() for path in buffers]
    originals = [stable.file_ref(path) for path in buffers]
    original_sources = sources()
    profile = prepare()
    report = {
        "schema": "merlin.graph-state-pipeline.boundary.v1",
        "status": "unsupported",
        "config": settings,
        "originals": originals,
        "sources": original_sources,
        "claims": dict(CLAIMS),
        "qualified": False,
        "profile": profile["manifest"],
        "full_LeanMLIR_route": "unavailable: no matching selected manifest",
    }
    try:
        capture = projection(buffers, destination, Path(reference).resolve(), settings)
        report["projection"] = stable.file_ref(destination / "projection/CAPTURE.json")
        report["snapshots"] = capture["buffers"]
        report["each_actual_stage_changed"] = capture["each_stage_changed"]
        query, roots, witness = author(capture, mode)
    except (ValueError, TypeError, KeyError, IndexError) as error:
        report["reason"] = {"kind": type(error).__name__, "detail": str(error)}
        destination.mkdir(parents=True, exist_ok=True)
        write(destination / "report.json", report)
        return report
    checked = check_query(query, destination / "kernel", roots)
    report.update(
        {
            "status": checked["status"],
            "kernel": stable.file_ref(destination / "kernel/report.json"),
            "exact_roots": roots,
            "witness": witness,
            "projection_trusted": True,
        }
    )
    for row in [*originals, *original_sources, *capture["buffers"]]:
        exact(row)
    if sources() != original_sources:
        raise ValueError("selected leaf/helper source closure changed during check")
    write(destination / "report.json", report)
    return report


def collect_refs(value):
    if isinstance(value, dict):
        if set(value) >= {"path", "sha256", "bytes"}:
            yield {k: value[k] for k in ("path", "sha256", "bytes")}
        for child in value.values():
            yield from collect_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from collect_refs(child)


def issue(report_path, destination):
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    if report["status"] != "conditional_graph_checked" or not valid_claims(report["claims"]):
        raise ValueError("only an actual checked complete boundary can receive a conditional receipt")
    root = report_path.parent
    kernel = json.loads(Path(report["kernel"]["path"]).read_text())
    profile = prepare()
    capture = json.loads(Path(report["projection"]["path"]).read_text())
    query, roots, _ = author(capture, report["config"]["mode"])
    if (
        Path(kernel["source"]["path"]).read_text() != query
        or roots != kernel["roots"]
        or roots != report["exact_roots"]
    ):
        raise ValueError("exact generated query/root binding differs")
    if kernel["status"] != "conditional_graph_checked" or kernel["audit"]["status"] != "checked":
        raise ValueError("actual standard-three kernel/audit result required")
    if report["sources"] != sources() or report["profile"] != profile["manifest"]:
        raise ValueError("live checker/profile identity differs")
    files = [stable.file_ref(path) for path in sorted(root.rglob("*")) if path.is_file()]
    dependencies = [*profile["fixed"]["bindings"], profile["manifest"]]
    dependencies += [r["copy"] for r in profile["copies"]]
    dependencies += [
        *capture["producer_sources"],
        capture["python_runtime"]["executable"],
        *capture["python_runtime"]["loaded"],
    ]
    for row in [*files, *dependencies, *report["originals"], *report["sources"]]:
        exact(row)
    receipt = {
        "schema": SCHEMA,
        "version": 1,
        "config": report["config"],
        "root": str(root),
        "originals": report["originals"],
        "snapshots": report["snapshots"],
        "report": stable.file_ref(report_path),
        "profile": profile["manifest"],
        "query": kernel["source"],
        "object": kernel["object"],
        "sources": report["sources"],
        "files": files,
        "dependencies": dependencies,
        "claims": dict(CLAIMS),
        "roots": roots,
    }
    write(Path(destination), receipt)
    return receipt


def validate(receipt):
    closed(receipt, KEYS)
    if receipt["schema"] != SCHEMA or type(receipt["version"]) is not int or receipt["version"] != 1:
        raise ValueError("unknown/legacy pipeline receipt version")
    closed(receipt["config"], {"factor", "tile", "index_bits", "mode"})
    settings = receipt["config"]
    config(settings["factor"], settings["tile"], settings["index_bits"], settings["mode"])
    for row in [
        *receipt["originals"],
        *receipt["snapshots"],
        receipt["report"],
        receipt["profile"],
        receipt["query"],
        receipt["object"],
        *receipt["sources"],
        *receipt["files"],
        *receipt["dependencies"],
    ]:
        reference_row(row)
    if not valid_claims(receipt["claims"]) or len(receipt["originals"]) != 5 or len(receipt["snapshots"]) != 5:
        raise ValueError("complete originals/snapshots and exact conditional claims required")
    if receipt["sources"] != sources():
        raise ValueError("current checker/helper source identity differs")
    profile = prepare()
    if receipt["profile"] != profile["manifest"]:
        raise ValueError("selected fixed profile identity differs")
    for row in collect_refs(receipt):
        exact(row)
    root = Path(receipt["root"]).resolve()
    files = [stable.file_ref(path) for path in sorted(root.rglob("*")) if path.is_file()]
    if files != receipt["files"]:
        raise ValueError("closed retained artifact roster differs")
    report = json.loads(Path(receipt["report"]["path"]).read_text())
    if (
        report["schema"] != "merlin.graph-state-pipeline.boundary.v1"
        or report["status"] != "conditional_graph_checked"
        or report["qualified"] is not False
        or report["sources"] != receipt["sources"]
        or report["profile"] != receipt["profile"]
        or report["exact_roots"] != receipt["roots"]
        or report["originals"] != receipt["originals"]
        or report["snapshots"] != receipt["snapshots"]
        or report["config"] != settings
        or not valid_claims(report["claims"])
    ):
        raise ValueError("original/report/config binding differs")
    kernel = json.loads(Path(report["kernel"]["path"]).read_text())
    if (
        kernel["source"] != receipt["query"]
        or kernel["object"] != receipt["object"]
        or kernel["roots"] != receipt["roots"]
    ):
        raise ValueError("exact proof query/object/root binding differs")
    capture = json.loads(Path(report["projection"]["path"]).read_text())
    query, roots, _ = author(capture, settings["mode"])
    if query != Path(receipt["query"]["path"]).read_text() or roots != receipt["roots"]:
        raise ValueError("retained query differs from deterministic complete projection author")
    if kernel["status"] != "conditional_graph_checked" or kernel["audit"]["status"] != "checked":
        raise ValueError("retained report does not claim an actual audited kernel result")
    if kernel["returncode"] != 0 or kernel["qualified"] is not False or kernel["profile"] != receipt["profile"]:
        raise ValueError("kernel result/profile/qualification binding differs")
    buffers = {Path(row["path"]).name: row for row in kernel["audit"]["buffers"]}
    if (
        kernel["audit"]["raw"] != json.loads(Path(buffers["audit.stdout"]["path"]).read_text())
        or kernel["audit"]["statements"]
        != json.loads(Path(buffers["statements.stdout"]["path"]).read_text())["statements"]
    ):
        raise ValueError("audit report differs from exact retained output")
    expected_dependencies = [
        *profile["fixed"]["bindings"],
        profile["manifest"],
        *[r["copy"] for r in profile["copies"]],
        *capture["producer_sources"],
        capture["python_runtime"]["executable"],
        *capture["python_runtime"]["loaded"],
    ]
    if receipt["dependencies"] != expected_dependencies:
        raise ValueError("complete dependency roster differs")
    if kernel["audit"]["raw"]["violations"] or set(kernel["audit"]["raw"]["axioms_used"]) - {
        "propext",
        "Classical.choice",
        "Quot.sound",
    }:
        raise ValueError("retained audit violates standard-three policy")
    return report, capture, kernel, profile


def replay(receipt_path, destination, *, reference):
    receipt_path = Path(receipt_path).resolve()
    receipt = json.loads(receipt_path.read_text())
    report, old_capture, old_kernel, profile = validate(receipt)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    settings = receipt["config"]
    paths = [Path(row["path"]) for row in receipt["originals"]]
    fresh = projection(paths, destination / "reproject", Path(reference).resolve(), settings)
    expected, roots, _ = author(fresh, settings["mode"])
    semantic_keys = (
        "schema",
        "revision",
        "factor",
        "tile",
        "index_bits",
        "stages",
        "index_operations",
        "each_stage_changed",
    )
    if {k: fresh[k] for k in semantic_keys} != {k: old_capture[k] for k in semantic_keys}:
        raise ValueError("complete semantic projection/headers differ from fresh original buffers")
    if (
        fresh["producer_sources"] != old_capture["producer_sources"]
        or fresh["python_runtime"] != old_capture["python_runtime"]
    ):
        raise ValueError("complete actual producer/parser/Python identity roster differs")
    if expected != Path(receipt["query"]["path"]).read_text() or roots != receipt["roots"]:
        raise ValueError("actual original buffers do not regenerate the retained complete query")
    # Reproduce the retained object at its exact original source path. Output
    # is private to replay; no original/frozen artifact is overwritten.
    source = Path(receipt["query"]["path"])
    target = destination / "retained-reproduction.olean"
    environment = {
        **os.environ,
        "LEAN_PATH": os.pathsep.join((str(source.parent), str(profile["lib"]))),
        "LEAN_SYSROOT": str(stable.LEAN.parent.parent),
        "LEAN_NUM_THREADS": "1",
    }
    rebuilt = execute(
        [source, "-o", target],
        source.parent,
        environment,
        destination / "reproduce.stdout",
        destination / "reproduce.stderr",
    )
    if rebuilt.returncode or stable.file_ref(target)["sha256"] != receipt["object"]["sha256"]:
        raise ValueError("retained object is not the exact original-query result")
    result = check_query(expected, destination / "fresh", roots)
    if result["status"] != "conditional_graph_checked":
        raise ValueError("fresh kernel/audit did not check")
    if result["audit"]["statements"] != old_kernel["audit"]["statements"]:
        raise ValueError("fresh exact-root theorem statements differ")

    def normalized(raw, query):
        return {
            **raw,
            "imports": [
                {
                    **r,
                    "olean": "$EXACT_QUERY_OBJECT"
                    if Path(r["olean"]).resolve() == Path(query).resolve()
                    else r["olean"],
                }
                for r in raw["imports"]
            ],
        }

    if normalized(result["audit"]["raw"], result["object"]["path"]) != normalized(
        old_kernel["audit"]["raw"], receipt["object"]["path"]
    ):
        raise ValueError("fresh exact transitive audit/import roster differs")
    for row in collect_refs(receipt):
        exact(row)
    fixed_context()
    response = {
        "schema": "merlin.graph-state-pipeline.fresh-replay.v1",
        "status": "conditional_graph_checked",
        "receipt": stable.file_ref(receipt_path),
        "fresh_report": stable.file_ref(destination / "fresh/report.json"),
        "retained_object_reproduced": True,
        "exact_original_query_path_reproduced": True,
        "claims": dict(CLAIMS),
        "qualified": False,
        "host_transport_proved_in_Lean": False,
    }
    write(destination / "report.json", response)
    return response
