"""Closed live receipts: complete originals, canonical model, query/object/audit.

Replay reprojects originals and reconstructs the proof before accepting retained
evidence. Hashes bind bytes, not issuer authenticity or physical runtime delivery.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import MappingProxyType

from . import lean_graph_state as stable
from .lean_graph_state_integer import author, configuration, intake
from .lean_graph_state_integer_kernel import context
from .lean_graph_state_integer_proof import check_query, execute, prepare, sources
from .lean_graph_state_pipeline_kernel import exact, write
from .lean_graph_state_pipeline_receipt import closed, collect_refs, reference_row

SCHEMA = "merlin.graph-state-integer.live-receipt.v1"
CLAIMS = MappingProxyType(
    {
        "relation": "canonical-entry complete return/error/terminal-world correspondence",
        "runtime_intermediate_typing_derived": True,
        "local_result_contracts_required": True,
        "canonical_entries_required": True,
        "python_projection_proved": False,
        "review_approved": False,
        "public_acceptance_issued": False,
        "native_fidelity_proved": False,
        "poison_freedom_proved": False,
        "source_success_for_all_models": False,
        "all_graph_passes_verified": False,
        "full_model_claim": False,
        "broker_integration": False,
        "old_failure_promoted": False,
    }
)
KEYS = {
    "schema",
    "version",
    "root",
    "settings",
    "mode",
    "model",
    "originals",
    "snapshots",
    "report",
    "capture",
    "profile",
    "query",
    "object",
    "sources",
    "files",
    "dependencies",
    "claims",
    "roots",
}


def valid_claims(value):
    return (
        type(value) is dict
        and set(value) == set(CLAIMS)
        and all(type(value[key]) is type(expected) and value[key] == expected for key, expected in CLAIMS.items())
    )


def load(path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("duplicate receipt key")
            result[key] = value
        return result

    def noninteger(_):
        raise ValueError("noninteger JSON numeric encoding refused")

    return json.loads(
        Path(path).read_text(), object_pairs_hook=pairs, parse_float=noninteger, parse_constant=noninteger
    )


def dependencies(profile, capture):
    return [
        *profile["fixed"]["bindings"],
        profile["manifest"],
        *[row["copy"] for row in profile["copies"]],
        *capture["producer_sources"],
        capture["python_runtime"]["executable"],
        *capture["python_runtime"]["loaded"],
    ]


def files(root):
    return [stable.file_ref(path) for path in sorted(Path(root).rglob("*")) if path.is_file()]


def issue(report_path, destination):
    report_path = Path(report_path).resolve()
    report = load(report_path)
    if report["status"] != "conditional_graph_checked" or report["qualified"] is not False:
        raise ValueError("actual conditional kernel check required before issuing a receipt")
    proof = load(report["proof"]["path"])
    capture = load(report["capture"]["path"])
    profile = prepare()
    query, roots, model = author(capture, report["mode"])
    if (
        proof["status"] != "conditional_graph_checked"
        or proof["audit"]["status"] != "checked"
        or proof["profile"] != profile["manifest"]
        or proof["roots"] != roots
        or Path(proof["source"]["path"]).read_text() != query
        or report["sources"] != sources()
        or report["model"] != model
    ):
        raise ValueError("exact complete projection/query/model/profile/root binding differs")
    receipt = {
        "schema": SCHEMA,
        "version": 1,
        "root": str(report_path.parent),
        "settings": report["settings"],
        "mode": report["mode"],
        "model": model,
        "originals": report["originals"],
        "snapshots": capture["buffers"],
        "report": stable.file_ref(report_path),
        "capture": report["capture"],
        "profile": profile["manifest"],
        "query": proof["source"],
        "object": proof["object"],
        "sources": sources(),
        "files": files(report_path.parent),
        "dependencies": dependencies(profile, capture),
        "claims": dict(CLAIMS),
        "roots": roots,
    }
    validate(receipt)
    write(Path(destination), receipt)
    return receipt


def validate(receipt):
    closed(receipt, KEYS)
    if receipt["schema"] != SCHEMA or type(receipt["version"]) is not int or receipt["version"] != 1:
        raise ValueError("unknown integer receipt version")
    closed(receipt["settings"], {"factor", "tile", "index_bits"})
    configuration(**receipt["settings"])
    if receipt["mode"] not in ("graph", "initialized-symbolic-provider-v1"):
        raise ValueError("unapproved/unknown checker model mode")
    if type(receipt["root"]) is not str or not Path(receipt["root"]).is_absolute():
        raise ValueError("exact original live root required; laptop path edits do not qualify it")
    if not valid_claims(receipt["claims"]):
        raise ValueError("conditional claim flags/types differ")
    if len(receipt["originals"]) not in (2, 6) or len(receipt["originals"]) != len(receipt["snapshots"]):
        raise ValueError("complete original/snapshot pair or six-buffer chain required")
    for row in collect_refs(receipt):
        reference_row(row)
        exact(row)
    profile = prepare()
    if receipt["profile"] != profile["manifest"] or receipt["sources"] != sources():
        raise ValueError("current source/fixed profile binding differs")
    if receipt["files"] != files(receipt["root"]):
        raise ValueError("closed retained artifact roster differs")
    report = load(receipt["report"]["path"])
    if (
        report["schema"] != "merlin.graph-state-integer.boundary.v1"
        or report["status"] != "conditional_graph_checked"
        or report["qualified"] is not False
        or report["originals"] != receipt["originals"]
        or report["sources"] != receipt["sources"]
        or report["capture"] != receipt["capture"]
        or report["settings"] != receipt["settings"]
        or report["mode"] != receipt["mode"]
        or report["model"] != receipt["model"]
        or report["review_approved"] is not False
        or report["native_fidelity_proved"] is not False
        or report["poison_freedom_proved"] is not False
    ):
        raise ValueError("source/report/model/claim binding differs")
    capture = load(receipt["capture"]["path"])
    proof = load(report["proof"]["path"])
    query, roots, model = author(capture, receipt["mode"])
    if (
        proof["schema"] != "merlin.graph-state-integer.kernel.v1"
        or proof["status"] != "conditional_graph_checked"
        or proof["returncode"] != 0
        or proof["qualified"] is not False
        or proof["audit"]["status"] != "checked"
        or proof["profile"] != receipt["profile"]
        or proof["source"] != receipt["query"]
        or proof["object"] != receipt["object"]
        or proof["roots"] != receipt["roots"]
        or roots != receipt["roots"]
        or query != Path(receipt["query"]["path"]).read_text()
        or model != receipt["model"]
        or capture["buffers"] != receipt["snapshots"]
        or dependencies(profile, capture) != receipt["dependencies"]
    ):
        raise ValueError("exact proof/query/object/projection/root/model roster differs")
    buffers = {Path(row["path"]).name: row for row in proof["audit"]["buffers"]}
    if (
        proof["audit"]["raw"] != load(buffers["audit.stdout"]["path"])
        or proof["audit"]["statements"] != load(buffers["statements.stdout"]["path"])["statements"]
        or set(proof["audit"]["statements"]) != set(roots)
        or proof["audit"]["raw"]["violations"]
        or set(proof["audit"]["raw"]["axioms_used"]) - {"propext", "Classical.choice", "Quot.sound"}
        or proof["audit"]["auditor"] != profile["selection"]["auditor"]
    ):
        raise ValueError("retained exact-root/transitive audit differs")
    return report, capture, proof, profile


def replay(receipt_path, destination, *, reference, expected_originals=None):
    receipt_path = Path(receipt_path).resolve()
    receipt = load(receipt_path)
    report, capture, old, profile = validate(receipt)
    if expected_originals is not None and receipt["originals"] != expected_originals:
        raise ValueError("externally pinned original roster differs; receipt hashes are not authenticity")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    paths = [Path(row["path"]) for row in receipt["originals"]]
    fresh = intake(reference, destination / "reproject", buffers=paths, settings=receipt["settings"])
    for keys in (
        ("schema", "revision", "prefix", "pipeline", "each_stage_changed"),
        ("producer_sources", "python_runtime", "owned_projection"),
    ):
        if {k: fresh[k] for k in keys} != {k: capture[k] for k in keys}:
            raise ValueError("fresh actual complete projection/source/runtime differs")
    query, roots, model = author(fresh, receipt["mode"])
    if query != Path(receipt["query"]["path"]).read_text() or roots != receipt["roots"] or model != receipt["model"]:
        raise ValueError("actual originals do not reproduce retained query/roots/model")
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
        raise ValueError("retained object does not reproduce at its exact original query path")
    checked = check_query(query, destination / "fresh", roots)
    if checked["status"] != "conditional_graph_checked" or checked["audit"]["statements"] != old["audit"]["statements"]:
        raise ValueError("fresh complete exact-root proof differs")

    def normalized(raw, obj):
        return {
            **raw,
            "imports": [
                {**row, "olean": "$QUERY" if Path(row["olean"]).resolve() == Path(obj).resolve() else row["olean"]}
                for row in raw["imports"]
            ],
        }

    if normalized(checked["audit"]["raw"], checked["object"]["path"]) != normalized(
        old["audit"]["raw"], receipt["object"]["path"]
    ):
        raise ValueError("fresh transitive theorem/import/axiom roster differs")
    for row in collect_refs(receipt):
        exact(row)
    context()
    result = {
        "schema": "merlin.graph-state-integer.fresh-replay.v1",
        "status": "conditional_graph_checked",
        "receipt": stable.file_ref(receipt_path),
        "fresh_report": stable.file_ref(destination / "fresh/report.json"),
        "retained_object_reproduced": True,
        "exact_original_query_path_reproduced": True,
        "claims": dict(CLAIMS),
        "model": model,
        "qualified": False,
        "original_authenticity_external": True,
        "host_transport_proved_in_Lean": False,
    }
    write(destination / "report.json", result)
    return result
