"""Reviewed provisional readout staging; preserves denial and existing prices."""

from __future__ import annotations

import copy
import json

from .feedback import joint_performance as PERF
from .joint_provider_revision import _replays
from .joint_scoped_revision import _denial
from .joint_structural_revision import _verify_file

KIND = "provisional_readout_staging"
SCHEMA = "merlin.readout_staging_review.v1"
PARTIAL_KIND = "provisional_partial_row_staging"
PARTIAL_SCHEMA = "merlin.partial_row_staging_review.v1"
LOAD_KIND = "provisional_load_frontend_arrival"
LOAD_SCHEMA = "merlin.load_frontend_arrival_review.v1"
BOOTSTRAP_KIND = "provisional_loader_cache_bootstrap"
BOOTSTRAP_SCHEMA = "merlin.loader_cache_bootstrap_review.v1"
BOOTSTRAP_STATE = {
    "schema": "merlin.tsi_elf_cache_bootstrap.v1",
    "loader": "elf64_le_riscv_exec_pt_load",
    "cache": "l2",
    "chunk_align": 4,
    "chunk_max_size": 1024,
    "replacement_scope": "one_loaded_line_per_set",
}
PARTIAL_SELECTORS = {
    "accumulator_streamed_partial_raw_rows": 1,
    "refuse_masked_write_approximation": 1,
    "scratchpad_partial_row_loads": 1,
}
SELECTORS = {
    "readout_request_stage_split": 1,
    "accumulator_streamed_strided_rows": 1,
}
ASSUMPTIONS = {
    "writer_a_ready_with_transaction_credit": 1,
    "readout_command_frontend_no_backpressure": 1,
}


def _delta(before, after, audit):
    expected = copy.deepcopy(before)
    target = expected["target"]
    if (
        target["scheduling"].get("version") != 5
        or target["scheduling"].get("accumulator_streamed_readout") != 1
        or target["hardware"].get("normalizations") != 0
    ):
        raise ValueError("readout staging requires the reviewed streamed provisional base")
    additions = audit.get("additions")
    if not isinstance(additions, dict) or set(additions) != {
        "frontend_stages",
        "probe_request_cycles",
        "write_response",
    }:
        raise ValueError("readout staging requires exactly the reviewed additional fields")
    response = additions["write_response"]
    if (
        not isinstance(response, dict)
        or set(response) != {"schema", "scope", "block_bytes", "pipeline_cycles"}
        or response["schema"] != "merlin.provisional_probe_write_response.v1"
        or response["scope"] != "offset_zero_wormhole_no_backpressure"
        or response["block_bytes"] != expected["memory"]["caches"]["l1d"]["line_bytes"]
        or any(
            type(v) is not int or not 0 < v <= 4096
            for v in (response["pipeline_cycles"], additions["frontend_stages"], additions["probe_request_cycles"])
        )
    ):
        raise ValueError("readout staging has unsupported geometry or stage values")
    for section, values in (
        ("scheduling", SELECTORS | {"readout_probe_request_cycles": additions["probe_request_cycles"]}),
        ("assumptions", ASSUMPTIONS),
        ("hardware", {"readout_extra_command_registered_stages": additions["frontend_stages"]}),
    ):
        dest = target.setdefault(section, {})
        if set(dest) & set(values):
            raise ValueError("readout staging cannot overwrite an existing mechanism")
        dest.update(values)
    probe = expected["memory"]["caches"]["l1d"]["coherence_probe_service"]
    if "write_response" in probe:
        raise ValueError("readout staging cannot replace existing response costs")
    probe["write_response"] = response
    expected["domain"]["feature_engine_sha256"] = after["domain"]["feature_engine_sha256"]
    expected["libraries"] = after["libraries"]
    if expected != after:
        raise ValueError("readout staging changes existing prices, capacity, scope or unreviewed semantics")



def _partial_delta(before, after, audit):
    """Only extend the reviewed row protocol; never change prices or scope."""
    expected = copy.deepcopy(before)
    target = expected["target"]
    schedule = target["scheduling"]
    if (any(schedule.get(k) != v for k, v in SELECTORS.items())
            or schedule.get("version") != 5
            or schedule.get("scratchpad_row_loads") != 1
            or schedule.get("scratchpad_request_admission") != 1
            or target["hardware"].get("reader_beat_bytes") != 16):
        raise ValueError("partial rows require the reviewed row request/readout base")
    if audit.get("additions") != {"reader_request_capture_stages": 1}:
        raise ValueError("partial rows require exactly the source-derived request capture stage")
    if set(schedule) & set(PARTIAL_SELECTORS) or "reader_request_capture_stages" in target["hardware"]:
        raise ValueError("partial rows cannot overwrite existing protocol facts")
    schedule.update(PARTIAL_SELECTORS)
    target["hardware"]["reader_request_capture_stages"] = 1
    expected["libraries"] = after["libraries"]
    # The observer and every existing field, including source identity, stay
    # fixed. Only the timing provider implements these additional row stages.
    if expected != after:
        raise ValueError("partial rows change prices, scope, facts or unreviewed semantics")

def _load_frontend_delta(before, after, audit):
    """Separate the reviewed load arrival stages from service occupancy."""
    expected = copy.deepcopy(before)
    target = expected["target"]
    schedule, hardware = target["scheduling"], target["hardware"]
    if (schedule.get("version") != 5
            or any(schedule.get(k) != 1 for k in (
                "scratchpad_row_loads", "scratchpad_request_admission",
                "scratchpad_partial_row_loads", "readout_request_stage_split"))
            or hardware.get("reader_request_capture_stages") != 1
            or hardware.get("readout_extra_command_registered_stages") != 6):
        raise ValueError("load frontend requires the reviewed row and command-stage base")
    if audit.get("additions") != {"load_extra_command_registered_stages": 6}:
        raise ValueError("load frontend requires the source-derived six registered stages")
    for section, values in (
        ("scheduling", {"scratchpad_load_frontend_arrival": 1}),
        ("hardware", {"load_extra_command_registered_stages": 6}),
        ("assumptions", {"load_command_frontend_no_backpressure": 1}),
    ):
        dest = target.setdefault(section, {})
        if set(dest) & set(values):
            raise ValueError("load frontend cannot overwrite existing mechanisms")
        dest.update(values)
    expected["libraries"] = after["libraries"]
    if expected != after:
        raise ValueError("load frontend changes prices, scope, facts or unreviewed semantics")


def _bootstrap_delta(before, after, audit):
    """Bind loader-derived cache state without permitting warm-address hints."""
    expected = copy.deepcopy(before)
    memory = expected["memory"]
    if "initial_state" in memory or "l2" not in memory.get("caches", {}):
        raise ValueError("loader bootstrap requires an unseeded shared-cache model")
    source_receipt = audit["loader_source_receipt"]
    PERF._read(source_receipt)
    state = BOOTSTRAP_STATE | {"source_receipt_sha256": source_receipt["sha256"]}
    if (audit.get("additions") != {"initial_state": state}
            or audit.get("private_or_firesim_labels_used") is not False):
        raise ValueError("loader bootstrap requires source-derived bounded initial state")
    memory["initial_state"] = state
    expected["domain"]["feature_engine_sha256"] = after["domain"]["feature_engine_sha256"]
    if expected != after:
        raise ValueError("loader bootstrap changes prices, provider, scope or unreviewed semantics")


def verify(old, new, evidence, *, previous, current, partial_rows=False, load_frontend=False,
           loader_bootstrap=False):
    if sum((partial_rows, load_frontend, loader_bootstrap)) > 1:
        raise ValueError("staging revisions must select one explicit mechanism")
    schema, audit_schema, origin, delta = (
        (BOOTSTRAP_SCHEMA, "merlin.public_loader_cache_bootstrap_facts.v1",
         "source_derived_loader_state_no_price_changes", _bootstrap_delta) if loader_bootstrap else
        (LOAD_SCHEMA, "merlin.public_load_frontend_arrival_facts.v1",
         "source_derived_load_frontend_no_price_changes", _load_frontend_delta) if load_frontend else
        (PARTIAL_SCHEMA, "merlin.public_partial_row_staging_facts.v1",
         "source_derived_row_protocol_no_price_changes", _partial_delta) if partial_rows else
        (SCHEMA, "merlin.public_readout_staging_facts.v1",
         "source_frontend_and_public_gsim_response_stages", _delta)
    )
    if (
        evidence.get("schema") != schema
        or evidence.get("status") != "provisional_untuned"
        or evidence.get("previous_runtime") != previous
        or evidence.get("runtime") != current
        or evidence.get("fit_authorized") is not False
        or evidence.get("promotion_authorized") is not False
    ):
        raise ValueError("readout staging requires exact provisional review")
    before, after = json.loads(old.model_json), json.loads(new.model_json)
    specs = [PERF._read(previous), PERF._read(current)]
    mutable = {"review_id", "libraries", "model", "model_sha256"}
    if loader_bootstrap or not (partial_rows or load_frontend):
        mutable.update({"observer", "source_manifest"})
    if {k: v for k, v in specs[0].items() if k not in mutable} != {
        k: v for k, v in specs[1].items() if k not in mutable
    }:
        raise ValueError("readout staging changes functional contract, facts or execution window")
    for left, right in ((before["libraries"], after["libraries"]), (specs[0]["libraries"], specs[1]["libraries"])):
        if len(left) != 2 or len(right) != 2 or left[0] != right[0]:
            raise ValueError("readout staging changes functional extension")
        if loader_bootstrap and left != right:
            raise ValueError("loader bootstrap cannot replace the timing provider")
    for name, model, spec in zip(("previous", "current"), (before, after), specs):
        mp = evidence["previous_model" if name == "previous" else "model"]
        source = evidence["source_manifests"][name]
        if (
            mp != {"path": spec["model"], "sha256": spec["model_sha256"]}
            or PERF._read(mp) != model
            or source["path"] != spec["source_manifest"]
            or PERF._read(source).get("sha256") != model["domain"]["feature_engine_sha256"]
        ):
            raise ValueError("readout staging model or observer source binding differs")
    audit = PERF._read(evidence["structural_audit"])
    if (
        audit.get("schema") != audit_schema
        or audit.get("qualification_issued") is not False
        or audit.get("parameter_origin") != origin
    ):
        raise ValueError("readout staging needs explicit public structural evidence")
    for field in (("loader_sources" if loader_bootstrap else "rtl_sources"), "public_observations"):
        leaves = audit.get(field)
        if not isinstance(leaves, list) or not 1 <= len(leaves) <= 32:
            raise ValueError("readout staging lacks bounded source/observation pins")
        for pin in leaves:
            _verify_file(pin)
    delta(before, after, audit)
    build = PERF._read(evidence["build"])
    if (
        not isinstance(build.get("observer"), dict)
        or build["observer"].get("path") != specs[1]["observer"]
        or build.get("provider") != {"path": specs[1]["libraries"][1], "sha256": after["libraries"][1]}
        or build.get("source_identity") != after["domain"]["feature_engine_sha256"]
    ):
        raise ValueError("readout staging build differs from selected implementation")
    for field in ("observer", "provider"):
        _verify_file(build[field])
    sources = build.get("sources")
    if not isinstance(sources, list) or not 1 <= len(sources) <= 64:
        raise ValueError("readout staging lacks compiled source closure")
    for pin in sources:
        _verify_file(pin)
    _replays(evidence)
    _denial(evidence)
