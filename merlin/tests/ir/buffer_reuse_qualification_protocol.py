"""Actual buffer request/completion guards; no mocked positive proof path."""

import argparse
import copy
import dataclasses
import json
import pickle
from pathlib import Path
from unittest.mock import patch

from loop_transform_qualification import save

from merlin.verify import buffer_reuse_gate as gate
from merlin.verify.buffer_reuse_intake import ALLOCATOR, canonical, prepare_chain
from merlin.verify.loop_transform_intake import file_ref


def request_controls(candidate, output):
    """Use actual reconstruction with a trap forbidding ANY native launch."""
    output.mkdir(parents=True, exist_ok=False)
    hint = json.loads(candidate.hint_json)
    hint["adapter"]["step_binding"]["privateViews"][0]["object"] = 999
    caller_hint = json.loads(candidate.hint_json)
    caller_hint["adapter"]["observers"][0]["buffer"]["name"] = 999
    parts = json.loads(candidate.query_parts_json)
    parts[1]["imports"] = ["ForgedModule"]
    source_map = json.loads(candidate.sources_json)
    source_map[0]["sha256"] = "0" * 64
    raw = json.loads(candidate.raw_chain_json)
    raw[1], raw[2] = raw[2], raw[1]
    variants = {
        "query": {"query": candidate.query + "\n-- unbound query\n"},
        "roots": {"roots": candidate.roots[:-1]},
        "part_import_map": {"query_parts_json": canonical(parts)},
        "source_map": {"sources_json": canonical(source_map)},
        "private_id_hint": {"hint_json": canonical(hint)},
        "caller_binding_hint": {"hint_json": canonical(caller_hint)},
        "entry_roster": {"entry_roster_json": "{}"},
        "middle_join": {"raw_chain_json": canonical(raw)},
        "allocator_finite": {"allocator": "finite_capacity"},
        "relation_full_error_world": {"relation": "full_error_world_equivalence"},
        "index32": {"index_bits": 32},
        "boolean_index": {"index_bits": True},
        "source_only_substitution": {"source": candidate.source + "\n// substituted original\n"},
        "final_only_substitution": {"final": candidate.final + "\n// substituted final\n"},
    }
    rows = []
    for label, fields in variants.items():
        forged = dataclasses.replace(candidate, **fields)
        destination = output / "forbidden-native" / label
        with patch.object(gate, "_run", side_effect=AssertionError("native launch for forged request")):
            result = gate.check(forged, lean=Path("/no-native-tool"), destination=destination)
        if result.status != "unavailable" or result.completion is not None or destination.exists():
            raise AssertionError("forged request admitted: " + label)
        rows.append({"name": label, "outcome": result.status, "reason": result.reason, "native_launched": False})
    constructed = gate.Completion(candidate.identity, "/no-saved-authority")
    for value in [None, {"status": "checked"}, constructed, copy.copy(constructed), copy.deepcopy(constructed)]:
        called = []
        try:
            gate.deliver(value, candidate, called.append)
        except gate.Unavailable:
            pass
        else:
            raise AssertionError("constructed/copied/saved completion delivered")
        if called:
            raise AssertionError("consumer called without live proof")
    try:
        pickle.dumps(constructed)
    except TypeError:
        pass
    else:
        raise AssertionError("completion serialized")
    for allocator in ["finite_capacity", "observable", "", None]:
        try:
            prepare_chain(tuple(json.loads(candidate.raw_chain_json)), allocator=allocator)
        except ValueError:
            pass
        else:
            raise AssertionError("unapproved allocator admitted")
    save(
        output / "RESULT.json",
        {
            "status": "SDK_FREE_HOST_GUARDS_PASSED_NO_PROOF_AUTHORITY",
            "forgeries": rows,
            "constructed_saved_copied_completions": 5,
            "serialization_refused": True,
            "allocator_modes_refused": 4,
            "actual_live_completion_copy_test": "PENDING_REAL_GATE_SUCCESS",
            "human_approved": False,
        },
    )


def artifact_controls(completion, candidate, output):
    """Dedicated real checked scratch only; exact intentional tamper/restoration.

    This is one live proof, not eight new positives. No native job is retried.
    Every before/changed/restored byte reference is retained. Original qualified
    outputs must not be supplied: caller owns this explicit attack scratch.
    """
    output.mkdir(parents=True, exist_ok=False)
    report_path = Path(completion.report)
    directory = report_path.parent.resolve()
    report = json.loads(report_path.read_text())
    gate.validate_completion(completion, candidate)
    targets = {
        "canonical_query": Path(report["query"]["path"]),
        "query_part": Path(report["generated_queries"][3]["source"]["path"]),
        "query_object": Path(report["generated_queries"][-1]["object"]["path"]),
        "raw_middle": Path(report["actual4"][1]["path"]),
        "raw_process_log": Path(report["process_evidence"][0]["path"]),
        "report": report_path,
        "rehashed_import_map_report": report_path,
    }
    rows = []
    for name, target in targets.items():
        if not target.resolve().is_relative_to(directory):
            raise ValueError("attack target escaped dedicated own scratch")
        original = target.read_bytes()
        before = file_ref(target)
        if name == "rehashed_import_map_report":
            changed = json.loads(original)
            changed["imports"][0]["sha256"] = "0" * 64
            data = canonical(changed).encode()
        else:
            data = original + b"\nSTALE"
        called = []
        try:
            target.write_bytes(data)
            changed_ref = file_ref(target)
            try:
                gate.deliver(completion, candidate, called.append)
            except gate.Unavailable as error:
                reason = str(error)
            else:
                raise AssertionError("stale artifact delivered: " + name)
        finally:
            target.write_bytes(original)
        if called or file_ref(target) != before:
            raise AssertionError("stale consumer/restoration failure")
        gate.validate_completion(completion, candidate)
        rows.append(
            {
                "kind": name,
                "before": before,
                "changed": changed_ref,
                "restored": file_ref(target),
                "consumer_called": False,
                "reason": reason,
            }
        )
        save(output / "ARTIFACTS.json", rows)
    for name in ["extra_object", "object_symlink"]:
        extra = Path(report["library_root"]) / ("F_ATTACK_" + name + ".olean")
        if extra.exists() or extra.is_symlink():
            raise ValueError("exclusive attack filename already exists")
        called = []
        try:
            if name == "extra_object":
                with extra.open("xb") as stream:
                    stream.write(b"not an object, never loaded")
            else:
                extra.symlink_to(report["generated_queries"][-1]["object"]["path"])
            try:
                gate.deliver(completion, candidate, called.append)
            except gate.Unavailable as error:
                reason = str(error)
            else:
                raise AssertionError("extra/symlink object delivered")
        finally:
            extra.unlink()
        if called:
            raise AssertionError("consumer received extra/symlink proof tree")
        gate.validate_completion(completion, candidate)
        rows.append({"kind": name, "consumer_called": False, "reason": reason, "temporary_member_removed": True})
        save(output / "ARTIFACTS.json", rows)
    for token, request in [(copy.copy(completion), candidate), (completion, dataclasses.replace(candidate))]:
        try:
            gate.deliver(token, request, lambda _: (_ for _ in ()).throw(AssertionError("copied authority delivered")))
        except gate.Unavailable:
            pass
        else:
            raise AssertionError("copy gained live authority")
    save(
        output / "RESULT.json",
        {
            "status": "DEDICATED_REAL_LIVE_ARTIFACT_GUARDS_PASSED",
            "attacks": len(rows),
            "native_proofs": 1,
            "copied_completion_refused": True,
            "copied_candidate_refused": True,
            "exact_bytes_restored_for_independent_controls": True,
            "human_approved": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    buffers = tuple((args.corpus / "positives/retained_two_epochs" / f"stage{i}.mlir").read_text() for i in range(4))
    request_controls(prepare_chain(buffers, allocator=ALLOCATOR), args.output)
