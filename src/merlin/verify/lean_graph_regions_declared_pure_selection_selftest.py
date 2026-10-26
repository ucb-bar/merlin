"""Isolated metadata-only selection/refusal tests; no SDK/runtime file opens.

The original F/W JSON manifests may be read against their original document
identities. All negative pins are counterfactual DATA, never new runtime pins.
Run directly with Python -I -S -B; no Merlin initializer or native process.
"""

from __future__ import annotations

import argparse
import copy
import json
import runpy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    api = runpy.run_path(
        str(args.sources / "lean_graph_regions_declared_loader_plan.py"), run_name="_original_metadata"
    )
    selector = runpy.run_path(str(args.sources / "lean_graph_regions_declared_pure_selection.py"))
    document = api["parse_json"](args.selection.read_bytes())
    provenance = api["parse_json"](args.provenance.read_bytes())
    sdk = api["parse_json"](api["read_original"](provenance["original_F_SDK_manifest"]))
    original_w = api["parse_json"](api["read_original"](provenance["original_W_fixed_context"]))

    def invoke(doc, f=sdk, w=original_w):
        return selector["select"](doc, provenance, f, w, strict=api["strict_ref"], references=api["references"])

    # Freeze the entire fixed roster before exercising any selected case.
    roster = [
        "all653_five_file_families_and_original11",
        "missing_olean",
        "missing_private",
        "missing_server",
        "missing_ilean",
        "missing_source",
        "missing_runtime",
        "wrong_pin",
        "wrong_ref_type",
        "extra_auxiliary_swap",
        "extra_held_swap",
        "wrong_parent_manifest",
        "nonInit_module",
        "conflicting_F_pin",
        "conflicting_W_pin",
        "missing_independent_W_compiled",
        "extra_runtime",
    ]
    (args.output / "ROSTER.json").write_text(json.dumps(roster, indent=2) + "\n")
    results = []

    def check(name, operation, expected):
        try:
            value = operation()
            accepted, error = True, None
        except (selector["SelectionRefusal"], api["Unavailable"]) as caught:
            value, accepted, error = None, False, str(caught)
        results.append(dict(case=name, passed=accepted == expected, accepted=accepted, error=error))
        return value

    positive = check(roster[0], lambda: invoke(document), True)
    if positive is None:
        (args.output / "FIRST-ERROR.json").write_text(json.dumps(results[-1], indent=2) + "\n")
        raise SystemExit(1)
    assert positive["family_counts"] == {
        suffix: 653 for suffix in (".olean", ".olean.private", ".olean.server", ".ilean", ".lean")
    }
    assert positive["runtime_count"] == 11 and positive["independently_bound_compiled"] == 1959
    selected = document["selected"]
    for name, suffix in zip(roster[1:6], (".olean", ".olean.private", ".olean.server", ".ilean", ".lean"), strict=True):
        doc = copy.deepcopy(document)
        path = next(path for path in selected if path.endswith("/Init" + suffix))
        del doc["selected"][path]
        check(name, lambda doc=doc: invoke(doc), False)
    doc = copy.deepcopy(document)
    del doc["selected"][document["runtime_refs"][0]["path"]]
    check("missing_runtime", lambda: invoke(doc), False)
    doc = copy.deepcopy(document)
    first = next(iter(doc["selected"]))
    doc["selected"][first]["sha256"] = "0" * 64
    check("wrong_pin", lambda: invoke(doc), False)
    doc = copy.deepcopy(document)
    doc["selected"][first]["bytes"] = True
    check("wrong_ref_type", lambda: invoke(doc), False)
    doc = copy.deepcopy(document)
    del doc["selected"][first]
    auxiliary, pin = next(iter(doc["excluded_auxiliary_refs"].items()))
    doc["selected"][auxiliary] = pin
    check("extra_auxiliary_swap", lambda: invoke(doc), False)
    doc = copy.deepcopy(document)
    del doc["selected"][first]
    held = document["held_paths_not_read_or_selected"][-1]
    doc["selected"][held] = dict(path=held, bytes=1, sha256="0" * 64)
    check("extra_held_swap", lambda: invoke(doc), False)
    doc = copy.deepcopy(document)
    doc["original_F_manifest"]["sha256"] = "0" * 64
    check("wrong_parent_manifest", lambda: invoke(doc), False)
    doc, f = copy.deepcopy(document), copy.deepcopy(sdk)
    doc["original_F_modules"][0] = f["modules"][0] = "Std.Time.Format.Basic"
    check("nonInit_module", lambda: invoke(doc, f), False)
    f = copy.deepcopy(sdk)
    f["files"]["synthetic-conflicting-original-reference"] = dict(selected[first], sha256="0" * 64)
    check("conflicting_F_pin", lambda: invoke(document, f), False)
    w = copy.deepcopy(original_w)
    w["synthetic-conflicting-original-reference"] = dict(selected[first], sha256="0" * 64)
    check("conflicting_W_pin", lambda: invoke(document, sdk, w), False)

    def omit_reference(value):
        if type(value) is dict:
            if set(value) == {"path", "sha256", "bytes"} and value["path"] == first:
                return None
            return {key: omit_reference(item) for key, item in value.items()}
        if type(value) is list:
            return [omit_reference(item) for item in value]
        return value

    check("missing_independent_W_compiled", lambda: invoke(document, sdk, omit_reference(original_w)), False)
    doc = copy.deepcopy(document)
    doc["runtime_refs"].append(dict(path="/synthetic/extra-runtime", bytes=1, sha256="0" * 64))
    check("extra_runtime", lambda: invoke(doc), False)
    assert [result["case"] for result in results] == roster
    summary = dict(
        results=results,
        passed=sum(result["passed"] for result in results),
        total=len(results),
        selected_count=3276,
        family_counts=positive["family_counts"],
        independently_bound_compiled=1959,
        sdk_runtime_file_reads=0,
        native_invocations=0,
        runtime_hashes_refreshed=False,
        v9_promoted=False,
        qualification=False,
    )
    (args.output / "RESULT.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}))
    raise SystemExit(0 if all(result["passed"] for result in results) else 1)


if __name__ == "__main__":
    main()
