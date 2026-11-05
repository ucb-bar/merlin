"""Actual finite pointwise reviewed owners/stress, with unchanged blockers.

The enclosing ledger remains an ordinary coverage producer's responsibility.
Diagnostic join controls below never construct saved release authority.
"""

import copy
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import original_pointwise_semantic_probes as B
from merlin_experiments.phase0 import original_reference_plan as P
from merlin_experiments.phase0 import original_reference_requirements as J
from merlin_experiments.phase0 import original_semantic_review as M
from merlin_experiments.phase0 import original_semantic_review_plan as Q
from merlin_experiments.phase0 import source_preparation_release as SP

from merlin.common import invocation_record as I
from merlin.common.jsonio import canonical_json
from merlin.targetgen import original_pointwise_stress as PS


def fixture(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = fixture("pointwise_semantic_source_fixtures", "test_original_pointwise_reference_flow.py")
K = fixture("pointwise_semantic_requirement_join", "test_original_reference_requirement_join.py")
live_pointwise_originals, observed = F.live_pointwise_originals, F.observed
BUDGET = Q.OriginalPointwiseReviewBudget(500000, 400000, 6000000, 100, 100, 3000000, 1000000, 180)


def context_rows(tmp_path, schema):
    """Only the tracked-file context reader seam; no live source issuer."""
    source = tmp_path / "public.cpp"
    source.write_text("public implementation context\n")
    canonical = {"checkout": str(tmp_path), "commit": "1" * 40, "path": str(source)}
    selected = tmp_path / "schema-selection.json"
    selected.write_bytes(
        canonical_json(
            {
                "schema": "merlin.independent_operator_schema_selection.v1",
                "status": "reviewed",
                "software_intake_sha256": "2" * 64,
                "namespace": "aten",
                "python": "/usr/bin/python3",
                "canonical_source": canonical,
            }
        )
    )
    references = SimpleNamespace(schema_intake=SimpleNamespace(record=lambda: {"selection_path": str(selected)}))
    context = M.R._pin(source)
    return (
        {
            "schema": schema,
            "canonical_source": canonical,
            "owners": [
                {
                    "id": name,
                    "implementation_context": [copy.deepcopy(context)]
                    if schema == Q.POINTWISE_SCHEMA
                    else copy.deepcopy(context),
                }
                for name in ("first", "second")
            ],
            "budget": {"max_context_source_bytes": 1000, "max_total_context_source_bytes": 2000},
        },
        references,
        source,
    )


@pytest.mark.parametrize("schema,per_call", [(Q.POINTWISE_SCHEMA, 1), (Q.SCHEMA, 2)])
def test_context_tracking_is_call_local_and_legacy_unchanged(tmp_path, monkeypatch, schema, per_call):
    document, references, source = context_rows(tmp_path, schema)
    tracked = []

    def track(root, path, commit):
        tracked.append((root, path, commit))
        return {"source": str(path), "commit": commit}

    monkeypatch.setattr(Q, "_tracked_source", track)
    for index in (1, 2):
        rows = Q.contexts(document, references=references, forbidden=())
        assert len(tracked) == per_call * index
        assert [row["owner"] for row in rows] == ["first", "second"]
        assert all(row["source"] == M.R._pin(source) for row in rows)
        assert rows[0]["tracked_context"] == rows[1]["tracked_context"]
        rows[0]["tracked_context"]["source"] = "mutated row"
        assert rows[1]["tracked_context"]["source"] == str(source)


@pytest.mark.parametrize("defect", ["hash", "contents", "total_bytes", "plain", "forbidden"])
def test_context_dedup_preserves_every_row_check_and_charge(tmp_path, monkeypatch, defect):
    document, references, source = context_rows(tmp_path, Q.POINTWISE_SCHEMA)
    tracked = []
    monkeypatch.setattr(Q, "_tracked_source", lambda *args: tracked.append(args) or {"source": str(source)})
    if defect == "hash":
        document["owners"][1]["implementation_context"][0]["sha256"] = "0" * 64
    elif defect == "total_bytes":
        document["budget"]["max_total_context_source_bytes"] = source.stat().st_size
    else:
        outside = defect in {"forbidden", "contents"}
        name, delegate = ("_outside", Q._outside) if outside else ("_plain", Q.R._plain)
        calls = []

        def per_row(path, *args):
            calls.append(path)
            if len(calls) == 2:
                if defect == "contents":
                    source.write_text("altered implementation context\n")
                else:
                    raise ValueError("second original context row refused")
            return delegate(path, *args)

        monkeypatch.setattr(Q if outside else Q.R, name, per_row)
    with pytest.raises(ValueError):
        Q.contexts(document, references=references, forbidden=())
    assert len(tracked) == 1


def review_selection(standard):
    references = standard.references
    original = json.loads(references.receipt_json)
    selected = json.loads(references.selection.read_bytes())
    schema = json.loads(references.schema_intake.receipt_json)
    canonical = json.loads(Path(schema["selection_path"]).read_bytes())["canonical_source"]
    defaults = M._defaults(original)
    owners = {}
    contexts = {
        "aten.relu.default": [
            "aten/src/ATen/native/Activation.cpp",
            "aten/src/ATen/native/TensorCompare.cpp",
            "aten/src/ATen/native/cpu/TensorCompareKernel.cpp",
        ],
        "aten.clamp.default": [
            "aten/src/ATen/native/TensorCompare.cpp",
            "aten/src/ATen/native/cpu/TensorCompareKernel.cpp",
        ],
        "aten.round.default": [
            "aten/src/ATen/native/UnaryOps.cpp",
            "aten/src/ATen/native/cpu/UnaryOpsKernel.cpp",
            "aten/src/ATen/cpu/vml.h",
            "aten/src/ATen/cpu/vec/vec256/vec256_float.h",
        ],
    }
    for source in original["members"]:
        if "form" not in source or "policy" not in source:
            continue
        selector = Q.selector(
            source["form"], defaults[(source["graph_path"], source["target"], source["call"]["schema"])]
        )
        key = canonical_json(selector)
        if key in owners:
            continue
        policy = P.policy(source["policy"], pointwise=True)
        rank = source["form"]["rank"]
        owners[key] = {
            "id": "pointwise_" + str(len(owners)),
            "selector": selector,
            "numerical_policy": source["policy"],
            "input_palettes": M._palette(selected, policy),
            "implementation_context": [
                M.R._pin(Path(canonical["checkout"]) / name) for name in contexts[source["target"]]
            ],
            "stress": {
                "per_member": ["pointwise_values"],
                "across_complete_cohorts": PS.required(policy, selector["parameters"]),
            },
            "stress_probes": {"profile": PS.PROFILE, "extent": 13 if rank >= 2 else 129, "max_cases": 100},
        }
    return {
        "schema": Q.POINTWISE_SCHEMA,
        "canonical_source": canonical,
        "cohorts": selected["cohorts"],
        "owners": list(owners.values()),
        "budget": dict(vars(BUDGET)),
        "execution_budget": selected["execution_budget"],
    }


def test_review_selection_mutations_do_not_alias_original_declarations(tmp_path):
    """Declaration-only seam; no source/reference or live issuer authority."""
    selected = tmp_path / "selection.json"
    selected.write_bytes(
        canonical_json(
            {
                "cohorts": {"guard": [1, 2], "withheld_transfer": [3]},
                "execution_budget": {"max_members": 150},
            }
        )
    )
    schemas = tmp_path / "schemas.json"
    schemas.write_bytes(canonical_json({"canonical_source": {"commit": "1" * 40}}))
    standard = SimpleNamespace(
        references=SimpleNamespace(
            receipt_json=canonical_json({"defaults": [], "members": []}),
            selection=selected,
            schema_intake=SimpleNamespace(receipt_json=canonical_json({"selection_path": str(schemas)})),
        )
    )
    declared_budget = dict(vars(BUDGET))
    first, second = review_selection(standard), review_selection(standard)
    for field in ("max_members", "max_probes", "max_total_probe_source_bytes"):
        first["budget"][field] = 1
    first["execution_budget"]["max_members"] = 1
    first["cohorts"]["guard"][0] = 99
    first["canonical_source"]["commit"] = "0" * 40
    assert vars(BUDGET) == declared_budget == second["budget"]
    assert second == review_selection(standard)
    assert json.loads(selected.read_bytes())["execution_budget"]["max_members"] == 150
    assert json.loads(schemas.read_bytes())["canonical_source"]["commit"] == "1" * 40


@pytest.fixture(scope="module")
def checked(observed, tmp_path_factory):
    owner = tmp_path_factory.mktemp("actual-pointwise-semantic-owner")
    review = owner / "review.json"
    F.G.write(review, review_selection(observed))
    return M.prepare(
        standard_ir=observed, review=review, budget=BUDGET, forbidden_roots=(), destination=owner / "cases"
    )


def test_actual_owner_and_realized_stress_keep_all_original_slots_and_unknowns(checked):
    record = checked.record()
    assert record["schema"] == M.POINTWISE_SCHEMA and len(record["members"]) == 72
    assert sum(row["state"] == "source_case_checked" for row in record["members"]) == 60
    assert len(record["original_calls"]) == 24
    complete = [row for row in record["original_calls"] if row["state"] == "finite_source_stress_checked"]
    assert len(complete) == 20, [
        (r["original_call"], r["unrealized_complete_cohort_stress"]) for r in record["original_calls"]
    ]
    probes = checked.pointwise_probes.record()
    assert (
        len(probes["families"]) == 24
        and sum(row["state"] == "finite_probes_checked" for row in probes["families"]) == 20
    )
    assert sum(len(family["members"]) for family in probes["families"]) == 31
    assert I.require_environment(Path(probes["invocation"]["path"]), environment=M.R.D.ENVIRONMENT)["returncode"] == 0
    for family in complete:
        assert not family["unrealized_complete_cohort_stress"]
        for row in family["supplementary_stress"]["members"]:
            assert row["state"] == "probe_checked" and row["stress"]["schema"] == PS.SCHEMA
            comparison = json.loads(Path(row["products"]["comparison"]["path"]).read_bytes())
            assert comparison["passed"] is True and comparison["checked_elements"] >= 1
    assert record["remaining_by_phase"] == M._UNKNOWN
    assert all(row["remaining_by_phase"] == M._UNKNOWN for row in record["members"])
    assert all(row["owner"] is None for row in record["members"] if row["state"] != "source_case_checked")


def test_reviewed_original_facets_reach_requirements_without_erasing_any_blocker(checked):
    document, arguments = K._inputs(checked.standard_ir)
    original = copy.deepcopy(document)
    ledger = J.join(document, **arguments)
    semantic = SP._semantic_facets(ledger, checked)
    assert semantic["schema"] == M.POINTWISE_SCHEMA
    assert len(ledger["original_source_reference_witnesses"]) == 72
    assert ledger["mandatory_source_blockers"] == original["mandatory_source_blockers"]
    assert ledger["original_required_ids"] == original["original_required_ids"]
    assert len(ledger["requirements"]) == 24
    assert (
        sum(
            row["original_semantic_facets"]["reviewed_finite_original_owner"] == "checked"
            for row in ledger["requirements"]
        )
        == 20
    )
    assert (
        sum(row["original_semantic_facets"]["complete_cohort_stress"] == "checked" for row in ledger["requirements"])
        == 20
    )
    assert all(
        row["source_input_state"] == "unavailable" and row["candidate_verdict"] == "not_evaluated"
        for row in ledger["requirements"]
    )
    assert ledger["release_authority"] == "not_issued"


@pytest.mark.parametrize("defect", ["policy", "palette", "context", "defaults", "argument", "result_dtype"])
def test_review_cannot_change_original_source_policy_or_correspondence(observed, tmp_path, defect):
    selected = review_selection(observed)
    owner = selected["owners"][0]
    if defect == "policy":
        owner["numerical_policy"]["atol"] = 0.1
    elif defect == "palette":
        owner["input_palettes"][0]["values"] = [0.0]
    elif defect == "context":
        owner["implementation_context"][0]["sha256"] = "0" * 64
    elif defect == "defaults":
        owner["selector"]["defaults"].append({"name": "invented"})
    elif defect == "argument":
        owner["selector"]["arguments"][0]["tensor"]["rank"] = 8
    else:
        owner["selector"]["ordered_result_dtypes"] = ["int8"]
    path = tmp_path / "review.json"
    F.G.write(path, selected)
    with pytest.raises(ValueError):
        M.prepare(standard_ir=observed, review=path, budget=BUDGET, forbidden_roots=(), destination=tmp_path / "cases")


@pytest.mark.parametrize(
    "limit", ["max_probes", "max_total_probe_source_bytes", "max_members", "max_total_tensor_payload_bytes"]
)
def test_complete_probe_preflight_denial_preserves_every_original_family_before_stimulus(
    observed, tmp_path, monkeypatch, limit
):
    selected = review_selection(observed)
    selected["budget" if limit in selected["budget"] else "execution_budget"][limit] = 1
    path = tmp_path / "review.json"
    F.G.write(path, selected)
    monkeypatch.setattr(
        B, "_stimulus", lambda *_args: pytest.fail("denied supplementary roster allocated shaped values")
    )
    result = B.prepare(
        standard_ir=observed, review=path, document=selected, forbidden=(), destination=tmp_path / "probes"
    )
    record = result.record()
    assert len(record["families"]) == 24
    assert all(family["state"] == "unavailable" for family in record["families"])
    assert record["invocation"] is None


def test_unselected_owner_leaves_exact_original_slots_and_stress_family_missing(observed, tmp_path):
    selected = review_selection(observed)
    removed = selected["owners"].pop()["id"]
    path = tmp_path / "review.json"
    F.G.write(path, selected)
    result = M.prepare(
        standard_ir=observed, review=path, budget=BUDGET, forbidden_roots=(), destination=tmp_path / "cases"
    )
    record = result.record()
    assert len(record["members"]) == 72
    assert sum(row["state"] == "source_case_checked" for row in record["members"]) == 57
    assert sum(row["state"] == "finite_source_stress_checked" for row in record["original_calls"]) == 19
    assert all(row["owner"] != removed for row in record["members"])


def test_real_full_output_mutation_and_saved_owner_refuse(checked):
    probes = checked.pointwise_probes
    with pytest.raises(ValueError, match="actual live"):
        replace(probes).verify()
    record = probes.record()
    row = next(row for family in record["families"] for row in family["members"] if row["state"] == "probe_checked")
    path = Path(row["products"]["actual"]["path"])
    before = path.read_bytes()
    try:
        actual = json.loads(before)
        encoded = actual["outputs"][0]["data_hex"]
        actual["outputs"][0]["data_hex"] = encoded[:-1] + ("0" if encoded[-1] != "0" else "1")
        path.write_bytes(canonical_json(actual))
        with pytest.raises(ValueError):
            checked.verify()
    finally:
        path.write_bytes(before)


@pytest.mark.parametrize(
    "defect",
    ["legacy_version", "missing_partition", "saved_state", "bool_extent", "missing_private_cohort", "duplicates"],
)
def test_pointwise_review_version_and_obligations_are_closed(observed, defect):
    selected = review_selection(observed)
    if defect == "legacy_version":
        selected["schema"] = Q.SCHEMA
    elif defect == "missing_partition":
        selected["owners"][0]["stress"]["across_complete_cohorts"].pop()
    elif defect == "saved_state":
        selected["owners"][0]["accepted"] = True
    elif defect == "bool_extent":
        selected["owners"][0]["stress_probes"]["extent"] = True
    elif defect == "missing_private_cohort":
        del selected["cohorts"]["withheld_transfer"]
    else:
        selected["owners"].append(copy.deepcopy(selected["owners"][0]))
    with pytest.raises((ValueError, TypeError)):
        Q.validate(selected)
