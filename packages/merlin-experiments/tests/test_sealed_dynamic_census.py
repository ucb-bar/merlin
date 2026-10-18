"""The sealed issuer's census of recipe-rewritten (dynamic) integer contractions."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from merlin_experiments.capture_execution import sealed_m2m
from merlin_experiments.capture_execution.sealed_static import _file_digest

from merlin.targetgen.quant_recipe import digest

_BAR = {"schema": "dynamic_site_agreement_v1", "cosine_min": 0.99, "rel_rmse_max": 0.1}
_SDPA = ["g:original:root:n10"]
_OTHER = ["g:original:root:n20"]


def _recipe(bar=_BAR) -> dict:
    dynamic = {"schema": "dynamic_contractions_v1", "families": ["activation_contraction"], "scales": "static_heads"}
    if bar is not None:
        dynamic["site_agreement"] = dict(bar)
    body = {"schema": "quant_recipe_v1", "status": "derived", "dynamic": dynamic}
    body["recipe_sha256"] = digest(body)
    return body


def _site(node: str, calls: int, lineage: list[str], **verification) -> dict:
    check = {
        "schema": "m2m.dynamic_site_verification.v1",
        "samples": 3,
        "finite": True,
        "reference": "float64_matmul_of_site_operands_rounded_f32",
        "cosine_min": 0.999,
        "cosine_mean": 0.9995,
        "max_abs_error": 0.01,
        "rel_rmse_max": 0.02,
    }
    check.update(verification)
    return {
        "node": node,
        "family": "activation_contraction",
        "status": "rewritten",
        "device_calls": calls,
        "shapes": [[1, 2, 4, 8], [1, 2, 8, 4]],
        "lineage": list(lineage),
        "verification": check,
    }


def _stage(tmp_path: Path, *, weight=2, dynamic=((_SDPA, 3), (_OTHER, 2)), extra=None) -> tuple[Path, dict]:
    """A stage with its program, a bound trace of every int8 generic, and matching metadata."""
    stage = tmp_path / "stage"
    stage.mkdir(exist_ok=True)
    (stage / "model.mlir").write_text("module {}")
    nodes, operations = [], []

    def int_mm(transformation: str, origin: list[str]) -> None:
        node_id = f"g:prepared:root:n{len(nodes)}"
        nodes.append(
            {
                "id": node_id,
                "target": "aten._int_mm.default",
                "transformation": transformation,
                "origin_node_ids": origin,
            }
        )
        operations.append(
            {
                "operation": "linalg.generic",
                "operand_types": ["tensor<4x8xi8>", "tensor<8x4xi8>", "tensor<4x4xi32>"],
                "result_types": ["tensor<4x4xi32>"],
                "source_node_ids": [node_id],
            }
        )

    for index in range(weight):
        int_mm("pt2e_integerization", [f"g:original:root:n{100 + index}", "g:quantized:root:n1"])
    for lineage, calls in dynamic:
        for _ in range(calls):
            int_mm("static_attention", [*lineage, "g:quantized:root:n2"])
    operations.append(
        {
            "operation": "linalg.generic",
            "operand_types": ["tensor<4xf32>"],
            "result_types": ["tensor<4xf32>"],
            "source_node_ids": [],
        }
    )
    if extra:
        extra(nodes, operations)
    trace = {
        "graphs": {"prepared": {"nodes": nodes}},
        "mlir": {"sha256": _file_digest(stage / "model.mlir"), "operations": operations},
    }
    (stage / "frontend-trace.json").write_text(json.dumps(trace))
    sites = [_site("matmul_qk", 1, _SDPA), _site("matmul_pv", 2, _SDPA), _site("matmul_x", 2, _OTHER)]
    recipe = _recipe()
    meta = {
        "recipe": recipe,
        "recipe_sha256": recipe["recipe_sha256"],
        "recipe_agreement": {"samples": 3},
        "quantization_stats": {
            "dynamic_contractions": {"rewritten": 3, "kept_in_float": [], "device_calls": 5, "sites": sites}
        },
    }
    return stage, meta


def _census(stage: Path, meta: dict, emitted: int = 2, exported: int = 7) -> int:
    return sealed_m2m._dynamic_integer_census(meta, meta["quantization_stats"], stage, emitted, exported)


def test_exact_dynamic_accounting_is_admitted(tmp_path: Path):
    stage, meta = _stage(tmp_path)
    assert _census(stage, meta) == 5


def test_capture_without_dynamic_contractions_keeps_the_historical_rule(tmp_path: Path):
    stage, meta = _stage(tmp_path)
    meta["quantization_stats"].pop("dynamic_contractions")
    assert _census(stage, meta) == 0  # emitted == exported is then still required by the caller
    (stage / "frontend-trace.json").unlink()
    assert _census(stage, meta) == 0  # and no trace is read


def _site_edit(index: int, **changes):
    def edit(meta):
        meta["quantization_stats"]["dynamic_contractions"]["sites"][index]["verification"].update(changes)

    return edit


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda m: m["quantization_stats"]["dynamic_contractions"]["sites"][0].pop("verification"),
            "finite verification",
        ),
        (_site_edit(0, samples=2), "finite verification"),
        (_site_edit(1, finite=False), "finite verification"),
        (_site_edit(1, cosine_min=float("nan")), "finite verification"),
        (_site_edit(2, rel_rmse_max=None), "finite verification"),
        (_site_edit(0, schema="other"), "finite verification"),
        (_site_edit(0, cosine_min=0.98), "declared agreement bar"),
        (_site_edit(2, rel_rmse_max=0.2), "declared agreement bar"),
        (
            lambda m: m.update(recipe=_recipe(bar=None), recipe_sha256=_recipe(bar=None)["recipe_sha256"]),
            "no per-site agreement bar",
        ),
        (lambda m: m["recipe"]["dynamic"]["site_agreement"].update(cosine_min=0.5), "does not match its digest"),
        (
            lambda m: (
                m.update(recipe=_recipe(bar={**_BAR, "cosine_min": 2.0}))
                or m.update(recipe_sha256=m["recipe"]["recipe_sha256"])
            ),
            "site_agreement",
        ),
        (lambda m: m["quantization_stats"]["dynamic_contractions"].update(device_calls=6), "sum to the census total"),
        (lambda m: m["quantization_stats"]["dynamic_contractions"].update(kept_in_float=[{"node": "x"}]), "none kept"),
        (lambda m: m["quantization_stats"]["dynamic_contractions"].update(rewritten=2), "none kept"),
        (
            lambda m: m["quantization_stats"]["dynamic_contractions"]["sites"][0].update(status="kept"),
            "rewritten contraction",
        ),
        (
            lambda m: m["quantization_stats"]["dynamic_contractions"]["sites"][0].update(family="contraction"),
            "rewritten contraction",
        ),
        (
            lambda m: m["quantization_stats"]["dynamic_contractions"]["sites"][0].update(lineage=[]),
            "no original lineage",
        ),
        (
            lambda m: m["quantization_stats"]["dynamic_contractions"]["sites"][0].update(
                shapes=[[1, 1 << 20], [1 << 20, 1]]
            ),
            "reduction bound",
        ),
        (lambda m: m.update(recipe_agreement={"samples": 0}), "agreement records no samples"),
    ],
)
def test_each_unproved_dynamic_site_is_refused(tmp_path: Path, edit, message):
    stage, meta = _stage(tmp_path)
    bad = copy.deepcopy(meta)
    edit(bad)
    with pytest.raises(sealed_m2m.SealedM2MError, match=message):
        _census(stage, bad)


def test_counts_must_add_up_and_every_contraction_has_one_owner(tmp_path: Path):
    stage, meta = _stage(tmp_path)
    with pytest.raises(sealed_m2m.SealedM2MError, match="emitted plus dynamic"):
        _census(stage, meta, exported=8)
    with pytest.raises(sealed_m2m.SealedM2MError, match="emitted plus dynamic"):
        _census(stage, meta, emitted=3)
    # A device call missing from the trace (a gap) and one owned by another lineage (a misattribution).
    for dynamic in (((_SDPA, 3), (_OTHER, 1)), ((_SDPA, 4), (_OTHER, 1)), ((_SDPA, 2), (_OTHER, 3))):
        stage, meta = _stage(tmp_path, dynamic=dynamic)
        with pytest.raises(sealed_m2m.SealedM2MError, match="differ from their owners"):
            _census(stage, meta)
    stage, meta = _stage(tmp_path, weight=3)
    with pytest.raises(sealed_m2m.SealedM2MError, match="differ from their owners"):
        _census(stage, meta)


def _twice(nodes, operations):
    operations.append(copy.deepcopy(operations[0]))  # one traced node emitted twice (an overlap)


def _unemitted(nodes, operations):
    nodes.append({"id": "g:prepared:root:n99", "target": "aten._int_mm.default", "transformation": "static_attention"})


def _two_owners(nodes, operations):
    operations[0]["source_node_ids"].append(operations[1]["source_node_ids"][0])


def _unknown(nodes, operations):
    nodes[0]["transformation"] = "unknown"


def _no_lineage(nodes, operations):
    nodes[-1]["origin_node_ids"] = ["g:quantized:root:n2"]


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (_twice, "exactly once"),
        (_unemitted, "exactly once"),
        (_two_owners, "unique traced owner"),
        (_unknown, "unknown traced transformation"),
        (_no_lineage, "no original lineage"),
    ],
)
def test_traced_ownership_overlap_gap_and_identity_are_refused(tmp_path: Path, extra, message):
    stage, meta = _stage(tmp_path, extra=extra)
    with pytest.raises(sealed_m2m.SealedM2MError, match=message):
        _census(stage, meta)


def test_trace_must_be_bound_to_the_stage_program(tmp_path: Path):
    stage, meta = _stage(tmp_path)
    (stage / "model.mlir").write_text("module { changed }")
    with pytest.raises(sealed_m2m.SealedM2MError, match="not bound to the stage program"):
        _census(stage, meta)
    (stage / "frontend-trace.json").unlink()
    with pytest.raises(sealed_m2m.SealedM2MError, match="lacks the stage's frontend trace"):
        _census(stage, meta)


def _integer_meta(meta: dict, *, exported: int, emitted: int = 2) -> dict:
    """A single-program int8 capture's metadata: two weight contractions plus ``meta``'s dynamic sites."""
    agreement = {
        "status": "passed",
        "reference": "pt2e_integer",
        "atol": 0.0,
        "rtol": 0.0,
        "executed_contractions": {"conv2d": 0, "linear": 2, "matmul": 0, "total": 2, "selected": 2, "observed": 2},
        "samples": 1,
        "finite": True,
        "max_abs": 0.0,
        "max_rel": 0.0,
        "outputs": [
            {"finite": True, "within_tolerance": True, "max_abs": 0.0, "max_rel": 0.0, "atol": 0.0, "rtol": 0.0}
        ],
    }
    out = copy.deepcopy(meta)
    out["recipe"]["software_numerical_engine"] = "integer_reference"
    out["recipe"]["recipe_sha256"] = digest(out["recipe"])
    out["recipe_sha256"] = out["recipe"]["recipe_sha256"]
    out["quantization_stats"]["annotated_contractions"] = 2
    out["integerization_receipt"] = {
        "schema": "m2m.pt2e-integerize.v1",
        "golden_agreement": agreement,
        "exported_integer_mm_count": exported,
        "integer_mm_emitted": emitted,
        "quantized_contractions_seen": 2,
        "quantized_contractions_integerized": 2,
        "quantized_contractions_remaining": 0,
        "quantized_by_kind": {
            "linear": {"seen": 2, "integerized": 2, "remaining": 0},
            **{kind: {"seen": 0, "integerized": 0, "remaining": 0} for kind in ("conv2d", "matmul", "unsupported")},
        },
        "refusals": [],
        "precision_decisions": [{"kind": "linear", "node": n, "decision": "integerized_i32"} for n in ("a", "b")],
        "precision_decision_counts": {"integerized_i32": 2, "preserve_float_qdq": 0, "unresolved": 0},
        "accumulator_bound_checked": True,
        "max_reduction_k": 4,
    }
    return out


def test_single_program_partition_counts_verified_dynamic_contractions(tmp_path: Path, monkeypatch):
    stage, meta = _stage(tmp_path)
    monkeypatch.setattr(
        sealed_m2m, "_materialized_v2", lambda *_args: {"status": "verified_materialized", "receipt_sha256": "a" * 64}
    )
    plan = {"dtype": "int8", "recipe": {"recipe_sha256": "unused"}}
    (stage / "meta.json").write_text(json.dumps(_integer_meta(meta, exported=7)))
    assert sealed_m2m._materialized_v3(stage, tmp_path, stage, plan)["integer_contractions"] == 7
    # The historical census, unchanged: without the dynamic block the extra contractions are unexplained.
    plain = _integer_meta(meta, exported=7)
    plain["quantization_stats"].pop("dynamic_contractions")
    (stage / "meta.json").write_text(json.dumps(plain))
    with pytest.raises(sealed_m2m.SealedM2MError, match="independently verified integer contractions"):
        sealed_m2m._materialized_v3(stage, tmp_path, stage, plan)
    plain["integerization_receipt"]["exported_integer_mm_count"] = 2
    (stage / "meta.json").write_text(json.dumps(plain))
    assert sealed_m2m._materialized_v3(stage, tmp_path, stage, plan)["integer_contractions"] == 2
    low = _integer_meta(meta, exported=7)
    low["quantization_stats"]["dynamic_contractions"]["sites"][1]["verification"]["cosine_min"] = 0.5
    (stage / "meta.json").write_text(json.dumps(low))
    with pytest.raises(sealed_m2m.SealedM2MError, match="declared agreement bar"):
        sealed_m2m._materialized_v3(stage, tmp_path, stage, plan)
