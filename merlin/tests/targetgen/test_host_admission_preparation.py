"""Prepared lookup preserves every ordinary admission and source-body check."""

from copy import deepcopy

import pytest

from merlin.targetgen import host_capabilities as H
from merlin.targetgen.host_admission_preparation import PreparedHostCapabilities


def selection(operations):
    return {
        "host": {
            "package_sha256": "a" * 64,
            "capability_spec_sha256": "b" * 64,
            "dtype_strategy": "typed",
            "capability_spec": {
                "schema": H.SCHEMA,
                "status": "reviewed",
                "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "typed"},
                "operations": operations,
                "evidence": {"scope": "synthetic declaration"},
            },
        }
    }


def declaration(name, *, ops=None, status="reviewed", dtype="f32"):
    row = {
        "id": name,
        "placement": "host",
        "families": ["elementwise_map"],
        "status": status,
        "signature": {"operand_dtypes": [dtype]},
    }
    if ops is not None:
        row["ops"] = ops
    return row


@pytest.mark.parametrize("order", [(0, 1, 2, 3), (3, 2, 1, 0), (2, 0, 3, 1)])
@pytest.mark.parametrize("dtype", ["f32", "i64", None])
def test_exact_frontend_carrier_family_order_and_unknown_fallback_are_identical(order, dtype):
    declarations = [
        declaration("family"),
        declaration("carrier", ops=["linalg.generic"]),
        declaration("frontend", ops=["aten.test", "linalg.generic"]),
        declaration("draft", ops=["aten.test"], status="unreviewed"),
    ]
    selected = selection([declarations[index] for index in order])
    selected["absent"] = {"capability_spec": None}
    prepared = PreparedHostCapabilities(selected)
    for row in (
        {"mlir_operation": "linalg.generic", "frontend_op": "aten.test"},
        {"mlir_operation": "other"},
        {"mlir_operation": "linalg.generic"},
    ):
        observed = {"family": "elementwise_map", "operand_dtype": dtype}
        assert H.admit_host_operation(prepared, row, observed) == H.admit_host_operation(selected, row, observed)
    prepared.require_current(selected)


def test_profile_validated_once_and_never_retains_an_admission_result(monkeypatch):
    selected = selection([declaration("carrier", ops=["linalg.generic"])])
    calls = []
    original = H.validate_host_capabilities

    def validate(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(H, "validate_host_capabilities", validate)
    prepared = PreparedHostCapabilities(selected)
    assert len(calls) == 1
    for dtype in ("f32", "i64", "f32"):
        result = H.admit_host_operation(
            prepared, {"mlir_operation": "linalg.generic"}, {"family": "elementwise_map", "operand_dtype": dtype}
        )
        assert result["status"] == ("admitted" if dtype == "f32" else "unsupported")
    assert len(calls) == 1
    H.admit_host_operation(selected, {"mlir_operation": "linalg.generic"}, {"family": "elementwise_map"})
    assert len(calls) == 2


def test_prepared_input_is_private_and_changed_profile_cannot_finish():
    selected = selection([declaration("family")])
    prepared = PreparedHostCapabilities(selected)
    escaped = prepared["host"]
    escaped["capability_spec"]["operations"].clear()
    row, signature = {"mlir_operation": "other"}, {"family": "elementwise_map", "operand_dtype": "f32"}
    assert H.admit_host_operation(prepared, row, signature)["status"] == "admitted"
    selected["host"]["capability_spec"]["operations"][0]["signature"]["operand_dtypes"] = ["i64"]
    with pytest.raises(ValueError, match="changed"):
        prepared.require_current(selected)
    assert H.admit_host_operation(PreparedHostCapabilities(selected), row, signature)["status"] == "unsupported"


def test_source_body_is_rechecked_for_every_supplied_occurrence():
    from merlin.common import mlir_query as mq

    def source(scalar):
        module = mq.parse(f"""module {{ func.func @f(%x:tensor<2xf32>, %y:tensor<2xf32>,
          %init:tensor<2xf32>) -> tensor<2xf32> {{
          %r = linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,
            affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
            ins(%x,%y:tensor<2xf32>,tensor<2xf32>) outs(%init:tensor<2xf32>) {{
              ^bb0(%a:f32,%b:f32,%old:f32):
              %v = arith.{scalar} %a,%b : f32
              linalg.yield %v : f32
            }} -> tensor<2xf32>
          func.return %r : tensor<2xf32> }} }}""")
        return next(mq.walk(module, "linalg.generic"))

    declared = declaration("typed_add", ops=["linalg.generic"])
    declared["signature"] = {"ordered_operand_dtypes": ["f32"] * 3, "ordered_result_dtypes": ["f32"], "ranks": [1]}
    declared["source_body"] = {"schema": "merlin.static_pointwise_source_body.v1", "operation": "arith.addf"}
    selected = selection([declared])
    prepared = PreparedHostCapabilities(selected)
    row = {"mlir_operation": "linalg.generic", "count": 1}
    observed = {
        "family": "elementwise_map",
        "ordered_operand_dtypes": ["f32"] * 3,
        "ordered_result_dtypes": ["f32"],
        "rank": 1,
    }
    good, bad = source("addf"), source("subf")
    for occurrences in ((good,), (bad,), None, (good, good)):
        assert H.admit_host_operation(prepared, row, observed, source_operations=occurrences) == (
            H.admit_host_operation(selected, row, observed, source_operations=occurrences)
        )
    assert H.admit_host_operation(prepared, row, observed, source_operations=(bad,))["status"] == "unsupported"


def test_program_wrapper_rejects_mutation_after_success_and_is_not_global():
    from merlin_experiments.phase1.feedback.private_source_admission import prepared_host_admission

    selected = selection([declaration("family")])
    seen = []

    @prepared_host_admission
    def run(capture, target, software, capability, host):
        seen.append(host)
        return H.admit_host_operation(
            host, {"mlir_operation": "other"}, {"family": "elementwise_map", "operand_dtype": "f32"}
        )

    assert run(None, None, None, None, selected)["status"] == "admitted"
    assert run(None, None, None, None, selected)["status"] == "admitted"
    assert seen[0] is not seen[1]

    @prepared_host_admission
    def changed(*args):
        selected["host"]["capability_spec"]["status"] = "unreviewed"
        return {"passed": True}

    with pytest.raises(ValueError, match="changed"):
        changed(None, None, None, None, selected)


def test_invalid_changed_profile_is_not_hidden_by_previous_validation():
    selected = selection([declaration("family")])
    PreparedHostCapabilities(selected)
    altered = deepcopy(selected)
    altered["host"]["capability_spec"]["operations"][0]["status"] = "accepted"
    with pytest.raises(ValueError, match="review status"):
        PreparedHostCapabilities(altered)


def test_foreign_prepared_subclass_cannot_select_unvalidated_documents():
    class Foreign(PreparedHostCapabilities):
        pass

    with pytest.raises(ValueError, match="profile mapping"):
        H.admit_host_operation(
            Foreign(selection([declaration("family")])), {"mlir_operation": "other"}, {"family": "elementwise_map"}
        )


def test_unissued_exact_host_carrier_refuses_before_admission():
    forged = object.__new__(PreparedHostCapabilities)
    with pytest.raises(ValueError, match="original issued"):
        H.admit_host_operation(forged, {"mlir_operation": "other"}, {"family": "elementwise_map"})


def test_preparations_are_released_with_their_program_carriers():
    import gc
    import weakref

    from merlin.targetgen import host_admission_preparation as P

    prepared = PreparedHostCapabilities(selection([declaration("family")]))
    identity = id(prepared)
    reference = weakref.ref(prepared)
    assert identity in P._ISSUED
    del prepared
    gc.collect()
    assert reference() is None and identity not in P._ISSUED
