"""Synthetic owner controls for negative-only original result-format propagation."""

import contextvars
import json
import sys
import threading
from hashlib import sha256
from types import ModuleType

import pytest
import yaml
from merlin_experiments.phase1.feedback import private_result_format_selection as S
from merlin_experiments.phase1.feedback.private_source_member_hardware import observe
from merlin_experiments.phase1.options import parse_options

from merlin.common import mlir_query as mq
from merlin.targetgen import eligibility as E
from merlin.targetgen import operation_accounting as OA
from merlin.targetgen.compute_units import SemanticCapability
from merlin.xdsl_dialects.lowering import compute_groups as CG

CONTRACT, FACTS = b'{"target":"public"}', b'{"facts":"synthetic"}'


@pytest.fixture
def owner(monkeypatch):
    """Substitute evidence issuance only; exercise actual E/H/OA consumers."""
    caps = {"elementwise_map": SemanticCapability("elementwise_map", dtypes=("int8",))}

    class Proof:
        stale = False
        reopens = 0

        def record(self):
            return {"target": "public", "contract_sha256": sha256(CONTRACT).hexdigest(),
                    "facts_sha256": sha256(FACTS).hexdigest(), "selection_sha256": "a" * 64}

        def refusal(self, family, output, selected):
            if selected != caps:
                raise ValueError("foreign capability map")
            return {"negative_source_proof": True} if (
                family == "elementwise_map" and output is not None and output not in {"int8", "int32"}
            ) else None

        def revalidate_sources(self):
            self.reopens += 1
            if self.stale:
                raise ValueError("original source changed")

    module = ModuleType("merlin.targetgen.rtl.result_format_ceiling")
    module.NativeResultFormatCeiling = Proof
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(E, "capability_map_from_contract", lambda contract: caps)
    return Proof(), caps


def scope(proof, **changes):
    return E.original_result_format_scope(proof, **{
        "target": "public", "original_contract_bytes": CONTRACT, "original_facts_bytes": FACTS, **changes,
    })


def conversion(output="bf16"):
    return E.RegionDescriptor(op="dequantize", family="elementwise_map", in_dtype="int8", out_dtype=output)


def test_default_missing_result_declaration_stays_unknown(owner):
    _, caps = owner
    verdict = E.is_eligible(conversion(), caps)
    assert verdict.undetermined and verdict.refusal == "result_dtype_unknown"


def test_outside_domain_refuses_but_membership_does_not_admit_conversion(owner):
    proof, caps = owner
    with scope(proof):
        outside = E.is_eligible(conversion(), caps)
        inside = E.is_eligible(conversion("int32"), caps)
        missing = E.is_eligible(
            E.RegionDescriptor(op="cast", family="elementwise_map", captured_input_formats=("int8", None)), caps
        )
    assert not outside.eligible and not outside.undetermined and outside.refusal == "result_dtype"
    assert inside.undetermined and inside.refusal == "result_dtype_unknown"
    assert missing.undetermined  # no actual source/output evidence can be supplied by the ceiling
    assert proof.reopens == 1
    assert E.is_eligible(conversion(), caps).undetermined


def test_actual_typed_group_member_and_oa_row_share_the_same_proof(owner):
    proof, caps = owner
    module = mq.parse('''builtin.module {
      func.func @public(%x: tensor<2x3xi8>, %s: tensor<f32>, %z: tensor<i64>) -> tensor<2x3xbf16> {
        %r = "quant_ext.dequantize_per_tensor"(%x, %s, %z)
          : (tensor<2x3xi8>, tensor<f32>, tensor<i64>) -> tensor<2x3xbf16>
        func.return %r : tensor<2x3xbf16>
      }
    }''')
    member = next(op for op in module.walk() if mq.op_name(op) == "quant_ext.dequantize_per_tensor")
    row = {"semantic_family": "elementwise_map", "disposition": "host_required",
           "shape_confidence": "result_type", "ordered_operand_types": [{"dtype": "i8", "shape": [2, 3]}],
           "ordered_result_types": [{"dtype": "bf16", "shape": [2, 3]}], "operand_format": "int8",
           "result_dtypes": ["bf16"], "mlir_operation": "quant_ext.dequantize_per_tensor"}
    with scope(proof):
        source = observe(member, CG.DEQUANTIZE, {})
        accounting = OA._hardware_admission(row, OA._observed_signature(row), {}, caps)
    assert (source["status"], source["refusal"]) == ("unsupported", "result_dtype")
    assert (accounting["status"], accounting["refusal"]) == ("unsupported", "result_dtype")
    assert observe(member, CG.DEQUANTIZE, {})["status"] == "unknown"


@pytest.mark.parametrize("changes", [{"target": "foreign"}, {"original_contract_bytes": b"foreign"},
                                     {"original_facts_bytes": b"foreign"}])
def test_original_target_and_byte_bindings_cannot_be_replaced(owner, changes):
    with pytest.raises(ValueError, match="original target/contract/facts"), scope(owner[0], **changes):
        pass


def test_foreign_map_expired_context_and_foreign_thread_refuse(owner):
    proof, caps = owner
    with scope(proof):
        with pytest.raises(ValueError, match="foreign capability"):
            E.is_eligible(conversion(), {**caps, "movement": SemanticCapability("movement", dtypes=("int8",))})
        copied = contextvars.copy_context()
        errors = []

        def foreign_thread():
            try:
                copied.run(E.is_eligible, conversion(), caps)
            except ValueError as error:
                errors.append(str(error))

        thread = threading.Thread(target=foreign_thread)
        thread.start()
        thread.join()
        assert len(errors) == 1 and "another process or thread" in errors[0]
    with pytest.raises(ValueError, match="expired"):
        copied.run(E.is_eligible, conversion(), caps)


def test_changed_source_refuses_before_publication_and_scope_is_restored(owner):
    proof, caps = owner
    with pytest.raises(ValueError, match="original source changed"), scope(proof):
        proof.stale = True
    assert E.is_eligible(conversion(), caps).undetermined


def test_explicit_selection_freezes_reopens_and_preserves_default(tmp_path):
    assert S.from_run(tmp_path, target="public") is None
    source = tmp_path / "operator.json"
    source.write_text(json.dumps({"original": "synthetic"}))
    run = tmp_path / "run"
    selected, record = S.freeze_for_run(source, target="public", run_dir=run)
    (run / "environment.yaml").write_text(yaml.safe_dump({"native_result_format_selection": record}))
    assert S.from_run(run, target="public") == selected
    assert S.freeze_for_run(source, target="public", run_dir=run, prior=record, resuming=True)[1] == record
    with pytest.raises(ValueError, match="removed"):
        S.freeze_for_run("", target="public", run_dir=run, prior=record, resuming=True)
    with pytest.raises(ValueError, match="run owner"):
        S.from_run(run, target="foreign")
    source.write_text(json.dumps({"original": "changed"}))
    with pytest.raises(ValueError, match="changed"):
        selected.verify()


def test_selection_is_never_an_environment_or_candidate_budget_knob():
    options = parse_options(["--run-id", "public"], environ={"MERLIN_NATIVE_RESULT_FORMAT_SELECTION": "candidate"})
    assert options.native_result_format_selection == ""
    explicit = parse_options(["--run-id", "public", "--native-result-format-selection", "original.json"], environ={})
    assert explicit.native_result_format_selection == "original.json"


def test_forged_run_selection_owner_refuses(tmp_path):
    forged = S.FrozenSelection(tmp_path / "fake", tmp_path / "fake", "a" * 64, "public", object())
    with pytest.raises(ValueError, match="original run owner"):
        forged.verify()
