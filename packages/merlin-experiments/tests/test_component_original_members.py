"""Original-member transport controls; synthetic owners are wiring only.

These pure fixtures isolate the binding algorithm. They do not issue real
schema, hardware, source preparation, runtime or candidate qualifications.
"""

import base64
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import original_reference_standard_ir as S
from merlin_experiments.phase0.original_call_sources import required_source_cohorts
from merlin_experiments.phase1 import component_original_members as M
from merlin_experiments.phase1 import component_qualification_members as Q
from merlin_experiments.phase2.contracts import StageGateError

from merlin.common.jsonio import canonical_json
from merlin.targetgen.frontend_original_call import _literal
from merlin.targetgen.original_operator_reference import OriginalReferenceBudget, prepare_original_reference
from merlin.targetgen.original_pointwise_reference import OriginalPointwiseReferencePolicy
from merlin.targetgen.original_pointwise_sources import INTEGER_FORM_SCHEMA, pointwise_source
from merlin.targetgen.original_reference_values import TypedReferenceTensor as T


def contract(dtype="int64", *, rank=1, extent=3):
    floating = dtype == "float32"
    policy = OriginalPointwiseReferencePolicy(
        "aten.clamp.default",
        (dtype,),
        (dtype,),
        dtype,
        "finite_f32" if floating else "bounded_exact",
        "not_applicable",
        "elementwise",
        "not_applicable",
        "rne" if floating else "exact_integer",
        True,
        False,
        "not_applicable",
        0.0,
        0.0,
        "preserve" if floating else "ignore",
    )
    bounds = (-0.0, 1.0) if floating else (-(1 << 63), (1 << 53) + 3)
    tensor = {
        "id": "input",
        "kind": "tensor",
        "dtype": dtype,
        "storage_dtype": dtype,
        "rank": rank,
        "layout": "torch.strided",
        "device": "cpu",
    }
    form = {
        "form_schema": INTEGER_FORM_SCHEMA,
        "status": "supported",
        "target": policy.operation,
        "arguments": [
            {
                "name": "self",
                "type": "Tensor",
                "alias": None,
                "value": {"kind": "ssa", "node_id": "input", "value": tensor},
            }
        ]
        + [
            {"name": name, "type": "Optional[number]", "alias": None, "value": _literal(value)}
            for name, value in zip(("min", "max"), bounds, strict=True)
        ],
        "result_arity": 1,
        "schema_returns": [{"type": "Tensor", "alias": None}],
        "result_roster": [{**tensor, "id": "output"}],
        "rank": rank,
        "operand_dtypes": [dtype],
        "result_dtypes": [dtype],
        "parameters": dict(zip(("min", "max"), bounds, strict=True)),
        "source_numerical_semantics": policy.record(),
    }
    source = pointwise_source(form, extent=extent, max_tensor_elements=10000)
    return prepare_original_reference(
        form,
        source,
        extent=extent,
        policy=policy,
        budget=OriginalReferenceBudget(10000, 100000, 100000, 20000),
        output_byteorder="little",
    )


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Substitute only live upstream authority in this algorithm unit control."""
    selected_contract = contract()
    metadata = selected_contract.verify()
    originals, emitted = [], []
    original_root = tmp_path / "originals"
    original_root.mkdir()
    for index, (cohort, extent) in enumerate(required_source_cohorts()):
        source = original_root / f"original-{index}.mlir"
        source.write_text("original typed source bytes\n")
        inputs = (T.from_values("X", "int64", metadata["inputs"][0]["shape"], [(1 << 53) + 3] * 3, byteorder="little"),)
        input_path = original_root / f"inputs-{index}.json"
        input_path.write_bytes(canonical_json([M.R._tensor_record(row) for row in inputs]))
        original = {
            "original_member_id": "original",
            "graph_path": str(tmp_path / "graph.json"),
            "node": "call",
            "target": "aten.clamp.default",
            "cohort": cohort,
            "extent": extent,
            "state": "reference_checked",
            "call": {"arguments": "complete original arguments"},
            "products": {"inputs": M.R._pin(input_path)},
            "required_unknowns": ["physical_effects"],
        }
        identity = {key: original[key] for key in M._IDENTITY}
        originals.append(original)
        emitted.append(
            {
                "original": identity,
                "reference_member_sha256": M._sha(M.R._json(original)),
                "state": "source_reference_ir_checked",
                "required_unknowns": ["physical_effects"],
                "products": {"source": M.R._pin(source)},
                "ordered_abi": {"entry_symbol": "forward", **{key: metadata[key] for key in ("inputs", "outputs")}},
            }
        )
    references = SimpleNamespace(sha256="1" * 64)
    references.record_without_verification = lambda: {"members": originals}
    standard = object.__new__(S.OriginalReferenceStandardIr)
    selection = tmp_path / "selection.json"
    selection.write_text("{}")
    object.__setattr__(standard, "references", references)
    object.__setattr__(standard, "selection", selection)
    object.__setattr__(standard, "source_pins", ())
    object.__setattr__(standard, "receipt_json", b"diagnostic-only")
    document = {"members": emitted, "totals": {"execution": dict.fromkeys(M.E._METRICS, 0)}}
    monkeypatch.setattr(S.OriginalReferenceStandardIr, "record", lambda self: copy.deepcopy(document))
    monkeypatch.setattr(M.P, "required_members", lambda _: {"members": copy.deepcopy(originals)})
    selection_policy = {
        "budget": {"max_members": 100, "max_source_bytes": 20000, "max_total_source_bytes": 100000},
        "execution_budget": {
            "schema": M.E.SCHEMA,
            "max_scalar_bits": 64,
            **{"max_" + key: 1000000 for key in M.E._METRICS},
            **{"max_total_" + key: 10000000 for key in M.E._METRICS},
        },
    }
    monkeypatch.setattr(M.P, "validate", lambda *_: selection_policy)
    monkeypatch.setattr(S, "_contract", lambda *_: selected_contract)
    return standard, document, originals, selection_policy, selected_contract


def test_complete_calls_cohorts_policy_storage_and_abi_are_bound(isolated, tmp_path):
    standard, document, _, _, expected = isolated
    owner = M.prepare(standard_ir=standard, destination=tmp_path / "members")
    record = owner.require_complete()
    assert len(record["members"]) == 3
    assert [row["original"] for row in record["members"]] == [row["original"] for row in document["members"]]
    assert [row["original"]["cohort"] for row in record["members"]] == [row[0] for row in required_source_cohorts()]
    member = owner.member(Path(record["members"][0]["capsule_root"]))
    selected, inputs = member.contract_inputs()
    assert selected.policy.record() == expected.policy.record()
    assert inputs[0].values() == ((1 << 53) + 3,) * 3
    assert record["members"][0]["required_unknowns"] == ["physical_effects"]
    assert member.compare_values({"Y": [(1 << 53) + 3] * 3})["status"] == "pass"
    assert member.compare_values({"Y": [(1 << 53) + 2, (1 << 53) + 3, (1 << 53) + 3]})["status"] == "fail"


@pytest.mark.parametrize("defect", ["source", "envelope", "extra", "symlink", "receipt", "copy", "bool_slot"])
def test_member_owner_reopens_full_products_and_live_identity(isolated, tmp_path, defect):
    owner = M.prepare(standard_ir=isolated[0], destination=tmp_path / "members")
    row = owner.require_complete()["members"][0]
    root = Path(row["capsule_root"])
    if defect == "source":
        (root / "source.mlir").write_text("changed")
    elif defect == "envelope":
        (root / "capsule.yaml").write_text("{}")
    elif defect == "extra":
        (root / "extra.py").write_text("extra")
    elif defect == "symlink":
        (root / "alias").symlink_to(root / "source.mlir")
    elif defect == "receipt":
        (owner.destination / "members.json").write_text("{}")
    elif defect == "copy":
        owner = copy.copy(owner)
    else:
        with pytest.raises(ValueError, match="exact live original member"):
            M.OriginalCandidateMember(owner, True).verify()
        return
    with pytest.raises(ValueError):
        owner.verify()


@pytest.mark.parametrize("defect", ["missing", "reordered", "identity", "reference", "abi"])
def test_original_row_mutations_refuse_without_creating_destination(isolated, tmp_path, defect):
    standard, document, _, _, _ = isolated
    if defect == "missing":
        document["members"].pop()
    elif defect == "reordered":
        document["members"].reverse()
    elif defect == "identity":
        document["members"][0]["original"]["extent"] = True
    elif defect == "reference":
        document["members"][0]["reference_member_sha256"] = "0" * 64
    else:
        document["members"][0]["ordered_abi"]["outputs"][0]["dtype"] = "float32"
    destination = tmp_path / "refused"
    with pytest.raises(ValueError):
        M.prepare(standard_ir=standard, destination=destination)
    assert not destination.exists()


@pytest.mark.parametrize("defect", ["unsupported", "budget"])
def test_unavailable_original_members_remain_denominator(isolated, tmp_path, defect):
    standard, document, _, selected, _ = isolated
    if defect == "unsupported":
        document["members"][1]["state"] = "unavailable"
    else:
        selected["execution_budget"]["max_total_tensor_payload_bytes"] = 1
    owner = M.prepare(standard_ir=standard, destination=tmp_path / "members")
    rows = owner.verify()["members"]
    assert len(rows) == 3 and any(row["state"] == "unavailable" for row in rows)
    with pytest.raises(ValueError, match="denominator"):
        owner.require_complete()


@pytest.mark.parametrize("observed", [{}, {"Y": [1]}, {"Y": [1, 2, 3], "extra": [1]}, {"Y": [1.0, 2.0, 3.0]}])
def test_candidate_outputs_cannot_omit_truncate_or_cast_original_integer_storage(isolated, tmp_path, observed):
    owner = M.prepare(standard_ir=isolated[0], destination=tmp_path / "members")
    with pytest.raises(ValueError):
        M.OriginalCandidateMember(owner, 0).compare_values(observed)


def test_f32_signed_zero_and_complete_shape_remain_original_policy(isolated, tmp_path, monkeypatch):
    selected = contract("float32", rank=0, extent=1)
    standard, document, originals, _, _ = isolated
    metadata = selected.verify()
    monkeypatch.setattr(S, "_contract", lambda *_: selected)
    for original, emitted in zip(originals, document["members"], strict=True):
        inputs = (T.from_values("X", "float32", (), [-0.0], byteorder="little"),)
        Path(original["products"]["inputs"]["path"]).write_bytes(
            canonical_json([M.R._tensor_record(row) for row in inputs])
        )
        original["products"]["inputs"] = M.R._pin(original["products"]["inputs"]["path"])
        emitted["reference_member_sha256"] = M._sha(M.R._json(original))
        emitted["ordered_abi"].update({key: metadata[key] for key in ("inputs", "outputs")})
    owner = M.prepare(standard_ir=standard, destination=tmp_path / "members")
    member = M.OriginalCandidateMember(owner, 0)
    assert member.compare_values({"Y": [-0.0]})["status"] == "pass"
    assert member.compare_values({"Y": [0.0]})["status"] == "fail"
    assert member.compare_values({"Y": [[-0.0]]})["status"] == "pass"


def test_lossless_transport_keeps_wide_integer_words_and_original_row_geometry(isolated, tmp_path):
    from merlin.runtime.out_b64 import OutB64Decoder

    owner = M.prepare(standard_ir=isolated[0], destination=tmp_path / "members")
    member = M.OriginalCandidateMember(owner, 0)
    _, inputs = member.contract_inputs()
    raw = inputs[0].data
    decoder, observed = OutB64Decoder(), {}
    for text in (
        "OUT_B64_BEGIN v1 Y 1 3 8 s",
        f"OUT_B64_CHUNK 00000000 {len(raw):04x} " + base64.b64encode(raw).decode(),
        "OUT_B64_END",
    ):
        assert decoder.consume(text.split(), observed)
    decoder.require_closed()
    assert observed == {"Y": [[(1 << 53) + 3] * 3]}
    assert member.compare_values(observed)["status"] == "pass"
    with pytest.raises(ValueError, match="geometry"):
        member.compare_values({"Y": [[(1 << 53) + 3]] * 3})


def test_normal_consumer_requires_same_original_witnesses_and_preserves_original_ids(isolated, tmp_path, monkeypatch):
    owner = M.prepare(standard_ir=isolated[0], destination=tmp_path / "members")
    rows = owner.require_complete()["members"]
    record = {
        "coverage_sha256": "coverage",
        "mandatory_source_blockers": [],
        "requirements": [
            {"original_id": "required", "original_reference_facets": {"required_source_slots": [0, 1, 2]}}
        ],
        "original_source_reference_witnesses": [
            {
                "source_slot": row["source_slot"],
                "original": row["original"],
                "reference_member_sha256": row["reference_member_sha256"],
                "standard_ir_products": {"source": row["source"]},
            }
            for row in rows
        ],
    }
    preparation = SimpleNamespace(
        semantic_cases=SimpleNamespace(standard_ir=isolated[0]),
        receipt_json=canonical_json(record),
        require_complete=lambda: record,
    )
    report = {
        "sha256": "coverage",
        "obligations": [{"id": "required", "mandatory": True, "cohort": "functional_guard", "members": []}],
    }
    # Only this algorithm control substitutes the unissued preparation type.
    from merlin_experiments.phase1 import component_qualification_domain as D

    monkeypatch.setattr(D, "_source_record", lambda _: record)
    selected = Q.obligations(report, preparation=preparation, original_members=owner)
    assert selected[0] == report["obligations"][0] and len(selected) == 4
    assert all(row["members"][0]["original_requirement_ids"] == ["required"] for row in selected[1:])
    record["original_source_reference_witnesses"][0]["source_slot"] = False
    with pytest.raises(StageGateError, match="membership"):
        Q.obligations(report, preparation=preparation, original_members=owner)


def test_copy_destination_cannot_enter_original_source_or_reference_owner(isolated, tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        M.prepare(standard_ir=isolated[0], destination=tmp_path / "originals" / "candidate")
    assert not (tmp_path / "originals" / "candidate").exists()


def test_ordered_original_binding_preloads_exact_wide_storage_without_legacy_golden(isolated, tmp_path, monkeypatch):
    from merlin.targetgen import capsule_golden as G
    from merlin.targetgen import native_component_inputs as N

    standard, document, _, _, selected = isolated
    for row in document["members"]:
        Path(row["products"]["source"]["path"]).write_text(
            "module { func.func @forward(%x: tensor<3xi64>) -> tensor<3xi64> { return %x : tensor<3xi64> } }"
        )
        row["products"]["source"] = M.R._pin(row["products"]["source"]["path"])
    owner = M.prepare(standard_ir=standard, destination=tmp_path / "members")
    member = M.OriginalCandidateMember(owner, 0)
    row = member.verify()
    root = Path(row["capsule_root"])
    cb = {
        "operand_naming": "positional",
        "kernel_abi": {"kind": "whole_program", "outputs": ["out0"]},
        "tensors": {
            "arg0": {"shape": [3], "dtype": "i64", "role": "input"},
            "out0": {"shape": [3], "dtype": "i64", "role": "output"},
        },
    }
    for name in ("canonical_input_values", "canonical_input_raws", "golden", "compare"):
        monkeypatch.setattr(G, name, lambda *_a, **_kw: pytest.fail("original inputs reached a legacy golden"))
    bound, values, mapping = N._bind(row["envelope"], cb, root / "source.mlir", member)
    _, inputs = member.contract_inputs()
    assert base64.b64decode(bound["tensors"]["arg0"]["preload_b64"]) == inputs[0].data
    assert values["arg0"]["values"] == [(1 << 53) + 3] * 3
    assert mapping["outputs"] == {"Y": "out0"} and cb["tensors"]["arg0"].get("preload_b64") is None
    assert selected.verify()["outputs"][0]["dtype"] == "int64"
    cb["tensors"]["out0"]["shape"] = [3.0]
    with pytest.raises(N.NativeComponentExecutionError, match="exact source tensor shape"):
        N._bind(row["envelope"], cb, root / "source.mlir", member)


def test_original_selection_cannot_be_omitted_or_replaced(isolated, tmp_path, monkeypatch):
    owner = M.prepare(standard_ir=isolated[0], destination=tmp_path / "members")
    record = {"coverage_sha256": "coverage", "mandatory_source_blockers": []}
    preparation = SimpleNamespace(
        semantic_cases=SimpleNamespace(standard_ir=isolated[0]), require_complete=lambda: record
    )
    from merlin_experiments.phase1 import component_qualification_domain as D

    # Refusal is before any score or ordinary grader can be used as authority.
    monkeypatch.setattr(D, "_source_record", lambda _: record)
    report = {"sha256": "coverage", "obligations": []}
    with pytest.raises(StageGateError, match="omitted"):
        Q.obligations(report, preparation=preparation)
    with pytest.raises(ValueError, match="live original"):
        Q.obligations(report, preparation=preparation, original_members=copy.copy(owner))


def test_original_binding_preserves_storage_identity_without_scalar_shape_aliases(isolated, tmp_path):
    standard, document, *_ = isolated
    for row in document["members"]:
        for slot in row["ordered_abi"]["inputs"] + row["ordered_abi"]["outputs"]:
            slot["dtype"] = "i64"
    owner = M.prepare(standard_ir=standard, destination=tmp_path / "members")
    assert owner.require_complete()["members"][0]["envelope"]["ordered_abi"]["outputs"][0]["dtype"] == "i64"
    document["members"][0]["ordered_abi"]["inputs"][0]["shape"] = [3.0]
    with pytest.raises(ValueError, match="ordered tensor ABI"):
        M.prepare(standard_ir=standard, destination=tmp_path / "changed")


def unreachable_service(*_args, **_kwargs):
    raise AssertionError("the diagnostic refusal must precede compiler or execution work")


def test_original_execution_error_keeps_private_details_out_of_caller_feedback(isolated, tmp_path, monkeypatch):
    from merlin.targetgen import native_component_execution as N
    from merlin.targetgen import package_runtime as P
    from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe, KernelStackFramePolicy
    from merlin.targetgen.contract.build_service import BuildOnlyService, file_digest
    from merlin.targetgen.contract.execution_service import FunctionalExecutionService
    from merlin.targetgen.contract.readback_policy import FULL_VALUES_B64, ReadbackPolicy

    owner = M.prepare(standard_ir=isolated[0], destination=tmp_path / "members")
    row = owner.require_complete()["members"][0]
    candidate, contract_root = tmp_path / "compiler", tmp_path / "contract"
    candidate.mkdir()
    contract_root.mkdir()
    (candidate / "tool.py").write_text("inert diagnostic compiler\n")
    (contract_root / "schema.json").write_text("{}\n")
    source = Path(__file__).absolute()
    pins = ((str(source), file_digest(source)),)
    recipe = HarnessBuildRecipe(
        tmp_path / "unselected-cc",
        (),
        (),
        tmp_path / "unselected.ld",
        0,
        kernel_stack_frame=KernelStackFramePolicy("unused_entry", 1024),
    )
    build = BuildOnlyService("owned_device", recipe, unreachable_service, pins)
    execution = FunctionalExecutionService(
        "owned_device",
        "owned_engine",
        unreachable_service,
        unreachable_service,
        pins,
        '{"scope":"private diagnostic fixture only"}',
    )
    monkeypatch.setattr(P, "active_package_executor", lambda: object())
    monkeypatch.setattr(P, "load_package", lambda *_a, **_kw: SimpleNamespace(manifest={"target": "owned_device"}))
    private_detail = str(tmp_path / "private-reference.json") + ": expected artificial answer bits 987654321"

    def refused(*_args, **_kwargs):
        raise ValueError(private_detail)

    monkeypatch.setattr(P, "integrity_scan", refused)
    output = tmp_path / "grade"
    with pytest.raises(N.NativeComponentExecutionError) as error:
        N.execute_component(
            package_dir=candidate,
            capsule_dir=Path(row["capsule_root"]),
            contract_root=contract_root,
            target="owned_device",
            out_dir=output,
            build_service=build,
            execution_service=execution,
            source_verifier=unreachable_service,
            readback_policy=ReadbackPolicy(FULL_VALUES_B64),
            timeout_s=30,
            original_member=M.OriginalCandidateMember(owner, 0),
        )
    assert str(error.value) == "original candidate execution failed; inspect candidate code and declared ABI"
    assert str(tmp_path) not in str(error.value) and "987654321" not in str(error.value)
    record = json.loads((output / "result.json").read_bytes())
    assert record["failure"]["detail"] == private_detail and "numeric_report" not in record
