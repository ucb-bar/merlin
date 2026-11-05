"""Finite source-domain checks; synthetic fixtures qualify no timing engine."""

import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import test_ordinary_archive_screen as archive_tests
from merlin_experiments.phase2 import _archive_source_domain as D
from merlin_experiments.phase2 import ordinary_archive_screen as S
from merlin_experiments.phase2.contracts import canonical_json
from test_ordinary_search_policy import ordinary_inputs as ordinary_inputs

from merlin.targetgen.contract.interface_emit import emit_interface_mlir
from merlin.targetgen.oot_starterkit.plan import validate_mixed_program_plan

source_fixture = archive_tests.source


def source(m=2, k=3, n=4):
    return f'''builtin.module {{
      func.func @forward(%a: tensor<{m}x{k}xi8>, %b: tensor<{k}x{n}xi8>) -> tensor<{m}x{n}xi32> {{
        %zero = arith.constant 0 : i32
        %init = tensor.splat %zero : tensor<{m}x{n}xi32>
        %out = linalg.generic {{indexing_maps = [affine_map<(m,n,k)->(m,k)>,
          affine_map<(m,n,k)->(k,n)>, affine_map<(m,n,k)->(m,n)>],
          iterator_types = ["parallel", "parallel", "reduction"]}}
          ins(%a, %b : tensor<{m}x{k}xi8>, tensor<{k}x{n}xi8>)
          outs(%init : tensor<{m}x{n}xi32>) {{
          ^bb0(%x: i8, %y: i8, %acc: i32):
            %wx = arith.extsi %x : i8 to i32
            %wy = arith.extsi %y : i8 to i32
            %p = arith.muli %wx, %wy : i32
            %sum = arith.addi %acc, %p : i32
            linalg.yield %sum : i32
          }} -> tensor<{m}x{n}xi32>
        func.return %out : tensor<{m}x{n}xi32>
      }}
    }}'''


def interface(m=2, k=3, n=4):
    return dict(abi_version="0.1", target="synthetic", tensors={
        "a": dict(shape=[m, k], dtype="i8", role="input"),
        "b": dict(shape=[k, n], dtype="i8", role="weight")}, commands=[
        dict(opcode="RES_PACK", operands=dict(src="b", dst="packed"), attributes=dict(layout="packed_rhs")),
        dict(opcode="MATMUL_RESIDENT", operands=dict(lhs="a", rhs="packed", dst="acc")),
        dict(opcode="COMMIT", operands=dict(src="acc", dst="out"), attributes=dict(epilogue=[], output_dtype="i32")),
        dict(opcode="EVICT", operands=dict(handle="packed"))])


def catalog(profile):
    return dict(sources={"source": dict(semantic_profile=profile)}, cases=[dict(case=dict(
        source_group_sha256="training-group", source=dict(source_sha256="source", visibility="public"),
        split="train", mechanism_family="matmul"), records=[dict(
            native_observation_available=True, scope_binding=dict(begin_pc=1, end_pc=2),
            originally_measured=True, original_numerical_status="pass", cycles=100)])])


def compiled_fixture(tmp_path):
    capdir, workspace = tmp_path / "capsule", tmp_path / "compilation"
    capdir.mkdir()
    workspace.mkdir()
    raw = source().encode()
    (capdir / "capsule.interface.mlir").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    plan = dict(schema="mixed_program_plan_v1", source_sha256=digest, source_op_count=3,
                entry_bindings=["a", "b"], output_bindings=["out"],
                source_values=[dict(op_index=i, result_index=0, tensor=name) for i, name in ((1, "init"), (2, "out"))],
                compiler_temporaries=[], schedule_instruction_count=3, prologue_instruction_range=[0, 0],
                epilogue_instruction_range=[3, 3], tasks=[dict(
                    task_index=i, kind="device" if i == 2 else "host", source_op_indices=[i],
                    instruction_start=i, instruction_end=i + 1, reads=[], writes=[] if i == 0 else [name],
                ) for i, name in enumerate((None, "init", "out"))])
    cb = dict(tensors={"a": dict(shape=[2, 3], dtype="i8", role="input"),
                       "b": dict(shape=[3, 4], dtype="i8", role="input"),
                       **{name: dict(shape=[2, 4], dtype="i32", role=role)
                          for name, role in (("init", "intermediate"), ("out", "output"))}},
              kernel_abi=dict(kind="whole_program", outputs=["out"], args=[
                  dict(tensor=name, access="write" if name in ("init", "out") else "read")
                  for name in ("a", "b", "init", "out")]), params=dict(global_program_plan=plan))
    assert validate_mixed_program_plan(raw, cb)["ok"]
    cb_raw = json.dumps(cb).encode()
    (workspace / "command_buffer.json").write_bytes(cb_raw)
    compiled = dict(status="observed", candidate_diagnostics=dict(
        artifacts=dict(source=dict(sha256=digest), command_buffer=dict(sha256=hashlib.sha256(cb_raw).hexdigest())),
        source_emission_join=dict(status="verified", source_sha256=digest)))
    member = SimpleNamespace(
        source_dir=capdir, source_sha256=hashlib.sha256(b"synthetic frozen closure").hexdigest(),
        descriptor=dict(operation=dict(op="model"), interface_mlir="capsule.interface.mlir"))
    return member, compiled, workspace, catalog(D.typed_profile(raw.decode()))


def test_complete_typed_source_matches_original_raw_readout_semantics():
    assert D.typed_profile(source()) == D.interface_profile(emit_interface_mlir(interface()))


@pytest.mark.parametrize("mutation", ["nonzero", "unsigned", "wrong_body", "transpose", "extra", "return_init"])
def test_labels_or_dot_appearance_cannot_hide_other_source_effects(mutation):
    text = source()
    if mutation == "nonzero":
        text = text.replace("constant 0", "constant 1")
    elif mutation == "unsigned":
        text = text.replace("arith.extsi", "arith.extui")
    elif mutation == "wrong_body":
        text = text.replace("arith.addi", "arith.subi")
    elif mutation == "transpose":
        text = text.replace("(k,n)>", "(n,k)>")
    elif mutation == "extra":
        text = text.replace("%out =", "%unused = arith.constant 1 : i32\n%out =")
    else:
        text = text.replace("func.return %out", "func.return %init")
    with pytest.raises(Exception):
        D.typed_profile(text)


@pytest.mark.parametrize("mutation", ["scale", "chain", "floating", "pack_layout"])
def test_archive_profiles_preserve_readout_dtype_composition_and_layout(mutation):
    cb = interface()
    if mutation == "scale":
        cb["commands"][2]["attributes"].update(epilogue=["acc_scale"], acc_scale=1.0)
    elif mutation == "chain":
        cb["commands"].insert(2, copy.deepcopy(cb["commands"][1]))
    elif mutation == "floating":
        cb["tensors"]["a"]["dtype"] = "f32"
    else:
        cb["commands"][0]["attributes"]["layout"] = "transposed_rhs"
    with pytest.raises(ValueError):
        D.interface_profile(emit_interface_mlir(cb))


def test_joint_shape_support_is_exact_and_uses_only_training_source_groups():
    profile = D.typed_profile(source())
    evidence = catalog(profile)
    assert D.train_support(evidence, profile) == ["training-group"]
    # All extents lie inside independently observed bounds, but the cross tuple
    # does not occur. Independent M/K/N ranges cannot grant joint support.
    evidence["sources"]["source"]["semantic_profile"] = D.typed_profile(source(2, 6, 4))
    assert D.train_support(evidence, profile) == []
    evidence = catalog(profile)
    for split in ("calibration", "test"):
        evidence["cases"][0]["case"]["split"] = split
        assert D.train_support(evidence, profile) == []


@pytest.mark.parametrize("damage", ["private", "missing", "numeric", "counter", "unmeasured"])
def test_geometry_alone_does_not_supply_eligible_public_training_evidence(damage):
    profile = D.typed_profile(source())
    evidence = catalog(profile)
    row = evidence["cases"][0]
    if damage == "private":
        row["case"]["source"]["visibility"] = "hidden"
    elif damage == "missing":
        row["records"][0]["native_observation_available"] = False
    elif damage == "numeric":
        row["records"][0]["original_numerical_status"] = "fail"
    elif damage == "counter":
        row["records"][0]["scope_binding"] = None
    else:
        row["records"][0]["originally_measured"] = False
    assert D.train_support(evidence, profile) == []


def test_live_current_source_and_compiled_plan_are_reopened(tmp_path):
    member, compiled, workspace, evidence = compiled_fixture(tmp_path)
    result = D.current(member, compiled, workspace, evidence)
    assert result["status"] == "source_domain_supported" and result["family"] == "matmul"
    assert result["training_source_groups"] == ["training-group"]
    assert result["authority"].endswith("no target or timing qualification")


@pytest.mark.parametrize("mutation", ["source", "compiled_source", "plan", "unverified", "capacity", "unseen"])
def test_source_family_cannot_be_granted_by_model_labels_or_copied_proofs(tmp_path, mutation):
    member, compiled, workspace, evidence = compiled_fixture(tmp_path)
    if mutation == "source":
        (member.source_dir / "capsule.interface.mlir").write_text(source() + "\n")
    elif mutation == "compiled_source":
        compiled["candidate_diagnostics"]["artifacts"]["source"]["sha256"] = "0" * 64
    elif mutation == "plan":
        path = workspace / "command_buffer.json"
        cb = json.loads(path.read_bytes())
        cb["params"]["global_program_plan"]["output_bindings"] = ["init"]
        raw = json.dumps(cb).encode()
        path.write_bytes(raw)
        compiled["candidate_diagnostics"]["artifacts"]["command_buffer"]["sha256"] = hashlib.sha256(raw).hexdigest()
    elif mutation == "unverified":
        compiled["candidate_diagnostics"]["source_emission_join"]["status"] = "UNKNOWN"
    elif mutation == "capacity":
        member.descriptor["performance"] = dict(capacity=dict(regime="spilling"))
    else:
        evidence["sources"]["source"]["semantic_profile"]["n"] = 8
    assert D.current(member, compiled, workspace, evidence)["status"] == "UNKNOWN"


@pytest.mark.parametrize("supported", [True, False])
def test_ordinary_owner_checks_live_model_semantics_before_launching_count_observer(
    source_fixture, tmp_path, monkeypatch, supported,
):
    original = source_fixture
    member, compiled, workspace, evidence = compiled_fixture(tmp_path)
    if not supported:
        evidence["sources"]["source"]["semantic_profile"]["n"] = 8
    owner = S.issue_archive_screen(original["policy"], **original["arguments"])
    owner = replace(owner, retained_feedback_json=canonical_json(evidence), families=("matmul",))
    elf = workspace / "original.elf"
    elf.write_bytes(b"synthetic exact original executable")
    compiled["elf_sha256"] = hashlib.sha256(elf.read_bytes()).hexdigest()
    calls = []
    monkeypatch.setattr(S.OBS, "observe", lambda path, **kwargs: calls.append(path) or dict(status="observed"))
    result = owner.observe_member(original["policy"], member, workspace, compiled,
                                  sandbox=None, remaining_s=5, functional=True)
    assert bool(calls) is supported
    assert result["status"] == ("observed" if supported else "UNKNOWN")
    assert result["source_domain"]["profile"]["n"] == 4
