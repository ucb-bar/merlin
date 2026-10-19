"""Source call and device work cannot borrow another call's successful proof."""

import pytest

from merlin.targetgen.owned_host_model_execution import _host_call_proof, _mandatory_source_commands


def test_proven_host_compute_veto_is_a_candidate_failure_without_native_tier_credit():
    from merlin_experiments.phase1.feedback.qa import _candidate_native_feedback

    from merlin.targetgen.owned_host_model_execution import _stopped_before_receipt

    check = _stopped_before_receipt({"stage": "owned_host_execution", "type": "HostComputeVeto",
                                     "detail": "scalar host work exceeds the source-derived budget"})
    assert check["status"] == "fail"
    assert check["failure"]["category"] == "host_compute_in_accelerator_group"
    feedback = _candidate_native_feedback({"model_execution_check": check})
    assert feedback["status"] == "fail"
    assert feedback["components"]["emitted_host_compute"] == "violation"
    assert "candidate_host_tensor_compute_violation" in feedback["violations"]
    assert all(status != "pass" for status in feedback.get("tiers", {}).values())
    unknown = _stopped_before_receipt({"stage": "owned_host_execution", "type": "ValueError",
                                       "detail": "unknown control-flow proof"})
    assert unknown["status"] == "incomplete"
    wrong_stage = _stopped_before_receipt({"stage": "prepare_runtime_bundle", "type": "HostComputeVeto"})
    assert wrong_stage["status"] == "incomplete"


def test_owned_build_audit_binds_logical_input_copy_to_original_source(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from merlin_experiments.phase1.feedback import private_device_audit, private_full_models
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    from merlin.compile.model_execution_inputs import file_sha256
    from merlin.targetgen import owned_host_model_execution as owned
    from merlin.xdsl_dialects.lowering import compute_groups as groups_api

    (tmp_path / "model.mlir").write_text("""module {
      func.func @forward(%a: tensor<2x2xi8>, %b: tensor<2x2xi8>, %init: tensor<2x2xi32>)
          -> tensor<2x2xi32> {
        %r = linalg.matmul ins(%a, %b : tensor<2x2xi8>, tensor<2x2xi8>)
          outs(%init : tensor<2x2xi32>) -> tensor<2x2xi32>
        return %r : tensor<2x2xi32>
      }
    }""")

    def groups(module, target):
        root = next(op for op in module.walk() if op.name == "linalg.matmul")
        root.attributes.update({"prov.region_id": StringAttr("original"),
                                "prov.source_node_ids": ArrayAttr([StringAttr("node")])})
        return [SimpleNamespace(index=4, placement=target, root=root, members=[root],
                                stages=[groups_api.CONTRACTION])]

    monkeypatch.setattr(groups_api, "form_groups", groups)
    monkeypatch.setattr(groups_api, "plan", lambda *a, **k: {})
    monkeypatch.setattr(groups_api, "require_explained", lambda _: None)
    monkeypatch.setattr(owned, "_source_placement", lambda *a, **k: {
        "eligible_source_regions": [{"source_group": 4}],
    })
    monkeypatch.setattr(private_full_models, "_linked_symbols", lambda *a: None)
    audits = []

    def audit(build, assigned, source, target, **kwargs):
        audits.append(source["conv_input_preparation"])
        return [{"status": "synthetic-native-boundary-only"}]

    monkeypatch.setattr(private_device_audit, "audit_built_device_host_compute", audit)
    entry = {"op": "conv2d", "Himg": 1, "Wimg": 2, "ci": 2, "kh": 1, "kw": 1, "N": 2,
             "stride": [1, 1], "padding": [0, 0, 0, 0]}
    emitted = {
        "device": "fixture", "granularity": "group", "signatures": {"kernel": []},
        "routed": [{"symbol": "kernel", "group": 9, "source_region_id": "original", "source_node_ids": ["node"],
                    "parallel": [2, 2], "reduction": [2], "dtypes": ["i8", "i8", "i32"]}],
        "entries": {"kernel": entry}, "programs": {"kernel": {"entry": dict(entry)}},
        "call_buffers": {"kernel": [
            {"role": "input_0", "shape": [1, 1, 2, 2], "dtype": "i8"},
            {"role": "weight_0", "shape": [2, 2], "dtype": "i8"},
            {"role": "out_0", "shape": [2, 2], "dtype": "i32"},
        ]},
    }
    elf, sidecar = tmp_path / "model.elf", tmp_path / "device.json"
    elf.write_bytes(b"synthetic test of orchestration; no native execution")

    def run():
        sidecar.write_text(json.dumps(emitted))
        return owned._audit_owned_host_device_build(tmp_path, {"output": {
            "elf": str(elf), "elf_sha256": file_sha256(elf),
            "device_sidecar": {"path": str(sidecar), "sha256": file_sha256(sidecar)},
        }}, target="fixture")

    result = run()
    assert result["joined_groups"] == {9: 4}
    assert audits[0]["kernel"]["source_group"] == 4
    assert audits[0]["kernel"]["call_buffers"] == emitted["call_buffers"]["kernel"]
    emitted["routed"][0]["parallel"] = [3, 2]
    emitted["entries"]["kernel"]["Wimg"] = 3
    emitted["programs"]["kernel"]["entry"]["Wimg"] = 3
    emitted["call_buffers"]["kernel"][0]["shape"] = [1, 1, 3, 2]
    emitted["call_buffers"]["kernel"][2]["shape"] = [3, 2]
    with pytest.raises(ValueError, match="original source contraction"):
        run()
    assert len(audits) == 1


def test_actual_host_cfg_preserves_each_reused_source_call(tmp_path):
    from merlin.llvmlower import toolchain

    if not toolchain.mlir_translate().is_file():
        pytest.skip("requires selected LLVM importer; no simulation")
    path = tmp_path / "host.ll"

    def source(body):
        path.write_text(
            'target triple = "riscv64-unknown-elf"\n'
            "declare void @device_call()\n"
            f"define void @forward(i1 %c) {{\n{body}\n}}\n"
            "define void @_mlir_ciface_forward(i1 %c) {\n"
            "call void @forward(i1 %c)\nret void\n}\n"
        )

    assigned = [{"symbol": "device_call"}, {"symbol": "device_call"}]
    source("call void @device_call()\ncall void @device_call()\nret void")
    assert _host_call_proof(path, assigned=assigned, output=tmp_path)["mandatory_call_roster"] == {"device_call": 2}
    source("call void @device_call()\nret void")
    with pytest.raises(ValueError, match="roster"):
        _host_call_proof(path, assigned=assigned, output=tmp_path)
    source(
        "br i1 %c, label %work, label %exit\nwork:\ncall void @device_call()\n"
        "call void @device_call()\nbr label %exit\nexit:\nret void"
    )
    with pytest.raises(ValueError, match="bypassed"):
        _host_call_proof(path, assigned=assigned, output=tmp_path)


def test_fused_owned_source_operations_need_individual_work_attribution():
    from xdsl.dialects.builtin import StringAttr
    from xdsl.parser import Parser

    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

    module = Parser(
        make_llvm_context(),
        """builtin.module {
      llvm.func @kernel(%p: !llvm.ptr) {
        %a = llvm.load %p : !llvm.ptr -> i64
        %b = llvm.load %p : !llvm.ptr -> i64
        llvm.return
      }
    }""",
    ).parse_module()
    function = next(op for op in module.body.block.ops if op.name == "llvm.func")
    first, second = [op for op in function.walk() if op.name == "llvm.load"]
    obligations = [{"source_op_index": i, "region_id": f"r{i}", "semantic_family": "contraction"} for i in range(2)]
    first.attributes["prov.region_id"] = StringAttr("r0")
    with pytest.raises(ValueError, match="operation 1"):
        _mandatory_source_commands(function, {"contraction": {first}}, obligations)
    second.attributes["prov.region_id"] = StringAttr("r1")
    assert len(_mandatory_source_commands(function, {"contraction": {first, second}}, obligations)) == 2
    obligations[1]["semantic_family"] = "elementwise_map"
    with pytest.raises(ValueError, match="elementwise_map"):
        _mandatory_source_commands(function, {"contraction": {first, second}}, obligations)


def test_claimed_numeric_result_without_actual_owned_build_never_promotes():
    from merlin.targetgen.capsule_grade import enforce_model_execution_check

    capsule = {"semantic": {"must_accelerate": True}, "required_oracle_tiers": ["L0", "L1", "L2", "L3"]}
    result = {
        "status": "pass",
        "kind": "model",
        "owned_host_native_execution": {
            "status": "verified",
            "numeric": {"status": "pass"},
            "tiers": {tier: {"status": "pass"} for tier in capsule["required_oracle_tiers"]},
        },
    }
    checked = enforce_model_execution_check(result, capsule, target="synthetic")
    assert checked["status"] == "incomplete"
    assert checked["model_execution_check"]["status"] == "incomplete"


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("optimized", [False, True])
def test_device_custody_reopens_real_translation_compile_and_symbol_only_rename(tmp_path, isolated, optimized):
    from merlin.llvmlower import device_build, toolchain
    from merlin.targetgen.owned_host_model_execution import _verify_device_custody

    objcopy = device_build._objcopy()
    if not objcopy or not toolchain.mlir_translate().is_file() or not toolchain.clang().is_file():
        pytest.skip("requires selected compiler tools; no accelerator or simulation")
    artifact, translated, raw, obj = [tmp_path / name for name in ("kernel.mlir", "kernel.ll", "raw.o", "kernel.o")]
    optimized_ir, optimized_mlir = tmp_path / "optimized.ll", tmp_path / "optimized.mlir"
    artifact.write_text("module { llvm.func @kernel() { llvm.return } }\n")
    custody = {}
    commands = [
        (
            "translation",
            "device_llvm_translation",
            [toolchain.mlir_translate(), "--mlir-to-llvmir", str(artifact), "-o", str(translated)],
            artifact,
            translated,
        ),
    ]
    if optimized:
        commands += [
            (
                "optimization",
                "device_llvm_optimization",
                [
                    toolchain.clang(),
                    "--target=riscv64-unknown-elf",
                    "-O2",
                    "-S",
                    "-emit-llvm",
                    str(translated),
                    "-o",
                    str(optimized_ir),
                ],
                translated,
                optimized_ir,
            ),
            (
                "optimized_import",
                "device_optimized_llvm_import",
                [
                    toolchain.mlir_translate(),
                    "--import-llvm",
                    "--mlir-print-op-generic",
                    str(optimized_ir),
                    "-o",
                    str(optimized_mlir),
                ],
                optimized_ir,
                optimized_mlir,
            ),
        ]
    object_input = optimized_ir if optimized else translated
    flags = ["-O2", "-Xclang", "-disable-llvm-passes"] if optimized else []
    commands += [
        (
            "compilation",
            "device_object_compilation",
            [toolchain.clang(), "--target=riscv64-unknown-elf", *flags, "-c", str(object_input), "-o", str(raw)],
            object_input,
            raw,
        ),
        (
            "rename",
            "device_object_symbol_rename",
            (
                device_build.rename_object_command(objcopy, "kernel", "kernel_0", raw, obj)
                if isolated
                else [objcopy, "--redefine-sym=kernel=kernel_0", str(raw), str(obj)]
            ),
            raw,
            obj,
        ),
    ]
    for role, stage, command, source, output in commands:
        run, custody[role], _elapsed_s = device_build._observed_build_tool(
            command, timeout=30, directory=tmp_path, stage=stage, inputs=(source,), output=output
        )
        assert run.returncode == 0, run.stderr
    from merlin.targetgen.native_model_execution import _digest

    entry = {"custody": custody}
    if optimized:
        entry["artifacts"] = {
            "optimized_llvm_ir": _digest(optimized_ir),
            "optimized_llvm_mlir": _digest(optimized_mlir),
        }
    assert _verify_device_custody(
        entry,
        artifact=artifact,
        llvm_ir=translated,
        obj=obj,
        original_symbol="kernel",
        kernel_symbol="kernel_0",
    ) == (optimized_mlir if optimized else None)
    if optimized:
        selected_copy = tmp_path / "shared-translation-copy.ll"
        selected_copy.write_bytes(translated.read_bytes())
        assert (
            _verify_device_custody(
                entry,
                artifact=artifact,
                llvm_ir=selected_copy,
                obj=obj,
                original_symbol="kernel",
                kernel_symbol="kernel_0",
            )
            == optimized_mlir
        )
        for role in ("optimization", "optimized_import"):
            missing = {**entry, "custody": {key: value for key, value in custody.items() if key != role}}
            with pytest.raises(ValueError, match="complete codegen custody"):
                _verify_device_custody(
                    missing,
                    artifact=artifact,
                    llvm_ir=translated,
                    obj=obj,
                    original_symbol="kernel",
                    kernel_symbol="kernel_0",
                )
        unrelated = tmp_path / "unrelated.ll"
        unrelated.write_bytes(optimized_ir.read_bytes())
        wrong = {**entry, "artifacts": {**entry["artifacts"], "optimized_llvm_ir": _digest(unrelated)}}
        with pytest.raises(ValueError, match="exact selected LLVM transform"):
            _verify_device_custody(
                wrong,
                artifact=artifact,
                llvm_ir=translated,
                obj=obj,
                original_symbol="kernel",
                kernel_symbol="kernel_0",
            )
        saved = optimized_ir.read_bytes()
        optimized_ir.write_bytes(saved + b"; changed\n")
        with pytest.raises(ValueError, match="changed"):
            _verify_device_custody(
                entry,
                artifact=artifact,
                llvm_ir=translated,
                obj=obj,
                original_symbol="kernel",
                kernel_symbol="kernel_0",
            )
        optimized_ir.write_bytes(saved)
        repeated_raw, repeated_obj = tmp_path / "reoptimized.raw.o", tmp_path / "reoptimized.o"
        run, repeated_compile, _ = device_build._observed_build_tool(
            [
                toolchain.clang(),
                "--target=riscv64-unknown-elf",
                "-O2",
                "-c",
                str(optimized_ir),
                "-o",
                str(repeated_raw),
            ],
            timeout=30,
            directory=tmp_path,
            stage="device_object_compilation",
            inputs=(optimized_ir,),
            output=repeated_raw,
        )
        assert run.returncode == 0, run.stderr
        run, repeated_rename, _ = device_build._observed_build_tool(
            device_build.rename_object_command(objcopy, "kernel", "kernel_0", repeated_raw, repeated_obj),
            timeout=30,
            directory=tmp_path,
            stage="device_object_symbol_rename",
            inputs=(repeated_raw,),
            output=repeated_obj,
        )
        assert run.returncode == 0, run.stderr
        repeated = {**entry, "custody": {**custody, "compilation": repeated_compile, "rename": repeated_rename}}
        with pytest.raises(ValueError, match="exact selected LLVM transform"):
            _verify_device_custody(
                repeated,
                artifact=artifact,
                llvm_ir=translated,
                obj=repeated_obj,
                original_symbol="kernel",
                kernel_symbol="kernel_0",
            )
    with pytest.raises(ValueError, match="rename transform"):
        _verify_device_custody(
            entry,
            artifact=artifact,
            llvm_ir=translated,
            obj=obj,
            original_symbol="kernel",
            kernel_symbol="wrong_symbol",
        )
    obj.write_bytes(obj.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="product changed"):
        _verify_device_custody(
            entry,
            artifact=artifact,
            llvm_ir=translated,
            obj=obj,
            original_symbol="kernel",
            kernel_symbol="kernel_0",
        )


def test_optimized_inventory_requires_complete_selected_structural_decode():
    from types import SimpleNamespace

    from merlin.targetgen.owned_host_model_execution import _optimized_command_functs

    module = SimpleNamespace(walk=lambda: [SimpleNamespace(name="llvm.inline_asm") for _ in range(3)])
    table = {"custom_opcode": 123, "names": {"0": "configure", "4": "compute"}}
    trace = {
        "abi": {"custom_opcode": "0x7b"},
        "instructions": [
            {"class": "CONFIG", "funct": 0},
            {"class": "COMPUTE", "funct": 4},
            {"class": "FENCE", "funct": None},
        ],
    }
    assert _optimized_command_functs(module, trace, table, source_functs=[0, 4, 4]) == [0, 4]
    for source in ([], [0], [0, 4, 6]):
        with pytest.raises(ValueError, match="introduced or discarded"):
            _optimized_command_functs(module, trace, table, source_functs=source)
    for invalid in (
        {**trace, "abi": {"custom_opcode": "0x0b"}},
        {**trace, "instructions": trace["instructions"][:2]},
        {**trace, "instructions": [{"class": "UNKNOWN", "funct": 0}, *trace["instructions"][1:]]},
        {**trace, "instructions": [{"class": "CONFIG", "funct": 99}, *trace["instructions"][1:]]},
    ):
        with pytest.raises(ValueError, match="decode|unknown command|ISA identity"):
            _optimized_command_functs(module, invalid, table, source_functs=[0, 4])


def test_empty_global_class_roster_is_explicit_only_for_source_bound_models(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from merlin.targetgen import target_experiment
    from merlin.targetgen.native_model_execution import NativeModelExecutionError, _frozen_model_policy

    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: synthetic\n")
    corpus = tmp_path / "corpus"
    (corpus / "isa").mkdir(parents=True)
    (corpus / "MANIFEST.yaml").write_text(
        "instruction_policy:\n  status: resolved\n  prohibited_instruction_roles: []\n"
    )
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    declaration = capsule / "capsule.yaml"
    declaration.write_text(
        "kind: model\nexpected:\n  instruction_classes: []\nrequired_oracle_tiers: [L0, L1, L2, L3]\n"
    )
    (capsule / "expected_instruction_coverage.yaml").write_text("instruction_classes: []\n")
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(descriptor))
    monkeypatch.setattr(
        target_experiment,
        "load_target_experiment",
        lambda path: SimpleNamespace(target="synthetic", capsule_corpus=corpus / "isa"),
    )
    with pytest.raises(NativeModelExecutionError, match="coverage"):
        _frozen_model_policy(capsule, target="synthetic")
    policy = _frozen_model_policy(capsule, target="synthetic", source_bound_groups=True)
    assert policy["expected_instruction_classes"] == []
    assert policy["required_tiers"] == ["L0", "L1", "L2", "L3"]
    declaration.write_text(declaration.read_text().replace("kind: model", "kind: layer"))
    with pytest.raises(NativeModelExecutionError, match="original model"):
        _frozen_model_policy(capsule, target="synthetic", source_bound_groups=True)


def test_owned_score_withholds_legacy_numeric_tiers_and_cost_when_proof_missing():
    from merlin.targetgen.capsule_grade import _candidate_score_view, _score_numeric_status

    row = {
        "kind": "model",
        "numeric": {"status": "pass"},
        "metrics": {"cycles": 10},
        "tiers": {"L3": {"status": "pass", "derived_from_rtl": True}},
        "model_execution_check": {
            "kind": "owned_host_model_execution",
            "status": "incomplete",
            "violations": ["candidate_completed_dispatch_unverified"],
        },
    }
    assert _score_numeric_status(row) is None
    view = _candidate_score_view(row)
    assert view["tiers"] == {} and view["numeric"] == {} and "metrics" not in view


def test_protected_engine_selection_uses_scoped_facts_without_changing_ambient(tmp_path, monkeypatch):
    import json
    import os

    from merlin.compile import model_execution_inputs as inputs
    from merlin.targetgen.owned_host_model_execution import _native_engine_for_facts
    from merlin.targetgen.rtl import facts as facts_api

    path = tmp_path / "selected-facts.json"
    document = {"scope": "synthetic selected input; no hardware qualification"}
    path.write_text(json.dumps(document))
    identity = {"path": str(path), "config": "SelectedConfig", "sha256": "synthetic_test_only"}
    monkeypatch.setenv("MERLIN_RTL_FACTS", "different-ambient-selection")
    monkeypatch.setattr(inputs, "selected_firrtl", lambda *a, **k: identity)

    def bind(target, engine, selected):
        assert facts_api.load_facts(target, regenerate=False) == document
        assert selected == identity
        assert os.environ["MERLIN_RTL_FACTS"] == "different-ambient-selection"
        return "synthetic boundary test; no execution"

    monkeypatch.setattr(inputs, "native_engine", bind)
    assert _native_engine_for_facts("synthetic", "gsim", identity) == "synthetic boundary test; no execution"
    assert "synthetic" not in facts_api._OBSERVED.get()
    assert os.environ["MERLIN_RTL_FACTS"] == "different-ambient-selection"
    monkeypatch.setattr(inputs, "selected_firrtl", lambda *a, **k: {**identity, "sha256": "changed"})
    with pytest.raises(ValueError, match="facts changed"):
        _native_engine_for_facts("synthetic", "gsim", identity)


def test_historical_model_flag_cannot_bypass_owned_receipt_audit(monkeypatch):
    from merlin.targetgen import capsule_grade, owned_host_model_execution

    capsule = {'semantic': {'must_accelerate': False}, 'lanes': {'require': ['on_mesh']}}
    result = {'status': 'pass', 'owned_host_native_execution': {'status': 'claimed_pass'}}
    observed = []

    def audit(actual, original, **kwargs):
        assert actual is result and original is capsule
        observed.append(kwargs['target'])
        return {'status': 'incomplete', 'kind': 'owned_host_model_execution',
                'violations': ['candidate_required_tiers_unverified']}

    monkeypatch.setattr(owned_host_model_execution, 'audit_owned_host_model', audit)
    checked = capsule_grade.enforce_model_execution_check(result, capsule, target='synthetic')
    assert observed == ['synthetic']
    assert checked['status'] == 'incomplete'
    assert checked['model_execution_check']['kind'] == 'owned_host_model_execution'
    assert capsule['semantic']['must_accelerate'] is False


def test_declared_eligible_mesh_model_without_actual_receipt_refuses_legacy_numeric_credit(monkeypatch):
    from merlin.targetgen import native_model_execution
    from merlin.targetgen.capsule_grade import enforce_model_execution_check
    monkeypatch.setattr(native_model_execution, 'independent_frozen_source_eligibility',
                        lambda *a, **k: {'n_eligible': 1})
    capsule = {'semantic': {'must_accelerate': False}, 'lanes': {'require': ['on_mesh']}}
    result = {'status': 'pass', 'kind': 'model', 'numeric': {'status': 'pass'},
              'mesh_execution': {'matmul_layers_on_mesh': 2}}
    checked = enforce_model_execution_check(result, capsule, target='synthetic')
    assert checked['status'] == 'incomplete'
    assert checked['model_execution_check']['kind'] == 'owned_host_model_execution'
    assert 'receipt or original frozen capsule is absent' in checked['model_execution_check']['detail']


def test_ineligible_host_model_without_receipt_keeps_legacy_check(monkeypatch):
    from merlin.targetgen import capsule_grade, native_model_execution, owned_host_model_execution

    monkeypatch.setattr(native_model_execution, 'independent_frozen_source_eligibility',
                        lambda *a, **k: {'n_eligible': 0})
    def unexpected(*args, **kwargs):
        raise AssertionError('pure host model must not require a device receipt')
    monkeypatch.setattr(owned_host_model_execution, 'audit_owned_host_model', unexpected)
    capsule = {'semantic': {'must_accelerate': False}, 'lanes': {'require': ['on_mesh']}}
    checked = capsule_grade.model_execution_check({}, capsule, target='synthetic')
    assert checked.get('kind') != 'owned_host_model_execution'
