"""Selected instruction evidence; fixtures grant no native or numerical qualification."""

import copy
import runpy

import pytest

from merlin.common.paths import repo_root
from merlin.runtime.backends import base
from merlin.targetgen import trace_check


def _bound(tmp_path, monkeypatch, *, width=1, protocol=False, zero_fill=None, readout=False):
    fixtures = runpy.run_path(str(repo_root() / "merlin/tests/runtime/test_bound_chipyard_rocc_backend.py"))
    contract, facts = fixtures["selection"].__wrapped__(tmp_path)
    classes = {2: "LOAD_CMD", 3: "STORE_CMD", 4: "COMPUTE_AND_FLIP_CMD", 6: "PRELOAD_CMD"}
    contract["rocc_operand_roles"]["instructions"] = [
        {"funct": code, "class": name, "operands": {}} for code, name in classes.items()
    ]
    facts["facts"]["interfaces"][0]["legal_funct"] = list(classes)
    contract["corpus_instruction_roles"] = {"by_role": {
        "memory": ["LOAD_CMD"], "acc_readout": ["STORE_CMD"],
        "matmul": ["COMPUTE_AND_FLIP_CMD"], "weight_load": ["PRELOAD_CMD"],
    }}
    if protocol:
        facts["facts"]["interfaces"].append({"name": "register_bundle_layouts", "bundles": {
            "Preload": {"width": 64, "fields": {"acc": {"offset": 4, "width": width, "slot_width": 1}}},
        }})
        contract["rocc_operand_protocol"] = {
            "version": 1,
            "encoding": {"semantic_class": {2: "MVIN", 3: "MVOUT", 4: "COMPUTE_PRELOADED", 6: "PRELOAD"}},
            "roles": {"classes": {"PRELOAD": {"fields": [
                {"name": "acc_bit", "operand": "rs2", "bundle": "Preload", "field": "acc"},
                {"name": "accumulate", "from": "acc_bit", "equals": 1},
            ]}}},
        }
    if readout:
        contract["rocc_operand_protocol"]["roles"]["classes"]["MVOUT"] = {"fields": [
            {"name": "narrow", "operand": "rs2", "bundle": "Preload", "field": "acc"},
            {"name": "readout", "all": [{"from": "narrow", "equals": 1}],
             "set": "i8", "clear": "i32"},
        ]}
    if zero_fill is not None:
        fields = [{"name": "dram", "operand": "rs1", "as": "operand"}]
        classes = contract["rocc_operand_protocol"]["roles"]["classes"]
        classes["MVOUT"] = {"fields": copy.deepcopy(fields)}
        classes["MVIN"] = {"fields": fields + [
            {"name": "dram_raw", "operand": "rs1", "as": "raw"},
            {"name": "dram_source", "all": [{"from": "dram_raw", "equals": zero_fill}],
             "set": "zero_fill", "clear": "memory"}]}
    backend = fixtures["_bound"]((contract, facts))
    monkeypatch.setattr(base, "get_backend", lambda target: backend)
    return backend


def _trace(backend, codes=(2, 3), *, rs2=0):
    semantics = backend.rocc_semantics
    isa = semantics.isa_constants("synthetic")
    rows = [{"index": 0, "class": "FENCE", "funct": None, "decoded": {}}]
    for code in codes:
        left = {"kind": "argbase", "raw": None, "arg_index": 0, "offset": 0}
        right = {"kind": "const", "raw": rs2, "arg_index": None, "offset": None}
        label, decoded = semantics.decode_instruction(code, left, right, isa)
        rows.append({"index": len(rows), "class": label, "funct": code,
                     "rs1": left, "rs2": right, "decoded": decoded})
    rows.append({"index": len(rows), "class": "FENCE", "funct": None, "decoded": {}})
    return {"abi": {"custom_opcode": hex(isa["CUSTOM_OPCODE"]), "funct3": hex(isa["FUNCT3"])},
            "instructions": rows}


def test_public_movement_vocabulary_uses_selected_load_and_store_roles(tmp_path, monkeypatch):
    # R30's public movement reproducer emits LOAD_CMD/STORE_CMD, not legacy MVIN/MVOUT.
    backend = _bound(tmp_path, monkeypatch)
    result = trace_check.check(_trace(backend), {"instruction_classes": ["LOAD_CMD"], "modes": {"movement": True}},
                               target="synthetic")
    assert result["status"] == "pass" and result["trace_requirement_status"] == "pass"
    missing = trace_check.check(_trace(backend, (2,)), {"modes": {"movement": True}}, target="synthetic")
    assert missing["status"] == "fail" and missing["trace_requirement_status"] == "failed"


@pytest.mark.parametrize("mutation", ["class", "decoded", "abi"])
def test_claimed_class_fields_or_other_isa_cannot_replace_original_decode(tmp_path, monkeypatch, mutation):
    trace = _trace(_bound(tmp_path, monkeypatch))
    if mutation == "class":
        trace["instructions"][1]["class"] = "STORE_CMD"
    elif mutation == "decoded":
        trace["instructions"][1]["decoded"] = {"accumulate": True}
    else:
        trace["abi"]["custom_opcode"] = "0x01"
    result = trace_check.check(trace, {}, target="synthetic")
    assert result["status"] == "fail" and result["trace_requirement_status"] == "invalid_binding"


def test_mode_fields_without_selected_protocol_remain_unavailable(tmp_path, monkeypatch):
    backend = _bound(tmp_path, monkeypatch)
    result = trace_check.check(_trace(backend, (6,)), {"modes": {"k_accumulate": True}}, target="synthetic")
    assert result["status"] == "fail"
    assert result["trace_requirement_status"] == "unavailable_protocol"
    assert "k_accumulate" in result["unavailable_requirements"]


def test_operand_layout_is_not_a_selected_tile_schedule(tmp_path, monkeypatch):
    backend = _bound(tmp_path, monkeypatch, protocol=True)
    result = trace_check.check(_trace(backend), {}, cb={"commands": [{"opcode": "MATMUL"}]}, target="synthetic")
    assert result["status"] == "pass" and result["trace_requirement_status"] == "pass"
    assert result["unavailable_requirements"] == []
    assert result["tile_schedule_advisory"]["required"] is False
    movement = trace_check.check(_trace(backend), {}, cb={"commands": []}, target="synthetic")
    assert movement["status"] == "pass"


def test_partial_protocol_does_not_invent_resident_load_identity(tmp_path, monkeypatch):
    backend = _bound(tmp_path, monkeypatch, protocol=True)
    result = trace_check.check(_trace(backend, (2, 3, 3)), {"modes": {"resident_reuse": True}}, target="synthetic")
    assert result["trace_requirement_status"] == "unavailable_fields"
    assert result["unavailable_requirements"] == ["resident_reuse"]


def test_weight_bank_stay_class_cannot_replace_accumulation_destination_flag(tmp_path, monkeypatch):
    from merlin.targetgen.rocc.normalized_operands import BoundOperandObserver

    backend = _bound(tmp_path, monkeypatch, protocol=True)
    semantics = backend.rocc_semantics
    isa = semantics.isa_constants("synthetic")
    protocol = {"version": 1, "encoding": {"semantic_class": {
        2: "MVIN", 3: "MVOUT", 4: "COMPUTE_ACCUMULATE", 6: "PRELOAD",
    }}, "roles": {"classes": {}}}
    # This selected fixture labels a retained-bank operation. No accumulator flag is observed.
    semantics.operand_observer = BoundOperandObserver(
        target="synthetic", protocol=protocol, facts=backend._data()["facts"], raw_isa=isa, word_bits=64,
    )
    result = trace_check.check(_trace(backend, (6, 4)), {"modes": {"k_accumulate": True}}, target="synthetic")
    # Unobservable is missing evidence, not a violation; it still never completes the requirement.
    assert result["status"] == "pass" and result["trace_requirement_status"] == "unavailable_fields"
    assert result["unavailable_requirements"] == ["k_accumulate"] and not trace_check.requirements_complete(result)


@pytest.mark.parametrize("bits,passed", [
    (0xffffffff, False), (0x7fc00000, False), (0x7f800000, False), (0x3f800000, False), (0x3f000000, True),
])
def test_keep_scale_sentinel_and_nonfinite_bits_are_not_genuine_scale_updates(tmp_path, monkeypatch, bits, passed):
    from merlin.targetgen.rocc.normalized_operands import BoundOperandObserver

    backend = _bound(tmp_path, monkeypatch, protocol=True)
    semantics = backend.rocc_semantics
    facts = backend._data()["facts"]
    layouts = next(row for row in facts["facts"]["interfaces"] if row.get("name") == "register_bundle_layouts")
    layouts["bundles"]["Scale"] = {"width": 64, "fields": {"scale": {"offset": 0, "width": 32}}}
    protocol = {"version": 1, "encoding": {"semantic_class": {
        2: "CONFIG_ST", 3: "MVOUT", 4: "COMPUTE_PRELOADED", 6: "PRELOAD",
    }}, "roles": {"classes": {"CONFIG_ST": {"fields": [
        {"name": "acc_scale", "operand": "rs2", "bundle": "Scale", "field": "scale", "as": "f32"},
    ]}}}}
    semantics.operand_observer = BoundOperandObserver(
        target="synthetic", protocol=protocol, facts=facts, raw_isa=semantics.isa_constants("synthetic"), word_bits=64,
    )
    result = trace_check.check(_trace(backend, (2,), rs2=bits), {"modes": {"acc_scale": True}}, target="synthetic")
    assert (result["status"] == "pass") is passed


@pytest.mark.parametrize("width,rs2,expected", [(1, 16, "pass"), (1, 0, "failed"), (None, 16, "unavailable_fields")])
def test_accumulation_consumes_actual_source_width_and_register_bits(tmp_path, monkeypatch, width, rs2, expected):
    backend = _bound(tmp_path, monkeypatch, protocol=True, width=width)
    result = trace_check.check(_trace(backend, (6,), rs2=rs2),
                               {"instruction_classes": ["PRELOAD_CMD"], "modes": {"k_accumulate": True}},
                               target="synthetic")
    assert result["trace_requirement_status"] == expected
    assert (result["status"] == "pass") == (expected != "failed")  # unavailable is not a violation
    assert trace_check.requirements_complete(result) == (expected == "pass")


def test_refusal_coverage_requires_original_evidence_and_retains_host_veto(tmp_path, monkeypatch):
    from merlin.targetgen import coverage_report, expected_refusal

    monkeypatch.setattr(coverage_report, "_selected_tiers", lambda _: ({"L2": "spike"}, frozenset(), None))
    monkeypatch.setattr("merlin.targetgen.eligibility.capability_map_for_target", lambda _: {})
    monkeypatch.setattr("merlin.targetgen.eligibility.undetermined_families_for_target", lambda _: frozenset())
    cap = {"name": "refusal", "operation": {"op": "matmul"}, "semantic": {"eligible": True, "must_accelerate": True},
           "expected": expected_refusal.declaration(
               reason="unsupported source stage", evidence="original stage binding"),
           }
    decline = {"reason": "declared unsupported stage"}
    status, failure, record = expected_refusal.verdict(cap, "declined", None, decline)
    row = {"capsule": "refusal", "status": status, "failure": failure, "tiers": {},
           "declined": decline, "expected_refusal": record}
    coverage = coverage_report._acceleratable_coverage([row], {"refusal": cap}, target="fixture")
    assert coverage["n_eligible"] == 0 and coverage["validated_expected_refusals"] == ["refusal"]
    forged = copy.deepcopy(row)
    forged["expected_refusal"]["expected"]["evidence"] = "different"
    coverage = coverage_report._acceleratable_coverage([forged], {"refusal": cap}, target="fixture")
    assert coverage["must_accelerate_violations"] == ["refusal"]
    ordinary = {**cap, "name": "ordinary", "expected": {}}
    host_veto = {"capsule": "ordinary", "status": "fail", "tiers": {"L2": {"status": "pass"}},
                 "failure": {"category": "host_compute_in_accelerator_group"}}
    coverage = coverage_report._acceleratable_coverage([host_veto], {"ordinary": ordinary}, target="fixture")
    assert coverage["must_accelerate_violations"] == ["ordinary"]


def test_provided_host_structural_scope_cannot_be_replaced_by_trace_skip_or_flag(tmp_path, monkeypatch):
    import json

    import yaml

    from merlin.compile.model_execution_inputs import file_sha256
    from merlin.targetgen import owned_host_leaf_execution as host

    cap = {"name": "host", "kind": "model_slice", "required_oracle_tiers": ["L2"],
           "software_screen": {"status": "admitted", "decisions": [{"placement": "host", "status": "admitted"}]}}
    (tmp_path / "capsule.yaml").write_text(yaml.safe_dump(cap))
    elf = tmp_path / "host.elf"
    elf.write_bytes(b"fixture ELF bytes; no execution authority")
    (tmp_path / "source.mlir").write_text("module {}")
    (tmp_path / "golden.yaml").write_text("{}")
    (tmp_path / "facts.json").write_text("{}")
    # Positive scope projection is fixture-only. Actual source hashing remains independently exercised.
    from merlin.targetgen.capsule_runner import _content_identity

    identity = _content_identity({key: tmp_path / name for key, name in (
        ("capsule", "capsule.yaml"), ("source", "source.mlir"), ("golden", "golden.yaml"),
    )}, root=tmp_path)
    record = {"source_identity": identity, "structure": {"elf": str(elf), "elf_sha256": file_sha256(elf)},
              "build": {"output": {"elf_sha256": file_sha256(elf)}},
              "native": {"L2": {"elf_sha256": file_sha256(elf), "numeric": {"status": "pass"}}},
              "source_eligibility": {"regions": []},
              "structure_inputs": {"target": "fixture", "rtl_facts": {
                  "path": str(tmp_path / "facts.json"), "sha256": file_sha256(tmp_path / "facts.json"),
              }}}
    row = {"capsule": "host", "status": "pass", "trace_check": {"status": "skipped"},
           "provided_host_qualification": record}
    assert not host.qualified_host_result({**cap, "__dir__": str(tmp_path)}, row, target="fixture")
    row["host_execution_check"] = host._host_scope_record(cap, record)
    # A caller-created annotation cannot replace the actual source/recipe gates.
    assert not host.qualified_host_result({**cap, "__dir__": str(tmp_path)}, row, target="fixture")
    monkeypatch.setattr(host, "_original_host_source", lambda *a, **k: (cap, None, record["source_eligibility"]))
    monkeypatch.setattr(host, "_host_leaf_structure", lambda *a, **k: record["structure"])
    assert host.qualified_host_result({**cap, "__dir__": str(tmp_path)}, row, target="fixture")
    corrupted = json.loads(json.dumps(row))
    corrupted["provided_host_qualification"]["native"]["L2"]["numeric"]["status"] = "fail"
    assert not host.qualified_host_result({**cap, "__dir__": str(tmp_path)}, corrupted, target="fixture")
    elf.write_bytes(b"changed ELF")
    assert not host.qualified_host_result({**cap, "__dir__": str(tmp_path)}, row, target="fixture")


@pytest.mark.parametrize("status", ["failed", "unavailable_protocol", "unavailable_fields", None, "pass"])
def test_public_numeric_native_pass_is_only_a_screen_when_structure_is_unqualified(status):
    from merlin_experiments.phase1.feedback import selfcheck

    capsule = {"name": "public_case", "kind": "layer", "label": "public"}
    row = {"capsule": "public_case", "status": "pass", "numeric": {"status": "pass"},
           "tiers": {"L2": {"status": "pass"}, "L3": {"status": "pass"}}}
    if status is not None:
        row["trace_check"] = {"status": "pass" if status == "pass" else "fail", "trace_requirement_status": status}
    certified, screened, refused, structural = selfcheck._operator_feedback_state(
        row, capsule, target="fixture", barrier_status="pass",
    )
    assert certified == (status == "pass")
    assert screened == (status != "pass") and not refused
    assert row["numeric"]["status"] == "pass" and row["tiers"]["L3"]["status"] == "pass"
    counts = selfcheck._selfcheck_counts(
        result_rows=1, gated_rows=0, requested_size=1,
        passed=1, certified=int(certified), scope="all", suite_size=1,
    )
    assert counts["all_pass"] == counts["certified_complete"] == certified
    assert (structural["status"] == "pass") == certified
    if status is None:
        assert structural["status"] == "unavailable_record"


def test_public_refusal_requires_original_binding_and_host_veto_never_certifies():
    from merlin_experiments.phase1.feedback import selfcheck

    from merlin.targetgen import expected_refusal

    cap = {"name": "refusal", "expected": expected_refusal.declaration(reason="unsupported", evidence="original")}
    decline = {"reason": "unsupported source stage"}
    status, failure, record = expected_refusal.verdict(cap, "declined", None, decline)
    row = {"capsule": "refusal", "status": status, "failure": failure, "tiers": {},
           "expected_refusal": record, "declined": decline}
    result = selfcheck._operator_feedback_state(row, cap, target="fixture", barrier_status=None)
    assert result[:3] == (True, False, True)
    forged = copy.deepcopy(row)
    forged["expected_refusal"]["expected"]["evidence"] = "other"
    result = selfcheck._operator_feedback_state(forged, cap, target="fixture", barrier_status=None)
    assert result[:3] == (False, False, False)
    assert selfcheck._operator_feedback_state(row, None, target="fixture", barrier_status=None)[0] is False
    placed = {"capsule": "ordinary", "status": "pass", "numeric": {"status": "pass"},
              "failure": {"category": "host_compute_in_accelerator_group"}, "trace_check": {"status": "pass"}}
    result = selfcheck._operator_feedback_state(placed, {"name": "ordinary"}, target="fixture", barrier_status="pass")
    assert not result[0] and result[3]["component"] == "host_accelerator_placement"


@pytest.mark.parametrize("structural", [None, False, True])
def test_authoritative_qa_requires_structure_without_relabeling_numeric_pass(tmp_path, monkeypatch, structural):
    import yaml
    from merlin_experiments.phase1.context import InvocationContext
    from merlin_experiments.phase1.feedback import qa

    cap_root = tmp_path / "corpus" / "case"
    cap_root.mkdir(parents=True)
    (cap_root / "capsule.yaml").write_text(yaml.safe_dump({"name": "case", "required_oracle_tiers": ["L3"]}))
    monkeypatch.setattr(qa, "_loop_target_sim_via", lambda context: ("fixture", ""))
    monkeypatch.setattr(qa.CR, "qa_checkpoint_adapters", lambda *args, **kwargs: {"L3": object()})
    score = {"n_capsules": 1, "n_passed": 1, "structural_evidence_all_pass": structural,
             "cohort_admission": {"n_admitted_capsules": 1,
                                  "admitted_name_set_sha256": qa.CG._name_set_sha256(["case"])},
             "per_capsule": [{"capsule": "case", "status": "pass", "numeric": "pass", "tiers": {"L3": "pass"}}]}
    monkeypatch.setattr(qa.CG, "grade", lambda *args, **kwargs: score)
    context = InvocationContext(tmp_path, tmp_path / "target.yaml", tmp_path, "fixture",
                                tmp_path / "runs", tmp_path / "reports", tmp_path / "bundles", ())
    verdict = qa.run("submission", str(cap_root.parent), tmp_path / "runs", {"public"}, False, 1, context=context)
    assert verdict["all_pass"] == (structural is True)
    assert verdict["n_passed"] == 1 and verdict["per_capsule"][0]["numeric_status"] == "pass"


@pytest.mark.parametrize("mutation", ["missing", "gated", "timeout", "duplicate", "roster", "refusal"])
def test_author_cohort_guard_never_certifies_a_shrunken_or_unbound_required_set(mutation):
    from merlin_experiments.phase1.feedback import loop_grading, qa

    score = {"n_capsules": 2, "n_passed": 2,
             "cohort_admission": {"n_admitted_capsules": 2,
                                  "admitted_name_set_sha256": qa.CG._name_set_sha256(["A", "B"])},
             "per_capsule": [{"capsule": "A", "status": "pass"}, {"capsule": "B", "status": "pass"}]}
    assert qa._original_cohort_complete(score)
    if mutation == "missing":
        score.update(n_capsules=1, n_passed=1, per_capsule=score["per_capsule"][:1])
    elif mutation == "gated":
        score["n_gated_deferred"] = 1
    elif mutation == "timeout":
        score["n_cert_not_measured"] = 1
    elif mutation == "duplicate":
        score["per_capsule"][1]["capsule"] = "A"
    elif mutation == "roster":
        score["cohort_admission"]["admitted_name_set_sha256"] = "0" * 64
    else:
        score["expected_refusal_scope"] = {"binding": "original_capsule_declaration_and_canonical_result_v1",
            "n_admitted_capsules": 2, "n_declared_refusals": 1, "n_validated_refusals": 0, "n_executable_capsules": 1}
    assert not qa._original_cohort_complete(score)
    red = {row["capsule"]: {"status": "pass"} for row in score["per_capsule"]}
    verdict = loop_grading._fast_loop_verdict_doc(red, ["L3"], [], structural_evidence=True,
                                                cohort_complete=qa._original_cohort_complete(score))
    assert verdict["all_pass"] is False and verdict["n_passed"] == len(red)


@pytest.mark.parametrize("qualified", [False, True])
def test_formal_host_scope_requires_provided_execution_evidence(qualified):
    from merlin_experiments.phase1.feedback import formal

    fixtures = runpy.run_path(str(repo_root() / "packages/merlin-experiments/tests/test_hidden_capsule_policy.py"))
    score = fixtures["clean_score"]()
    score.update(tier_reached={"L2": 2, "L3": 2}, pass_evidence={"rtl_backed": 2},
                 trace_all_pass=None, provided_host_execution_all_pass=qualified,
                 structural_evidence_all_pass=qualified,
                 structural_evidence_scope={"n_instruction_trace_capsules": 0, "n_model_execution_capsules": 0,
                                            "n_provided_host_capsules": 2})
    complete, failures = formal.phase_completion(score)
    assert complete is qualified
    assert ("provided_host_execution_evidence_not_complete" in failures) == (not qualified)


@pytest.mark.parametrize("sentinel", [0, 19])
def test_source_generation_is_selected_semantics_not_literal_address_exception(tmp_path, monkeypatch, sentinel):
    backend = _bound(tmp_path, monkeypatch, protocol=True, zero_fill=sentinel)
    for code, value, allowed in ((2, sentinel, True), (2, sentinel + 1, False),
                                  (3, sentinel, False)):
        trace = _trace(backend, (code,))
        trace['instructions'][1]['rs1'] = {'kind': 'const', 'raw': value}
        result = trace_check.check(trace, {}, target='synthetic', address_model='pointer_args')
        baked = [v for v in result['violations'] if 'BAKED DRAM' in v]
        assert not baked if allowed else baked
    # A pointer or unresolved input cannot acquire the generated-source label.
    for operand in ({'kind': 'argbase', 'raw': None, 'arg_index': 0, 'offset': 0},
                    {'kind': 'unknown', 'raw': None}):
        trace = _trace(backend, (2,)); trace['instructions'][1]['rs1'] = operand
        view = trace_check._bound_trace_view(trace, 'synthetic')[0]
        assert view['instructions'][1]['decoded']['dram_source'] is None
    # Caller-authored annotations cannot replace original raw decoding.
    forged = _trace(backend, (2,))
    forged['instructions'][1]['rs1'] = {'kind': 'const', 'raw': sentinel + 1}
    forged['instructions'][1]['decoded'] = {'dram_source': 'zero_fill'}
    result = trace_check.check(forged, {}, target='synthetic', address_model='pointer_args')
    assert result['trace_requirement_status'] == 'invalid_binding'


def _accumulation_loop(opcode, *, accumulate=True, unknown_exit=False, poison=False, iterations=2):
    from merlin.targetgen.rocc.decode import _parse_module

    expression = 'llvm.shl %one, %wide : i64' if poison else 'llvm.select %first, %zero, %flag : i1, i64'
    flag = 16 if accumulate else 0
    exit_branch = 'llvm.cond_br %opaque, ^done, ^done' if unknown_exit else 'llvm.br ^done'
    text = f'''builtin.module {{
      llvm.func @entry(%opaque: i1) {{
        %zero = llvm.mlir.constant(0 : i64) : i64
        %one = llvm.mlir.constant(1 : i64) : i64
        %two = llvm.mlir.constant({iterations} : i64) : i64
        %wide = llvm.mlir.constant(64 : i64) : i64
        %flag = llvm.mlir.constant({flag} : i64) : i64
        llvm.inline_asm has_side_effects "fence", "~{{memory}}" : () -> ()
        llvm.br ^loop(%zero : i64)
      ^loop(%iv: i64):
        %more = llvm.icmp "slt" %iv, %two : i64
        llvm.cond_br %more, ^body, ^exit
      ^body:
        %first = llvm.icmp "eq" %iv, %zero : i64
        %bits = {expression}
        %dest = llvm.or %bits, %zero : i64
        llvm.inline_asm has_side_effects ".insn r {opcode}, 0x3, 6, x0, $0, $1", "r,r,~{{memory}}" %zero, %dest : (i64, i64) -> ()
        llvm.inline_asm has_side_effects ".insn r {opcode}, 0x3, 4, x0, $0, $1", "r,r,~{{memory}}" %zero, %zero : (i64, i64) -> ()
        %next = llvm.add %iv, %one : i64
        llvm.br ^loop(%next : i64)
      ^exit:
        {exit_branch}
      ^done:
        llvm.inline_asm has_side_effects "fence", "~{{memory}}" : () -> ()
        llvm.return
      }}
    }}'''
    module = _parse_module(text)
    assert module is not None
    return module


@pytest.mark.parametrize('accumulate,expected', [(True, 'pass'), (False, 'failed')])
def test_counted_loop_accumulation_uses_visited_selected_operand_bits(tmp_path, monkeypatch, accumulate, expected):
    from merlin.targetgen.rocc.decode import decode_module

    backend = _bound(tmp_path, monkeypatch, protocol=True)
    module = _accumulation_loop(backend.rocc_semantics.isa_constants('synthetic')['CUSTOM_OPCODE'],
                                accumulate=accumulate)
    trace = decode_module(module, target='synthetic')
    result = trace_check.check(trace, {'modes': {'k_accumulate': True}}, target='synthetic', lowered_module=module)
    assert result['trace_requirement_status'] == expected, result
    assert (result['status'] == 'pass') is accumulate


@pytest.mark.parametrize('case', ['unknown_branch', 'poison', 'budget'])
def test_incomplete_control_witness_discards_earlier_accumulation(tmp_path, monkeypatch, case):
    from merlin.targetgen.rocc.control_flow import constant_control_operand_witness
    from merlin.targetgen.rocc.decode import decode_module

    backend = _bound(tmp_path, monkeypatch, protocol=True)
    module = _accumulation_loop(backend.rocc_semantics.isa_constants('synthetic')['CUSTOM_OPCODE'],
                                unknown_exit=case == 'unknown_branch', poison=case == 'poison')
    trace = decode_module(module, target='synthetic')
    kwargs = {'max_blocks': 1} if case == 'budget' else {}
    assert constant_control_operand_witness(module, target='synthetic', expected_trace=trace,
        semantic_class='PRELOAD', field='accumulate', value=True, **kwargs) is None
    if case != 'budget':
        result = trace_check.check(trace, {'modes': {'k_accumulate': True}}, target='synthetic', lowered_module=module)
        assert result['trace_requirement_status'] == 'unavailable_fields', result


def test_control_witness_refuses_different_original_module(tmp_path, monkeypatch):
    from merlin.targetgen.rocc.decode import decode_module

    backend = _bound(tmp_path, monkeypatch, protocol=True)
    opcode = backend.rocc_semantics.isa_constants('synthetic')['CUSTOM_OPCODE']
    module = _accumulation_loop(opcode)
    trace = decode_module(module, target='synthetic')
    trace['instructions'][1]['rs1']['raw'] = 7
    # Rebuild claimed raw decode: still cannot substitute another source module.
    row = trace['instructions'][1]
    row['class'], row['decoded'] = backend.rocc_semantics.decode_instruction(
        row['funct'], row['rs1'], row['rs2'], backend.rocc_semantics.isa_constants('synthetic'))
    result = trace_check.check(trace, {'modes': {'k_accumulate': True}}, target='synthetic', lowered_module=module)
    assert result['trace_requirement_status'] == 'invalid_binding'
    assert 'original lowered-module' in result['violations'][0]


def test_already_proved_accumulation_does_not_interpret_again(tmp_path, monkeypatch):
    from merlin.targetgen.rocc import control_flow
    from merlin.targetgen.rocc.decode import _parse_module

    backend = _bound(tmp_path, monkeypatch, protocol=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('already proved mode must not trigger traversal')
    monkeypatch.setattr(control_flow, 'constant_control_operand_witness', forbidden)
    trace_check.check(_trace(backend, (6,), rs2=16), {'modes': {'k_accumulate': True}},
                      target='synthetic', lowered_module=_parse_module('builtin.module {}'))


def _readout_loop(opcode, **kwargs):
    from xdsl.dialects.builtin import StringAttr
    from merlin.targetgen.rocc.decode import _asm_template

    module = _accumulation_loop(opcode, **kwargs)
    for op in tuple(module.walk()):
        if op.name != 'llvm.inline_asm':
            continue
        template = _asm_template(op)
        if ', 6,' in template:
            op.properties['asm_string'] = StringAttr(template.replace(', 6,', ', 3,'))
        elif ', 4,' in template:
            op.parent_block().erase_op(op)
    module.verify()
    return module


@pytest.mark.parametrize('narrow,expected', [(True, 'pass'), (False, 'failed')])
def test_counted_readout_uses_original_selected_operand_bits(tmp_path, monkeypatch, narrow, expected):
    from merlin.targetgen.rocc.decode import decode_module

    backend = _bound(tmp_path, monkeypatch, protocol=True, readout=True)
    module = _readout_loop(backend.rocc_semantics.isa_constants('synthetic')['CUSTOM_OPCODE'], accumulate=narrow)
    trace = decode_module(module, target='synthetic')
    before = copy.deepcopy(trace)
    result = trace_check.check(trace, {'modes': {'i8': True}}, target='synthetic', lowered_module=module)
    assert result['trace_requirement_status'] == expected, result
    assert (result['status'] == 'pass') is narrow
    assert trace == before  # The observer does not rewrite the original static trace.


@pytest.mark.parametrize('case', ['unknown_branch', 'poison', 'budget'])
def test_incomplete_readout_path_cannot_qualify_earlier_i8(tmp_path, monkeypatch, case):
    from merlin.targetgen.rocc.control_flow import constant_control_operand_witness
    from merlin.targetgen.rocc.decode import decode_module

    backend = _bound(tmp_path, monkeypatch, protocol=True, readout=True)
    module = _readout_loop(backend.rocc_semantics.isa_constants('synthetic')['CUSTOM_OPCODE'],
                           unknown_exit=case == 'unknown_branch', poison=case == 'poison')
    trace = decode_module(module, target='synthetic')
    kwargs = {'max_blocks': 1} if case == 'budget' else {}
    assert constant_control_operand_witness(module, target='synthetic', expected_trace=trace,
        semantic_class='MVOUT', field='readout', value='i8', **kwargs) is None
    if case != 'budget':
        result = trace_check.check(trace, {'modes': {'i8': True}}, target='synthetic', lowered_module=module)
        assert result['trace_requirement_status'] == 'unavailable_fields', result


def test_readout_witness_cannot_replace_a_different_original_trace(tmp_path, monkeypatch):
    from merlin.targetgen.rocc.decode import decode_module

    backend = _bound(tmp_path, monkeypatch, protocol=True, readout=True)
    module = _readout_loop(backend.rocc_semantics.isa_constants('synthetic')['CUSTOM_OPCODE'])
    trace = decode_module(module, target='synthetic')
    row = trace['instructions'][1]
    row['rs1'] = {'kind': 'const', 'raw': 7, 'arg_index': None, 'offset': None}
    row['class'], row['decoded'] = backend.rocc_semantics.decode_instruction(
        row['funct'], row['rs1'], row['rs2'], backend.rocc_semantics.isa_constants('synthetic'))
    result = trace_check.check(trace, {'modes': {'i8': True}}, target='synthetic', lowered_module=module)
    assert result['trace_requirement_status'] == 'invalid_binding'
    assert 'original lowered-module' in result['violations'][0]


def test_already_observed_i8_needs_no_control_interpretation(tmp_path, monkeypatch):
    from merlin.targetgen.rocc import control_flow
    from merlin.targetgen.rocc.decode import _parse_module

    backend = _bound(tmp_path, monkeypatch, protocol=True, readout=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('already observed mode must not trigger traversal')
    monkeypatch.setattr(control_flow, 'constant_control_operand_witness', forbidden)
    result = trace_check.check(_trace(backend, (3,), rs2=16), {'modes': {'i8': True}},
                              target='synthetic', lowered_module=_parse_module('builtin.module {}'))
    assert 'i8' not in result.get('unavailable_requirements', [])
    assert not any('mode i8' in reason for reason in result['violations'])


@pytest.mark.parametrize('readout', [False, True])
@pytest.mark.parametrize('unknown_exit', [False, True])
def test_large_loop_witness_requires_complete_compressed_control_proof(tmp_path, monkeypatch, readout, unknown_exit):
    from merlin.targetgen.rocc.control_flow import constant_control_operand_witness
    from merlin.targetgen.rocc.decode import decode_module

    backend = _bound(tmp_path, monkeypatch, protocol=True, readout=readout)
    build = _readout_loop if readout else _accumulation_loop
    module = build(backend.rocc_semantics.isa_constants('synthetic')['CUSTOM_OPCODE'],
                   iterations=100_000, unknown_exit=unknown_exit)
    trace = decode_module(module, target='synthetic')
    observed = constant_control_operand_witness(
        module, target='synthetic', expected_trace=trace,
        semantic_class='MVOUT' if readout else 'PRELOAD',
        field='readout' if readout else 'accumulate', value='i8' if readout else True,
    )
    # A bounded exact traversal can witness iteration 1 after proving the entire
    # closed loop and exit. A later opaque branch still invalidates that witness.
    assert observed is (None if unknown_exit else True)
