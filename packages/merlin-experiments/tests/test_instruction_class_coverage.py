"""Actual LLVM command operands, selected class projection and frozen obligations."""
import copy
import json
import runpy

import pytest
from xdsl.parser import Parser

from merlin.common.paths import repo_root
from merlin.runtime.backends import base
from merlin.runtime.backends.chipyard_rocc_bound import BoundRoCCBackend
from merlin.targetgen import trace_check
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
from merlin.targetgen.owned_host_model_execution import _require_instruction_classes
from merlin.targetgen.rocc import decode, semantics


@pytest.fixture
def selected(monkeypatch):
    fixture = runpy.run_path(str(repo_root() / "merlin/tests/targetgen/test_bound_operand_observer.py"))
    contract, facts = fixture["selection"]()

    def install(*, protocol=True):
        chosen = copy.deepcopy(contract)
        if not protocol:
            chosen.pop("rocc_operand_protocol")
        owner = semantics.bind(target=chosen["name"], contract=chosen, facts=facts)
        backend = BoundRoCCBackend(chosen["name"], json.dumps({"contract": chosen}), owner)
        monkeypatch.setattr(base, "get_backend", lambda _: backend)
        # decode imports this selector through its ordinary public backend path.
        return owner

    return contract, facts, install


def emitted(*, selector="1", unknown=False):
    # Input is a pointer; only the configuration discriminator is a constant.
    # Different opcode/funct/selector values from the target that exposed the bug.
    instructions = [(0, "%selector"), (4, "%address"), (9, "%address")]
    if unknown:
        instructions.append((31, "%address"))
    asm = "\n".join(
        f'"llvm.inline_asm"({arg}, %extent) <{{asm_string = ".insn r 0x2b, 0x7, {funct}, x0, $0, $1", '
        'constraints = "r,r,~{memory}", has_side_effects}> : (i64, i64) -> ()'
        for funct, arg in instructions)
    selector_op = ("%selector = llvm.ptrtoint %input : !llvm.ptr to i64" if selector == "pointer"
                   else f"%selector = llvm.mlir.constant({selector}) : i64")
    return Parser(make_llvm_context(), f'''module {{
      llvm.func @kernel(%input: !llvm.ptr) {{
        {selector_op}
        %address = llvm.ptrtoint %input : !llvm.ptr to i64
        %extent = llvm.mlir.constant(1048576) : i64
        {asm}
        "llvm.inline_asm"() <{{asm_string = "fence", constraints = "~{{memory}}", has_side_effects}}> : () -> ()
        llvm.return
      }}
    }}''').parse_module()


def observe():
    trace = decode.decode_module(emitted(), target="independent_array")
    before = copy.deepcopy(trace)
    coverage = trace_check.instruction_class_coverage(trace, target="independent_array")
    assert trace == before
    return trace, coverage


def test_actual_emitted_selector_projects_coverage_without_inventing_flush(selected):
    _, _, install = selected
    install()
    trace, coverage = observe()
    assert coverage["raw_classes"] == ["CONTROL", "FENCE", "INPUT", "OUTPUT"]
    assert coverage["normalized_classes"] == ["CONFIG_LD", "FENCE", "MVIN", "MVOUT"]
    _require_instruction_classes(["CONFIG_LD", "MVIN", "MVOUT", "INPUT"], coverage["observed_classes"])
    for missing in ("CONFIG_ST", "FLUSH", "COMPUTE_PRELOADED"):
        with pytest.raises(ValueError, match="omit frozen required instruction classes"):
            _require_instruction_classes([missing], coverage["observed_classes"])
    assert coverage["operand_protocol_selected"]
    trace["instructions"][0]["decoded"] = {"invented": 1}
    with pytest.raises(ValueError, match="original raw instruction decode"):
        trace_check.instruction_class_coverage(trace, target="independent_array")


@pytest.mark.parametrize("selector,unknown", [("pointer", False), ("3", False), ("1", True)])
def test_unresolved_selector_and_unknown_instruction_remain_refusals(selected, selector, unknown):
    _, _, install = selected
    install()
    trace = decode.decode_module(emitted(selector=selector, unknown=unknown), target="independent_array")
    with pytest.raises(ValueError, match="unresolved|original raw instruction decode"):
        trace_check.instruction_class_coverage(trace, target="independent_array")


def test_no_selected_protocol_keeps_original_raw_coverage(selected):
    _, _, install = selected
    install(protocol=False)
    _, coverage = observe()
    assert not coverage["operand_protocol_selected"] and coverage["normalized_classes"] == []
    assert coverage["observed_classes"] == coverage["raw_classes"]
    with pytest.raises(ValueError, match="omit frozen required instruction classes"):
        _require_instruction_classes(["MVIN"], coverage["observed_classes"])


@pytest.mark.parametrize("collision", ["base", "subtype"])
def test_overlapping_vocabulary_with_different_identity_refuses(selected, collision):
    contract, _, install = selected
    if collision == "base":
        contract["rocc_operand_roles"]["instructions"][1]["class"] = "MVOUT"
    else:
        contract["rocc_operand_roles"]["instructions"][1]["class"] = "CONFIG_LD"
    install()
    with pytest.raises(ValueError, match="ambiguous|collides"):
        observe()
