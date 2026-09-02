"""Every interface op the parser accepts must have ABI semantics the graded agent can implement.

Two authorities decide different halves of the same question and nothing kept them in step. Which ops
a capsule may legally contain is decided by `interface_emit._NAMED_OP_OPERAND_KEYS` -- what
`boundary.grammar_mnemonics` calls the parser's own tables. What each op MEANS is stated in
`merlin/contract/command_buffer_abi.yaml`, which `targetgen.generate_prompt` hands to the agent as the
command-buffer contract.

So an op could be added to the parser, emitted into capsules, and left with no operands, no attributes
and no semantics in the contract the agent reads -- leaving the agent to guess the ABI. This repo has
measured what that produces: an arm invented an instruction encoding because the prompt never named
the shipped ISA files.

Found on introduction: five ops in that state, plus a sixth caught in the act -- `bias_add` was added
to the parser and to `interface_grammar.md` in one change and to the ABI only after this gate existed.
"""
from __future__ import annotations

import importlib.util
import sys

import yaml

from merlin.common.paths import merlin_dir, repo_root
from merlin.targetgen.contract import interface_emit as IE


def _gate():
    p = repo_root() / "build_tools" / "scripts" / "check_opcode_documentation.py"
    spec = importlib.util.spec_from_file_location("_opcode_gate", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_opcode_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def _abi_opcodes() -> set[str]:
    doc = yaml.safe_load(
        (merlin_dir() / "contract" / "command_buffer_abi.yaml").read_text(encoding="utf-8")) or {}
    return set(doc.get("opcodes") or ())


def test_the_parser_defines_the_bias_add_op_it_emits():
    """The op and its parser row must land together.

    `grammar_mnemonics` reads the parser's tables, so an emitted mnemonic the tables do not define
    makes the canonical parser REFUSE the module -- and every capsule using it classifies UNKNOWN, not
    as a defect in the capsule. Measured: the six PF capsules were in exactly that state between the
    op being emitted and this row existing.
    """
    assert "bias_add" in IE.defined_mnemonics()
    assert IE._NAMED_OP_OPERAND_KEYS["bias_add"] == ["src", "bias"]


def test_no_new_op_reaches_a_capsule_without_abi_semantics():
    """The gate's own verdict, ratchet included. The five pre-existing entries may only shrink."""
    gate = _gate()
    rep = gate.audit()
    assert rep["status"] == "ok", rep.get("detail")
    allowed = gate._ratchet(gate._DEFAULT_RATCHET)
    new = {m: oc for m, oc in rep["undocumented"].items() if m not in allowed}
    assert not new, (
        f"interface op(s) {sorted(new)} are accepted by the parser and have no ABI semantics; write "
        f"the command_buffer_abi.yaml entry in the same change that adds the parser row")
    stale = sorted(allowed - set(rep["undocumented"]))
    assert not stale, (
        f"ratchet entries {stale} are now documented — delete them; a ratchet that does not shrink "
        f"stops being a ratchet")


def test_the_gate_would_have_caught_the_op_that_prompted_it():
    """Non-vacuity, in the specific shape that actually happened.

    An empty ratchet must flag every undocumented op; that only proves the gate can fail. What matters
    is that a NEW op is separated from the inherited five, since that is the state a real change
    arrives in.
    """
    gate = _gate()
    rep = gate.audit()
    documented = set(rep["documented"])
    assert "bias_add" in documented, "the op that prompted this gate must itself be documented"

    with_empty = {m: oc for m, oc in rep["undocumented"].items()}
    assert len(with_empty) == 5, (
        f"expected exactly the five inherited ops, got {sorted(with_empty)}; if this changed, either "
        f"debt was paid (shrink the ratchet) or a new op arrived undocumented")


def test_bias_add_semantics_state_the_accumulator_dtype_rule():
    """The one thing an implementer gets wrong without being told.

    The bias lands on the accumulator before any requant, so on an i8 x i8 -> i32 datapath the vector
    is i32. An implementer who assumed the operand dtype would compute a different function from the
    golden, and the fused/unfused cycle comparison the op exists for would not be summable.
    """
    doc = yaml.safe_load(
        (merlin_dir() / "contract" / "command_buffer_abi.yaml").read_text(encoding="utf-8"))
    entry = doc["opcodes"]["BIAS_ADD"]
    assert set(entry["operands"]) == {"src", "bias", "dst"}
    sem = entry["semantics"].lower()
    assert "accumulator" in sem, "the dtype domain must be stated, not left to inference"
    assert "requant" in sem, "its position relative to requant is what fixes the domain"
    assert "bias[j]" in entry["semantics"], "the broadcast axis must be unambiguous"


def test_a_structural_op_is_not_reported_as_undocumented():
    """`tensor` declares a leaf and issues no command, so it owes the ABI nothing."""
    gate = _gate()
    rep = gate.audit()
    assert "tensor" in rep["structural"]
    assert "tensor" not in rep["undocumented"]
