"""`to_generic_form` — the one-way spelling bridge from the contract surface to what MLIR can read.

An MLIR dialect registered dynamically from IRDL (`mlir-opt --irdl-file=`, `irdl::loadDialects`)
has no custom parser, because a generated dialect has no `assemblyFormat` to run. It can read only
the generic op spelling. The capsule corpus is written in the pretty form, so before this bridge
existed `--irdl-file` parsed 0 of the 370 `merlin_iface` capsules — and did so with rc=1 and an
EMPTY stderr, which is why the gap went unnoticed for months.

These tests are about the property that makes the bridge trustworthy: the module means the SAME
thing afterwards, and anything it cannot re-spell raises instead of passing through half-converted.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.contract.interface_emit import (
    InterfaceGrammarError, op_mnemonics, parse_interface_mlir, to_generic_form)

_PRETTY = """module attributes {merlin_iface.version = "0.1", merlin_iface.target = "t", \
merlin_iface.abi_version = "0.1"} {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<16x16xi8>
  %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<16x16xi8>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<16x16xi8>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %A0, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
"""


def test_every_op_is_respelled_generically():
    out = to_generic_form(_PRETTY)
    for mnemonic in ("tensor", "resident_pack", "matmul", "commit", "evict"):
        assert f'"merlin_iface.{mnemonic}"(' in out
    assert "= merlin_iface." not in out


def test_the_module_header_is_left_alone():
    """`module attributes {merlin_iface.version = ...}` spells METADATA in the same namespace, and
    the grammar's shape decomposition finds a `version` "op" there. Rewriting it produces a bogus
    `"merlin_iface.version"() : () ->` and the module loses its version, target and abi_version."""
    out = to_generic_form(_PRETTY)
    assert _PRETTY.splitlines()[0] == out.splitlines()[0]
    assert '"merlin_iface.version"' not in out


def test_an_operandless_op_gains_an_empty_operand_list():
    """`merlin_iface.tensor` prints a BARE result type (`attr-dict : type($result)`); generic form
    needs the functional `() -> T`. Which of the two the op uses is read off the line, not tabled."""
    assert '"merlin_iface.tensor"() {name = "W", role = "weight"} : () -> tensor<16x16xi8>' \
        in to_generic_form(_PRETTY)


def test_a_resultless_op_keeps_its_operands():
    assert '"merlin_iface.evict"(%W_res) : (!merlin_iface.resident) -> ()' in to_generic_form(_PRETTY)


def test_the_op_sequence_survives_the_respelling():
    """Meaning-preservation, stated as the thing that would actually break: the ops, in order."""
    assert op_mnemonics(to_generic_form(_PRETTY)) == op_mnemonics(_PRETTY)


def test_it_refuses_a_module_the_grammar_does_not_define():
    """FAIL CLOSED, for the reason `parse_interface_mlir` does: a partially converted module parses
    as a DIFFERENT, shorter program, and the caller cannot tell a short one from a complete one."""
    bad = _PRETTY.replace("merlin_iface.matmul", "merlin_iface.not_a_real_op")
    with pytest.raises(InterfaceGrammarError):
        to_generic_form(bad)


def test_the_shipped_examples_convert_and_keep_their_program():
    """Against real contract text, not only the fixture above."""
    examples = sorted((repo_root() / "merlin" / "contract" / "examples").glob("*.interface.mlir"))
    assert examples
    for ex in examples:
        pretty = ex.read_text()
        generic = to_generic_form(pretty)
        assert generic != pretty
        assert op_mnemonics(generic) == op_mnemonics(pretty)
        # The header is what carries target/abi_version into the command buffer.
        assert parse_interface_mlir(pretty)["target"] in generic
