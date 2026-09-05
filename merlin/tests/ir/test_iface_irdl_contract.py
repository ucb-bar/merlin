"""The `merlin_iface` IRDL must actually register, and its constraints must actually fire.

Every check here is behavioural -- it runs `mlir-opt --irdl-file=` and reads the exit code. The
previous state of this contract passed review by inspection while rejecting *every* valid module
(`irdl.type @"!acc"` names the type `!acc`, so `!merlin_iface.acc` did not resolve), and nothing
noticed because no gate ever invoked the file. A source-string assertion would not have caught it.
"""
from __future__ import annotations
import subprocess
import pytest
from merlin.common.paths import repo_root
from merlin.targetgen.contract.interface_emit import op_mnemonics
from merlin.targetgen.rtl.gen_iface_irdl import (
    _CAPSULES, _header, _restore_type_params, _strip_type_sigil, lower_c_preds, verify)

_IRDL = repo_root() / "merlin/contract/merlin_iface.irdl.mlir"
_MLIROPT = repo_root() / "third_party/llvm-install/bin/mlir-opt"

# A minimal conformant module in GENERIC form -- the only form a dynamically registered dialect can
# parse (it has no custom parser), and in the `acc<...>` spelling the capsule corpus actually uses.
_VALID = """module attributes {merlin_iface.version = "0.1"} {
  %W = "merlin_iface.tensor"() {name = "W", role = "weight"} : () -> tensor<63x31xf8E4M3FN>
  %A = "merlin_iface.tensor"() {name = "A", role = "input"} : () -> tensor<32x63xf8E4M3FN>
  %R = "merlin_iface.resident_pack"(%W) {layout = "packed_rhs"} : (tensor<63x31xf8E4M3FN>) -> !merlin_iface.resident
  %acc = "merlin_iface.matmul"(%A, %R) : (tensor<32x63xf8E4M3FN>, !merlin_iface.resident) -> !merlin_iface.acc<bf16>
  %Y = "merlin_iface.commit"(%acc) {name = "Y", epilogue = [], output_dtype = "bf16"} : (!merlin_iface.acc<bf16>) -> tensor<32x31xbf16>
  "merlin_iface.evict"(%R) : (!merlin_iface.resident) -> ()
}
"""


def _run(tmp_path, src: str) -> int:
    f = tmp_path / "m.mlir"
    f.write_text(src)
    return subprocess.run([str(_MLIROPT), f"--irdl-file={_IRDL}", str(f), "-o", "/dev/null"],
                          capture_output=True, text=True).returncode


pytestmark = pytest.mark.skipif(not _MLIROPT.exists(), reason="LLVM mlir-opt not installed")


def test_the_contract_accepts_a_conformant_module(tmp_path):
    """The regression that hid for months: the IRDL rejected everything, valid included."""
    assert _run(tmp_path, _VALID) == 0


def test_the_parameterised_acc_type_keeps_its_parameter(tmp_path):
    """`tblgen-to-irdl` drops `let parameters`, which makes `acc<bf16>` unparseable."""
    assert _run(tmp_path, _VALID.replace("!merlin_iface.acc<bf16>", "!merlin_iface.acc")) == 1


@pytest.mark.parametrize("bad,why", [
    ("!merlin_iface.resident", "undeclared type"),
    ("merlin_iface", "undeclared dialect-qualified type"),
])
def test_an_undeclared_type_is_rejected(tmp_path, bad, why):
    assert _run(tmp_path, _VALID.replace(bad, bad + "_NOPE")) == 1, why


def test_a_wrong_operand_type_is_rejected(tmp_path):
    """matmul's rhs must be the resident handle, not the raw weight tensor."""
    src = _VALID.replace(
        '"merlin_iface.matmul"(%A, %R) : (tensor<32x63xf8E4M3FN>, !merlin_iface.resident)',
        '"merlin_iface.matmul"(%A, %W) : (tensor<32x63xf8E4M3FN>, tensor<63x31xf8E4M3FN>)')
    assert _run(tmp_path, src) == 1


def test_sigil_is_stripped_from_type_symbols():
    raw = '    irdl.type @"!acc"\n      %0 = irdl.base @merlin_iface::@"!acc"\n'
    assert "!" not in _strip_type_sigil(raw).replace("irdl.", "")


def test_declared_parameters_are_reattached_and_lowercased():
    out = _restore_type_params('    irdl.type @"acc" \n', {"acc": ["elemtype"]})
    assert "irdl.parameters(elemtype: %param0)" in out and "%param0 = irdl.any" in out


def test_a_type_without_declared_parameters_is_left_alone():
    line = '    irdl.type @"resident" \n'
    assert _restore_type_params(line, {"resident": []}) == line


# --------------------------------------------------------------------------------------------
# The constraints that used to be inert.
# --------------------------------------------------------------------------------------------
# `irdl.c_pred` is the one IRDL constraint op that does not implement VerifyConstraintInterface, so
# mlir-opt drops it from the enclosing `irdl.all_of` with no diagnostic and the conjunction loads
# EMPTY -- a constraint that cannot fail. Every case below verified CLEAN before the generator
# lowered those predicates into IRDL's own vocabulary. Each is behavioural for the same reason the
# tests above are: a source-string assertion would pass on a spec that checks nothing.

def test_a_result_that_is_not_a_tensor_is_rejected(tmp_path):
    """`commit` must produce a ranked tensor. Was `irdl.c_pred isa<RankedTensorType>`, i.e. inert."""
    src = _VALID.replace("(!merlin_iface.acc<bf16>) -> tensor<32x31xbf16>",
                         "(!merlin_iface.acc<bf16>) -> i32")
    assert _run(tmp_path, src) == 1


def test_an_unranked_tensor_result_is_rejected(tmp_path):
    """`builtin.tensor` is RankedTensorType exactly: an unranked tensor is a different base."""
    src = _VALID.replace("-> tensor<32x31xbf16>", "-> tensor<*xbf16>")
    assert _run(tmp_path, src) == 1


def test_a_non_array_epilogue_is_rejected(tmp_path):
    """`epilogue` is a StrArrayAttr; an integer is not an array. Was inert."""
    assert _run(tmp_path, _VALID.replace("epilogue = []", "epilogue = 42 : i64")) == 1


def test_a_non_string_name_is_rejected(tmp_path):
    """`name` is a StrAttr. The generator used to RELAX this to `irdl.any` to dodge a sigil bug,
    so any attribute at all was accepted."""
    assert _run(tmp_path, _VALID.replace('name = "Y"', "name = 42 : i64")) == 1


def test_the_contract_carries_no_constraint_that_cannot_fail(tmp_path):
    """No `irdl.c_pred`, and no `irdl.all_of()` of nothing.

    Both load as unconditional acceptance while reading as enforcement. If a future ODS constraint
    reaches the generator unrecognised it stays a c_pred, and this fails rather than the file
    quietly gaining a constraint that never fires."""
    text = _IRDL.read_text()
    body = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("//"))
    assert "irdl.c_pred" not in body
    assert "irdl.all_of()" not in body and "irdl.any_of()" not in body


def test_the_contract_states_what_it_does_not_check():
    """The ODS constraints IRDL cannot express are NAMED in the file, not silently absent."""
    header = "\n".join(l for l in _IRDL.read_text().splitlines() if l.startswith("//"))
    assert "does NOT check" in header
    assert "element type must not be a token" in header
    assert "element-wise constraint over a builtin ArrayAttr" in header


# --------------------------------------------------------------------------------------------
# `lower_c_preds` unit behaviour -- in particular that it FAILS LOUD on an unknown predicate.
# --------------------------------------------------------------------------------------------

_RAW_OP = '''module {
  irdl.dialect @d {
    irdl.operation @o {
      %0 = irdl.c_pred "(::llvm::isa<::mlir::RankedTensorType>($_self))"
      %1 = irdl.all_of(%0)
      %2 = irdl.c_pred "SOMETHING THE TABLE HAS NEVER SEEN"
      %3 = irdl.all_of(%1, %2)
      irdl.results(result: %3)
    }
  }
}
'''


def test_an_expressible_predicate_becomes_an_irdl_base():
    out, notes, unknown = lower_c_preds(_RAW_OP)
    assert '%0 = irdl.base "!builtin.tensor"' in out


def test_an_unrecognised_predicate_is_reported_and_left_in_place():
    """Fail closed. Dropping an unknown predicate silently would loosen the contract with no trace;
    keeping it silently would leave a constraint that cannot fire. It is kept AND reported."""
    out, notes, unknown = lower_c_preds(_RAW_OP)
    assert unknown and "SOMETHING THE TABLE HAS NEVER SEEN" in unknown[0]
    assert "SOMETHING THE TABLE HAS NEVER SEEN" in out
    assert "UNRECOGNISED" in _header(notes, unknown)


def test_an_inexpressible_predicate_leaves_the_all_of_and_enters_the_header():
    raw = _RAW_OP.replace(
        '"SOMETHING THE TABLE HAS NEVER SEEN"',
        '"[](::mlir::Type elementType) { return !((::llvm::isa<::mlir::TokenType>(elementType))); }'
        '(::llvm::cast<::mlir::ShapedType>($_self).getElementType())"')
    out, notes, unknown = lower_c_preds(raw)
    assert not unknown
    assert "irdl.c_pred" not in out
    assert "%3 = irdl.all_of(%1)" in out          # the dropped arg is gone, the kept one remains
    assert notes and notes[0][0] == "o"
    assert "must not be a token" in _header(notes, unknown)


def test_a_slot_naming_only_a_dropped_constraint_keeps_a_named_any(tmp_path):
    """A dropped constraint referenced straight from `irdl.results(...)` cannot just vanish -- the
    slot would dangle and the file would not parse. It becomes an explicit `irdl.any`."""
    raw = '''module {
  irdl.dialect @d {
    irdl.operation @o {
      %0 = irdl.c_pred "[](::mlir::Type elementType) { return !((::llvm::isa<::mlir::TokenType>(elementType))); }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())"
      irdl.results(result: %0)
    }
  }
}
'''
    out, _, _ = lower_c_preds(raw)
    assert "%0 = irdl.any" in out
    f = tmp_path / "d.irdl.mlir"
    f.write_text(out)
    m = tmp_path / "m.mlir"
    m.write_text('module {\n  %0 = "d.o"() : () -> i32\n}\n')
    assert subprocess.run([str(_MLIROPT), f"--irdl-file={f}", str(m), "-o", "/dev/null"],
                          capture_output=True, text=True).returncode == 0


# --------------------------------------------------------------------------------------------
# The corpus, not the fixtures.
# --------------------------------------------------------------------------------------------

def test_every_capsule_the_contract_declares_ops_for_parses_and_verifies():
    """The whole point of the contract: it must hold against the shipped corpus.

    Before the generic-form bridge existed this was 0 of 370 -- an IRDL-registered dialect has no
    custom parser, and the failure was rc=1 with an EMPTY stderr, so it read as nothing at all. The
    expected count is DERIVED (capsules whose mnemonics the IRDL actually declares), not pinned to a
    number, so the corpus can grow without editing this test.
    """
    ok, n, fails, _ = verify(_IRDL)
    caps = [c for c in sorted(_CAPSULES.rglob("capsule.interface.mlir"))
            if "merlin_iface." in c.read_text()]
    declared = {line.strip().split("@", 1)[1].split()[0].rstrip("{").strip()
                for line in _IRDL.read_text().splitlines()
                if line.strip().startswith("irdl.operation @")}
    in_scope = [c for c in caps if set(op_mnemonics(c.read_text())) <= declared]
    assert in_scope, "no capsule uses only ops the contract declares -- the check is vacuous"
    assert ok == len(in_scope), f"{len(in_scope) - ok} in-scope capsule(s) failed: {fails[:5]}"
    # Everything that did fail failed for the ONE known reason: the reference ODS declares 7 ops
    # while the interface grammar defines more (rmsnorm, attention_qk, bias_add, matmul_batched,
    # rope). That is a divergence between the two halves of the contract, not a capsule defect --
    # but it must stay visible, so any OTHER failure reason breaks this test.
    assert all("unregistered operation" in f for f in fails), fails[:5]
