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
from merlin.targetgen.rtl.gen_iface_irdl import _restore_type_params, _strip_type_sigil

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
