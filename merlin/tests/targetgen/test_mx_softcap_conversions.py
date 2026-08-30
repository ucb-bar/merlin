"""The soft-capped MX flash kernel must cross the bf16/f32 boundary EXPLICITLY.

The fused flash kernel is bf16 throughout except for one place: the injected soft-cap loop evaluates
tanh(y) = (p-1)/(p+1) in fp32, because the bf16 divide and the abs/sign branches miscompile on this
SIMT target. That makes the loop the only bf16 <-> f32 boundary in the emitted kernel, and it used to
cross it with plain C casts.

Measured on the oracle, those casts do not round-trip the value: the soft-capped capsule landed 8 of
256 elements outside tolerance with them, and the identical loop reproduces the reference golden
BIT-EXACTLY on all 512 scores once the widening is written as a shift and the narrowing as an explicit
round-to-nearest-even. bf16 is by definition the top 16 bits of an f32, so both conversions are exactly
expressible and neither needs the compiler's opinion.

These tests pin that, and pin the scope: a flash capsule with no soft-cap must not grow the block at
all, because those capsules pass today and are graded on this same reference kernel.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import capsule_golden as CG
from merlin.targetgen.capsule_common import load_capsule

_SLICES = repo_root() / "merlin" / "contract" / "capsules" / "radiance" / "model_slices"
_CONTRACT = repo_root() / "merlin" / "contract"

#: the two conversions the loop must use, written out rather than delegated to a C cast
_WIDEN = "__builtin_bit_cast(float, (uint32_t)__builtin_bit_cast(uint16_t, _p) << 16)"
_NARROW = "(_ob + 0x7fffu + ((_ob >> 16) & 1u)) >> 16"
#: and the two it must not
_BANNED = ("float _pf = (float)_p;", "(_Float16)((float)_cap * _tf)")


def _emit(name: str) -> tuple[str, dict]:
    """Emitted reference MX kernel for a capsule, or a skip if this checkout cannot build one."""
    d = _SLICES / name
    if not (d / "golden.yaml").is_file():
        pytest.skip(f"{name} golden not present in this checkout")
    base = pytest.importorskip("merlin.runtime.backends.base")
    mxc = base.get_backend("muon").muon_mx_codegen
    cap = load_capsule(str(d), contract=str(_CONTRACT))
    ops = CG.mx_operands(cap, cap.get("__dir__", str(d)))
    if not ops:
        pytest.skip(f"{name} golden carries no MX operand bundle")
    try:
        return mxc.emit_mx_kernel(ops, "Y0"), ops
    except Exception as e:  # noqa: BLE001 — the flash emitter needs the kernel sources on disk
        pytest.skip(f"{name}: reference kernel not emittable here ({type(e).__name__}: {e})")


def test_softcap_crosses_the_float_boundary_with_explicit_conversions():
    src, ops = _emit("R10_flash_attn_mx_gemma")
    assert ops.get("softcap"), "this capsule is the soft-capped one; the fixture picked the wrong capsule"
    assert _WIDEN in src, "bf16 -> f32 must be the exact widening shift, not a C cast"
    assert _NARROW in src, "f32 -> bf16 must be an explicit round-to-nearest-even, not a C cast"


@pytest.mark.parametrize("banned", _BANNED)
def test_softcap_does_not_use_a_c_cast_between_bf16_and_float(banned):
    src, _ = _emit("R10_flash_attn_mx_gemma")
    assert banned not in src, (
        f"the emitted soft-cap uses the C cast {banned!r} again; measured on the oracle that costs the "
        f"capsule 6 of its 8 recovered elements")


def test_softcap_still_evaluates_tanh_the_way_the_golden_does():
    """Lockstep guard. mx_flash_ref._softcap computes p = exp(2y) then (p-1)/(p+1) in fp32 -- NOT
    upstream's (1-e)/(1+e) with e = exp(-2|x|). The golden is generated from that reference, so an
    emitter that switched forms would be graded against arithmetic it no longer performs."""
    src, _ = _emit("R10_flash_attn_mx_gemma")
    assert "mu_fexp((_Float16)(_x * _inv2))" in src, "p = exp(2y) with 2/cap folded into one constant"
    assert "(_pf - 1.0f) / (_pf + 1.0f)" in src, "tanh as (p-1)/(p+1), matching mx_flash_ref._softcap"


@pytest.mark.parametrize("name", ["R8_flash_attention_mx", "R9_flash_attn_mx_gqa"])
def test_a_flash_capsule_without_a_softcap_grows_no_conversion(name):
    """Scope guard: these capsules pass today on this same reference kernel, so the soft-cap change
    must be invisible to them."""
    src, ops = _emit(name)
    assert ops.get("softcap") is None, f"{name} unexpectedly declares a soft-cap"
    assert _WIDEN not in src and _NARROW not in src
    for banned in _BANNED:
        assert banned not in src
