"""An MX golden must REFUSE a reduction extent with a partial block-scale group, not zero its tail.

The MX references assign one E8M0 scale per whole group of ``mx.GROUP`` (32) elements and compute the
group count with floor division. Every array involved is zero-initialised, so a K with a remainder used
to come back with the tail silently zeroed -- and the golden then did not depend on part of its own
input. MEASURED before the guard:

    K=33  covered 32 of 33 columns, tail zeroed
    K=48  covered 32 of 48 columns -- a THIRD of the reduction -- tail zeroed
    perturbing A[0,32] with K=33 left the requant result bit-identical

A golden that ignores part of its input is worse than a missing golden: it certifies a backend that
ignores the same tail and fails one that does not. No capsule on disk trips this (every MX K is 32 or
64), so the guard is inert today -- and it is the reason MX coverage is aligned-only, because a
non-aligned MX capsule cannot be minted at all. Supporting one means scaling a partial final group in the
reference, never relaxing these checks.
"""
from __future__ import annotations

import sys

import pytest

from merlin.common.paths import repo_root

# the golden generator lives beside the corpus, not in the package
_CAPS = str(repo_root() / "merlin" / "contract" / "capsules")
if _CAPS not in sys.path:
    sys.path.insert(0, _CAPS)


@pytest.fixture(scope="module")
def gen():
    return pytest.importorskip("generate_corpus")


@pytest.fixture(scope="module")
def mx(gen):
    try:
        return gen._mx_ref()
    except Exception as e:                                  # noqa: BLE001 — mlc not present in this env
        pytest.skip(f"MX reference unavailable: {type(e).__name__}: {e}")


def test_the_group_size_is_derived_not_a_literal(mx):
    """The reference exposes the group; a test that retyped 32 would pass while the fact changed."""
    assert getattr(mx, "GROUP", None), "the MX reference no longer exposes GROUP"
    assert int(mx.GROUP) > 1


@pytest.mark.parametrize("K", [33, 48, 63])
def test_requant_refuses_a_partial_group(gen, mx, K):
    """K=48 is the dangerous case: a multiple of the 16 tile edge, so it looks aligned to the cert cover
    while dropping 16 of its 48 reduction elements."""
    np = pytest.importorskip("numpy")
    pal = gen._mx_safe_palette(mx, "fp8_e4m3")
    P = np.arange(1.0, 4 * K + 1.0).reshape(4, K)
    with pytest.raises(ValueError) as ei:
        gen._mx_requant_blocks(P, pal, group=mx.GROUP)
    msg = str(ei.value)
    assert str(mx.GROUP) in msg, "the refusal does not name the group size"
    assert str(K) in msg, "the refusal does not name the offending extent"
    assert "multiple" in msg.lower(), "the refusal does not state the requirement"


@pytest.mark.parametrize("K", [32, 64])
def test_requant_still_accepts_whole_groups(gen, mx, K):
    """The guard must be INERT for the corpus on disk — every MX capsule there has K of 32 or 64."""
    np = pytest.importorskip("numpy")
    pal = gen._mx_safe_palette(mx, "fp8_e4m3")
    P = np.arange(1.0, 4 * K + 1.0).reshape(4, K)
    out = gen._mx_requant_blocks(P, pal, group=mx.GROUP)
    assert out is not None


def test_flash_reference_refuses_a_partial_key_block(mx):
    """The flash path has its own floor division over the key length, with the same consequence."""
    np = pytest.importorskip("numpy")
    from merlin.targetgen import mx_flash_ref as MFR

    Skv = int(mx.GROUP) + 1
    with pytest.raises(ValueError) as ei:
        MFR.flash_attention_fp8(mx, np.zeros((4, Skv)), np.zeros((Skv, 16)),
                                np.zeros((max(1, Skv // int(mx.GROUP)), 16), dtype=np.uint8),
                                M=4, Skv=Skv, Dv=16, att_scale=1.0)
    assert str(mx.GROUP) in str(ei.value)


def test_every_mx_capsule_on_disk_has_a_whole_group_K():
    """The corpus itself must satisfy the guard, or regeneration would start failing.

    Reads K structurally: for a matmul/linear the weight is ``[K, N]``, so K is its first axis. Ops with
    a different operand grammar (attention_mx, gemv_batched, model) are not asserted here — they carry
    their reduction extent elsewhere and the flash guard covers the one that matters.
    """
    import yaml

    caps = repo_root() / "merlin" / "contract" / "capsules"
    bad = []
    for p in caps.rglob("capsule.yaml"):
        try:
            c = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:                                   # noqa: BLE001
            continue
        ins = {i.get("name"): (i.get("shape") or []) for i in (c.get("inputs") or [])}
        # BLOCK-SCALED IS A PROPERTY OF THE CAPSULE, not of a dtype token, and the capsule declares it:
        # a block-scaled datapath carries one E8M0 exponent per whole K group, so the capsule declares
        # `role: scale` operands and a non-scaled one does not. Keying on a dtype list instead swept in
        # `fp8_e4m3`, which is the canonical name of the block-scaled `mxfp8` AND the ordinary fp8 of a
        # target whose refmodel has no groups at all -- so an fp8 capsule with a perfectly legal ragged
        # K was reported as carrying a partial block-scale group it does not have.
        if not any(str(i.get("role")) == "scale" for i in (c.get("inputs") or [])):
            continue
        if (c.get("operation") or {}).get("op") not in ("matmul", "linear", "resident_reuse"):
            continue
        w = ((c.get("operation") or {}).get("attributes") or {}).get("weight")
        shape = ins.get(w) or []
        if len(shape) == 2 and int(shape[0]) % 32:
            bad.append(f"{c.get('name')}: K={shape[0]}")
    assert not bad, f"MX capsule(s) on disk carry a partial block-scale group: {bad}"
