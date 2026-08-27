"""The cross-backend capsules must be GRADED and accelerator-INELIGIBLE at the same time.

Those two properties together are the whole forcing function, and each is easy to lose:

  * graded but accelerator-ELIGIBLE  -> the SIMT cluster serves it and nothing is forced;
  * accelerator-ineligible but WITHHELD -> `_split_ineligible` drops it and the capsule measures nothing.

Both failure modes are silent — the suite still reports a number — so they are pinned here rather than
left to be noticed. The dtype is derived from the target's own declaration, not chosen for effect:
radiance declares `movement` over fp32/fp16/bf16 only, while int8 appears in its contraction formats,
which is exactly the gap that makes an int8 movement graded-but-unservable.
"""
from __future__ import annotations

import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.targetgen import coverage_report as CR, eligibility as EL
from merlin.targetgen.capsule_runner import _split_ineligible

PROFILE = merlin_dir() / "contract/capsules/profiles/radiance.yaml"


def _entries():
    doc = yaml.safe_load(PROFILE.read_text())
    for v in doc.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "name" in v[0]:
            return v
    return []


def _rx():
    return [e for e in _entries() if str(e.get("name", "")).startswith("RX")]


def test_the_profile_declares_two_cross_backend_capsules():
    rx = _rx()
    assert len(rx) == 2, [e["name"] for e in rx]
    assert {e["operand_dtype"] for e in rx} == {"int8"}
    # one scalar-shaped, one vector-shaped: same missing datapath, different reuse target
    areas = sorted(e["M"] * e["N"] for e in rx)
    assert areas[0] * 4 <= areas[1], f"the two shapes are not meaningfully different: {areas}"


def test_int8_is_declared_somewhere_so_they_are_not_hard_withheld():
    """If int8 vanished from every capability, these would flip to WITHHELD and silently stop testing
    anything — the capsules would still be listed and the suite would still report a number."""
    cmap = EL.capability_map_for_target("radiance")
    all_dtypes = {x for cap in cmap.values() for x in (getattr(cap, "dtypes", ()) or ())}
    assert any(EL._dtype_ok("i8", tuple(all_dtypes)) for _ in (0,)), all_dtypes


def test_movement_is_declared_float_only_so_there_is_no_simt_datapath():
    """The other half of the forcing function. If radiance ever declares integer movement, these
    capsules stop forcing anything and should be re-thought rather than left in place."""
    cmap = EL.capability_map_for_target("radiance")
    mv = cmap.get("movement")
    if mv is None:
        pytest.skip("radiance declares no movement capability at all")
    dts = {str(d).lower() for d in (getattr(mv, "dtypes", ()) or ())}
    assert dts and not any(d.startswith("i") or d.startswith("int") for d in dts), dts


def test_a_minted_cross_backend_capsule_is_graded_not_withheld():
    """The end-to-end property, on capsules built from the real profile."""
    import sys

    sys.path.insert(0, str(merlin_dir() / "contract/capsules"))
    sys.path.insert(0, str(merlin_dir() / "python"))
    try:
        import generate_corpus as GC

        from merlin.targetgen import corpus_spec as CS
        from merlin.targetgen.target_experiment import load_target_experiment
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"corpus generator unavailable here: {type(e).__name__}: {e}")

    import tempfile
    from pathlib import Path

    te = load_target_experiment(
        merlin_dir() / "experiments/capsule_bench/targets/radiance/target_experiment.yaml")
    profile = yaml.safe_load(PROFILE.read_text())
    binding = CS.derive_binding(te, profile.get("datapath", {}))
    out = Path(tempfile.mkdtemp(prefix="rx_caps_"))
    caps = []
    for e in _rx():
        try:
            d = GC._write_capsule_inner(e, binding, out)
        except Exception as ex:  # noqa: BLE001
            pytest.skip(f"cannot mint {e['name']} here: {type(ex).__name__}: {ex}")
        caps.append(yaml.safe_load((Path(d) / "capsule.yaml").read_text()))

    cmap = EL.capability_map_for_target("radiance")
    for c in caps:
        v = EL.is_eligible(CR._capsule_region(c), cmap)
        assert getattr(v, "eligible", True) is False, (
            f"{c['name']} IS accelerator-eligible — it forces nothing")
    keep, withheld = _split_ineligible(caps, "radiance")
    assert len(keep) == len(caps) and not withheld, (
        f"cross-backend capsules must be GRADED; withheld={[w['capsule'] for w in withheld]}")
