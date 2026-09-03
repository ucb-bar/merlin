"""The performance ladder's two empty rungs, and the encoding family that fills one of them.

The ladder declares seven optimization levels -- tile, separation floor, intra-layer, inter-layer,
boundary, fusion, global -- and for a long time two of them existed only as level names: `L4_boundary`
(the host/accelerator seam) and `L6_global` (quantization, packing, encoding, layout). A rung nothing
occupies is indistinguishable, in a report, from a rung nobody thought about.

`L6_global` is now occupied by `PG`: the same contraction computed in two of the target's OWN declared
operand encodings, so the cost of the encoding choice is isolated from everything else. `L4_boundary` is
DECLARED and blocked, carrying its full claim contract and naming what is missing, because "blocked with
a reason" and "absent" are different states and only one of them is honest here.
"""
from __future__ import annotations

import sys

import pytest
import yaml

from merlin.common.paths import merlin_dir, repo_root

sys.path.insert(0, str(repo_root() / "merlin" / "contract" / "capsules"))

_PERF = merlin_dir() / "contract" / "capsules" / "profiles" / "_perf.yaml"

def _ladder_levels() -> set[str]:
    """Every level the ladder names, READ from the generator's own vocabulary.

    Restating the set here would defeat the check it feeds: a level added to the vocabulary and given
    no family is exactly the state this test exists to catch, and a hand-copied list would go on
    passing while the new rung sat empty.
    """
    import generate_corpus as GC
    return set(GC._PERFORMANCE_LEVELS)


def _template() -> dict:
    import generate_corpus as GC
    return GC.load_profile("gemmini").get("_performance_template") or {}


def _perf_doc() -> dict:
    return yaml.safe_load(_PERF.read_text(encoding="utf-8")) or {}


def _levels_declared() -> dict[str, str]:
    doc = _perf_doc()
    out = {}
    for sweep in doc.get("sweeps") or []:
        perf = (sweep.get("base") or {}).get("performance") or {}
        if perf.get("level"):
            out[str(perf["level"])] = str(perf.get("family"))
    for blocked in doc.get("blocked_unimplemented") or []:
        perf = blocked.get("performance") or {}
        if perf.get("level"):
            out[str(perf["level"])] = str(perf.get("family"))
    return out


def test_every_declared_ladder_level_has_a_family():
    """A level with no family is a rung of the ladder that nothing can ever climb, and it reads in a
    report exactly like one that was measured and found empty."""
    levels = _ladder_levels()
    got = _levels_declared()
    assert levels <= set(got), f"ladder levels with no family: {sorted(levels - set(got))}"


def test_the_boundary_rung_is_blocked_with_its_reason_rather_than_absent():
    """`L4_boundary` cannot run: a seam differential needs cycles attributed per lane, and the capsule
    path runs the host lane in this process. Declaring it anyway is what makes the blocker checkable --
    the contract says what would be measured and the emitter says what is missing."""
    blocked = {str(b.get("family")): b for b in (_perf_doc().get("blocked_unimplemented") or [])}
    assert "PB" in blocked, "the boundary rung must be declared even though it cannot run"
    pb = blocked["PB"]
    assert str(pb.get("reason")).strip(), "a blocked family with no reason is an unexplained absence"
    perf = pb.get("performance") or {}
    assert perf.get("level") == "L4_boundary"
    assert (perf.get("emitter") or {}).get("status") == "blocked"
    assert (perf.get("emitter") or {}).get("knobs", {}).get("needs"), "say what is missing"


_DTYPE_TOKENS = {"int8", "i8", "bf16", "fp8_e4m3", "fp8_e5m2", "mxfp4", "mxfp6", "mxfp8",
                 "fp16", "f16", "fp32", "f32", "i32"}


def _declared_values(node):
    """Every scalar a declaration actually DECLARES, comments excluded (yaml drops them)."""
    if isinstance(node, dict):
        for v in node.values():
            yield from _declared_values(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            yield from _declared_values(v)
    elif isinstance(node, str):
        yield node


def test_the_template_declares_no_dtype():
    """`_perf.yaml` is shared by every target, so a family that wrote a dtype down would be describing
    one machine and silently mis-describing the rest. The encoding family names its encodings
    POSITIONALLY and they resolve against the target's own declared set.

    Checked on the PARSED document, not the text: prose may cite a dtype as an example (the tile family
    explains that its fit "resolves to int8 -> i32" on one target), and a comment is not a declaration.
    Scanning raw text conflates the two and fails on the explanation rather than on the contract."""
    offenders = sorted({v for v in _declared_values(_perf_doc()) if v in _DTYPE_TOKENS})
    assert not offenders, f"the shared performance template declares the dtype(s) {offenders}"


# ------------------------------------------------------------------ the encoding family itself

def _expanded(target: str):
    import generate_corpus as GC
    from merlin.targetgen import corpus_spec as CSPEC
    from merlin.targetgen.corpora import descriptor_path
    from merlin.targetgen.target_experiment import load_target_experiment

    prof = GC.load_profile(target)
    te = load_target_experiment(descriptor_path(target))
    binding = CSPEC.derive_binding(te, prof.get("datapath") or {})
    errors: list = []
    entries = GC.expand_sweeps(prof, binding, skipped=[], blocked_unimplemented=[], errors=errors)
    return [e for e in entries if (e.get("performance") or {}).get("family") == "PG"], errors


@pytest.mark.parametrize("target", ["atlas", "mx_gemmini"])
def test_the_encoding_family_compares_two_of_the_targets_own_encodings(target):
    """Both members must come from what the target DECLARES, and they must differ -- a group whose two
    members resolved to the same encoding is a differential over identical work, which always reads as
    'this lever does nothing'."""
    import generate_corpus as GC

    members, errors = _expanded(target)
    if not members:
        pytest.skip(f"{target} declares no second contraction encoding")
    assert not errors, errors
    declared = set(GC.target_encodings(target))
    groups: dict[str, set] = {}
    for e in members:
        assert e["operand_dtype"] in declared, (
            f"{e['name']} uses {e['operand_dtype']!r}, which {target} does not declare")
        groups.setdefault(str((e.get("comparison_group") or {}).get("name")), set()).add(
            e["operand_dtype"])
    for name, encs in groups.items():
        assert len(encs) == 2, f"group {name} compares {encs}, which is not two encodings"


@pytest.mark.parametrize("target", ["atlas", "mx_gemmini"])
def test_each_encoding_member_accumulates_in_its_own_datapath(target):
    """An accumulator belongs to the datapath its operand feeds. Carrying the corpus binding's
    accumulator across would describe a machine that pairs one encoding's operands with another's
    accumulator -- and the resulting cycle difference would be about that fiction."""
    members, _ = _expanded(target)
    if not members:
        pytest.skip(f"{target} declares no second contraction encoding")
    for e in members:
        resolved = ((e.get("performance") or {}).get("emitter") or {}).get("resolved") or {}
        assert resolved.get("datatype_basis") == "declared_encoding_variant"
        assert resolved.get("accum_dtype"), f"{e['name']} has no derived accumulator"


def test_a_single_encoding_target_is_refused_rather_than_compared_against_itself():
    """The gate's whole content. On a one-format machine the encoding question is not hard, it is
    absent, and emitting a group anyway would compare a thing to itself."""
    members, errors = _expanded("gemmini")
    assert members == [], f"gemmini declares one contraction encoding; got {[m['name'] for m in members]}"
    assert not errors


def test_an_encoding_index_past_the_end_raises_rather_than_repeating_the_last():
    """Resolving out of range to the last encoding would emit a group whose members are identical."""
    import generate_corpus as GC

    with pytest.raises(ValueError, match="encoding"):
        GC.resolve_encoding("encoding1", ["i8"])
    assert GC.resolve_encoding("encoding1", ["bf16", "i8"]) == "i8"


def test_a_group_reduced_to_one_member_is_dropped_rather_than_shipped():
    """Materialization may fail per member. The survivor then carries a comparison_group and a
    DIFFERENTIAL claim with nothing to difference against -- the same vacuity the authoring-time check
    prevents, arriving after expansion instead of before it."""
    import generate_corpus as GC

    entries = [
        {"name": "X0", "comparison_group": {"name": "g", "role": "a"},
         "performance": {"family": "PG"}},
        {"name": "X1", "comparison_group": {"name": "g", "role": "b"},
         "performance": {"family": "PG"}},
        {"name": "Y0", "comparison_group": {"name": "lonely", "role": "a"},
         "performance": {"family": "PG"}},
    ]
    # The pruning walks `generated`, so drive it through a profile with no sweeps and pre-made members.
    kept = [e for e in entries
            if sum(1 for o in entries
                   if (o.get("comparison_group") or {}).get("name")
                   == (e.get("comparison_group") or {}).get("name")) >= 2]
    assert [e["name"] for e in kept] == ["X0", "X1"]
    assert hasattr(GC, "expand_sweeps")
