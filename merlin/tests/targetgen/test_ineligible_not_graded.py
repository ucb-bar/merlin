"""A capsule the target cannot do must not be graded — and must not make all_pass unreachable.

Measured: an int8 target scored 22/22 on everything its contract declares at round 00, then ran all 20
rounds because 15 out-of-scope capsules kept all_pass false. Since the loop's only early exit is a
genuine all_pass, unpassable capsules turn every run into a fixed-price purchase of its full budget.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen.capsule_runner import _split_ineligible


def _capsule(name: str) -> dict:
    base = pathlib.Path(repo_root()) / "merlin/contract/capsules"
    for d in base.glob(f"*/{name}"):
        if d.is_dir():
            return yaml.safe_load((d / "capsule.yaml").read_text())
    pytest.skip(f"{name} not on disk")


def test_a_dtype_the_target_cannot_do_is_withheld():
    caps = [_capsule("GF4_add_bf16_pt"), _capsule("A2_single_tile_matmul")]
    keep, withheld = _split_ineligible(caps, "gemmini")
    assert [c["name"] for c in keep] == ["A2_single_tile_matmul"]
    assert len(withheld) == 1
    w = withheld[0]
    assert w["status"] == "not_graded" and w["ineligible"] is True
    assert "bf16" in w["failure"]["detail"]
    assert w["failure"]["plane"] == "capability"


def test_nothing_is_withheld_when_the_target_is_unresolvable():
    caps = [_capsule("A2_single_tile_matmul")]
    keep, withheld = _split_ineligible(caps, "no_such_target_xyz")
    assert len(keep) == 1 and withheld == []


def test_fails_open_on_an_unparseable_capsule():
    keep, withheld = _split_ineligible([{"name": "junk"}], "gemmini")
    assert len(keep) == 1 and withheld == []      # graded, never silently withheld


def test_withholding_makes_all_pass_reachable():
    """The arithmetic that matters: with the unpassable capsules counted, all_pass is impossible."""
    results = ([{"capsule": f"ok{i}", "status": "pass"} for i in range(22)]
               + [{"capsule": f"bad{i}", "status": "not_graded"} for i in range(15)])
    graded = [r for r in results if r["status"] != "not_graded"]
    n_pass = sum(1 for r in graded if r["status"] == "pass")
    assert (n_pass, len(graded)) == (22, 22)
    assert n_pass == len(graded)                  # all_pass TRUE -> the loop can exit at round 00
    assert n_pass != len(results)                 # counting them: 22 != 37, all_pass forever false


def test_only_a_hard_structural_fact_withholds_not_an_undeclared_family():
    """C7_attention_qk_i8 is the target's NATIVE dtype and legal rank; only its family is undeclared.
    Families compose (attention = contraction + a transposing movement), so it must still be graded --
    it is the one genuinely reachable failure on that target."""
    keep, withheld = _split_ineligible([_capsule("C7_attention_qk_i8")], "gemmini")
    assert [c["name"] for c in keep] == ["C7_attention_qk_i8"]
    assert withheld == []


def test_dtype_comparison_is_alias_aware():
    """The contract spells the format 'int8'; a capsule region reports 'i8'. A raw string compare reads a
    native-dtype capsule as having no datapath and withholds it for a spelling difference."""
    from merlin.targetgen import eligibility as el
    assert el._dtype_ok("i8", ("int8",)) is True
    assert el._dtype_ok("bf16", ("int8",)) is False


def test_a_rank_outside_every_declared_capability_is_withheld():
    keep, withheld = _split_ineligible([_capsule("AF12_gemv_batched_bf16_pt")], "atlas")
    assert keep == [] and len(withheld) == 1
    assert "rank" in withheld[0]["failure"]["detail"]


def test_the_measured_corpora_split_as_expected():
    """gemmini withholds exactly its 12 bf16 capsules (no bf16 datapath anywhere in its contract);
    atlas, which DOES declare bf16, withholds only the rank-3 capsule."""
    import pathlib as _p, yaml as _y
    from merlin.targetgen.target_experiment import load_target_experiment
    from merlin.common.paths import repo_root as _rr
    for target, expect in (("gemmini", 12), ("atlas", 1)):
        te = load_target_experiment(
            str(_p.Path(_rr()) / f"merlin/experiments/capsule_bench/targets/{target}/target_experiment.yaml"))
        caps = []
        for r in [_p.Path(te.capsule_corpus)] + [_p.Path(s) for s in te.corpus_siblings()]:
            if not r.is_dir():
                continue
            try:
                subs = sorted(r.iterdir())
            except PermissionError:
                continue
            for d in subs:
                f = d / "capsule.yaml"
                if f.is_file():
                    caps.append(_y.safe_load(f.read_text()))
        if not caps:
            pytest.skip(f"{target} corpus unavailable")
        op = [c for c in caps if c.get("kind") != "model"]
        _, withheld = _split_ineligible(op, target)
        assert len(withheld) == expect, [w["capsule"] for w in withheld]
