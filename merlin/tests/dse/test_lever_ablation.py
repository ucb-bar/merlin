"""The leave-one-out lever ablation: what it measures, and everything it refuses to measure.

Every test here defends one property the tool exists to hold. The load-bearing ones are the
negative ones, because the failures they prevent all LOOK like results:

  * a contribution taken from two ``ours_ns`` values across two sessions -- the mistake that
    produced a 1.61x weight-transpose "result" whose two ExecuTorch arms were 13% apart. The
    attribution must be a RATIO OF RATIOS, with ExecuTorch as the anchor that absorbs board drift;
  * a contribution inside the K1's 2.6% noise band presented as a finding rather than as an absence
    of one;
  * a refused cell folded in as a 0.0 contribution, which reads as "measured, no effect";
  * a pair whose two cells were built from different compiler sources -- routine on this shared
    tree, where other sessions commit mid-run -- differenced anyway.
"""
from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from merlin.common.paths import repo_root

DRIVER = repo_root() / "build_tools" / "scripts" / "k1_lever_ablation.py"
INSTRUMENT = repo_root() / "build_tools" / "scripts" / "k1_int8_fair_compare.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("k1_lever_ablation", DRIVER)
    m = importlib.util.module_from_spec(spec)
    # Registered before exec: the module defines a dataclass, and `dataclasses` resolves string
    # annotations through sys.modules[cls.__module__] -- absent, the decorator raises at import.
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _row(model, cell_id, *, ours=None, et=None, digest="D0", dropped=None, status=None,
         refusal="", dirty=None):
    """A ledger row shaped exactly like the one the driver writes."""
    return {"key": f"{model}::{cell_id}", "model": model, "cell_id": cell_id,
            "dropped": dropped, "features": [], "feature_arg": "",
            "status": status or ("measured" if (ours and et) else "refused"),
            "refusal": refusal, "ours_ns": ours, "executorch_warm_ns": et,
            "source_digest": digest, "source_dirty": list(dirty or [])}


# --- the cell list -----------------------------------------------------------------------------

def test_cell_list_is_the_full_set_plus_one_leave_one_out_per_lever(mod):
    """N levers means N+1 cells: the full set, and each lever removed exactly once. A cell list that
    quietly dropped or duplicated an arm would produce an attribution table with a lever nobody
    measured."""
    feats = ["a", "b", "c"]
    cells = mod.plan_cells(feats)
    assert len(cells) == len(feats) + 1

    full = cells[0]
    assert full.cell_id == mod.FULL_CELL
    assert full.dropped is None
    assert list(full.features) == feats
    assert full.feature_arg == "a,b,c"

    drops = cells[1:]
    assert [c.cell_id for c in drops] == ["drop_a", "drop_b", "drop_c"]
    assert [c.dropped for c in drops] == feats
    # Each leave-one-out cell carries EVERY OTHER lever, and the dropped one is really gone.
    assert [c.feature_arg for c in drops] == ["b,c", "a,c", "a,b"]
    for c in drops:
        assert c.dropped not in c.features
        assert len(c.features) == len(feats) - 1


def test_single_lever_leave_one_out_passes_an_explicitly_empty_feature_string(mod):
    """With one lever the control arm has NO features, and that must be passed as an explicit empty
    ``--features ''``. The instrument reads an OMITTED --features as "the package's own certified
    feature list" -- a different, larger set -- so omitting it would silently measure the wrong
    control and the lever's contribution would be against the wrong baseline."""
    cells = mod.plan_cells(["only_one"])
    assert [c.cell_id for c in cells] == ["full", "drop_only_one"]
    assert cells[1].features == ()
    assert cells[1].feature_arg == ""


def test_dry_run_command_carries_the_cells_feature_set_and_the_timeouts(mod):
    """The command printed by --dry-run is the command that will run. It must name the cell's own
    feature set (not the full one) and carry the compile ceiling through -- the 900s module default
    is a kernel budget and a whole-model build exceeds it, which surfaces as a BLOCKED cell
    indistinguishable from a codegen defect."""
    class _P:
        model = "m"
        ours_bundle_root = repo_root() / "bundle"
    class _A:
        package = "out/artifacts/targets/rvv/hand_v0_int8"
        n, warmup, iters, et_n_lo, et_n_hi = 3, 2, 5, 1, 6
        compile_timeout_s = 7000
    cell = mod.plan_cells(["a", "b"])[1]          # drop_a -> features "b"
    cmd = mod.instrument_command(_P(), cell, _A(), repo_root() / "out.json")
    assert str(INSTRUMENT) in cmd
    assert cmd[cmd.index("--features") + 1] == "b"
    assert cmd[cmd.index("--compile-timeout-s") + 1] == "7000"
    assert cmd[cmd.index("--model") + 1] == "m"


# --- the ratio of ratios -----------------------------------------------------------------------

def test_contribution_is_a_ratio_of_ratios_not_an_ours_ns_delta(mod):
    """THE test. Two cells whose ExecuTorch anchors differ by 30% -- ordinary board drift between
    two sessions. A naive ours_ns comparison and the anchored ratio-of-ratios do not merely differ
    in magnitude here, they disagree on the SIGN, which is exactly how a 1.61x weight-transpose
    "result" was once read off two sessions whose ET arms were 13% apart.

      full : ours=100, ET=100  -> ratio 1.000  (we are at parity with ET)
      drop : ours=113, ET=130  -> ratio 0.869  (we are 13% BETTER than ET without the lever)

    Naive: dropping the lever costs 13% -> "the lever is worth +13%".
    Anchored: the ratio got BETTER without the lever -> the lever COSTS ~13%.
    """
    full = _row("m", "full", ours=100.0, et=100.0)
    drop = _row("m", "drop_L", ours=113.0, et=130.0, dropped="L")
    c = mod.contribution(full, drop)

    assert c["method"].startswith("ratio_of_ratios")
    assert c["ratio_with"] == pytest.approx(1.0)
    assert c["ratio_without"] == pytest.approx(113.0 / 130.0)
    assert c["contribution"] == pytest.approx(113.0 / 130.0 - 1.0)     # ~ -0.1308

    naive = 113.0 / 100.0 - 1.0                                        # +0.13, the wrong answer
    assert c["contribution"] < 0 < naive                               # opposite signs
    assert abs(c["contribution"] - naive) > 0.25                       # materially different
    assert c["status"] == "hurts"


def test_a_cell_without_an_executorch_anchor_refuses_instead_of_using_ours_ns(mod):
    """A cell whose ExecuTorch arm produced no warm slope has no anchor. The bare ours_ns from it is
    NOT a comparand across sessions, so the pair refuses -- there is deliberately no fallback path
    from ours_ns alone to a number."""
    full = _row("m", "full", ours=100.0, et=100.0)
    drop = _row("m", "drop_L", ours=113.0, et=None, dropped="L", status="measured")

    ratio, why = mod.cell_ratio(drop)
    assert ratio is None
    assert "anchor" in why

    c = mod.contribution(full, drop)
    assert c["contribution"] is None
    assert c["status"] == "refused"
    # The refusal must SAY that ours_ns alone was not enough, not hide the wall it did have.
    assert "113" in c["reason"] or "ours_ns" in c["reason"]


# --- the noise band ----------------------------------------------------------------------------

def test_a_contribution_inside_the_k1_noise_band_is_labelled_within_noise(mod):
    """The K1's measured band is 2.6%. Anything inside it is not a result; the tool must say so
    rather than print a small number that a reader will cite."""
    assert mod.NOISE_BAND == 0.026
    full = _row("m", "full", ours=100.0, et=100.0)
    drop = _row("m", "drop_L", ours=101.5, et=100.0, dropped="L")     # +1.5%, inside the band
    c = mod.contribution(full, drop)
    assert c["contribution"] == pytest.approx(0.015)
    assert c["within_noise"] is True
    assert c["status"] == "within_noise"
    assert "noise" in c["reason"]

    # Just outside the band is a result, and is NOT labelled within_noise.
    out = mod.contribution(full, _row("m", "drop_L", ours=104.0, et=100.0, dropped="L"))
    assert out["within_noise"] is False
    assert out["status"] == "helps"


def test_the_band_is_two_sided_so_a_small_regression_is_also_within_noise(mod):
    """A -1% "regression" is as unmeasurable as a +1% "win". Testing only the positive side would
    let a lever be reported as harmful on noise."""
    full = _row("m", "full", ours=100.0, et=100.0)
    c = mod.contribution(full, _row("m", "drop_L", ours=99.0, et=100.0, dropped="L"))
    assert c["contribution"] == pytest.approx(-0.01)
    assert c["within_noise"] is True
    assert c["status"] == "within_noise"


# --- refusals are not zeros --------------------------------------------------------------------

def test_a_refused_cell_yields_no_contribution_not_a_zero(mod):
    """A build that outran its ceiling, a gate that failed, an ExecuTorch arm that did not export:
    each is an ABSENT measurement. Folding it in as 0.0 would read as "measured, no effect" and
    would put the lever in the table as evidence of nothing mattering."""
    full = _row("m", "full", ours=100.0, et=100.0)
    drop = _row("m", "drop_L", dropped="L", status="refused",
                refusal="the ExecuTorch arm did not export: qnnpack partition empty")
    c = mod.contribution(full, drop)
    assert c["contribution"] is None
    assert c["contribution"] != 0
    assert c["status"] == "refused"
    assert "did not export" in c["reason"]

    # And it must not be counted as an attributed pair anywhere downstream.
    s = mod.attribute([full, drop], ["m"], ["L"])
    assert s["counts"]["attributed"] == 0
    assert s["counts"]["refused"] == 1
    assert s["per_model"]["m"]["levers"]["L"]["contribution"] is None


def test_a_missing_cell_is_incomplete_not_a_zero_contribution(mod):
    """A pair the session never got to is not a measurement of zero either."""
    full = _row("m", "full", ours=100.0, et=100.0)
    c = mod.contribution(full, None)
    assert c["contribution"] is None
    assert c["status"] == "incomplete"
    s = mod.attribute([full], ["m"], ["L"])
    assert s["counts"]["incomplete"] == 1
    assert s["counts"]["attributed"] == 0


def test_a_refused_full_cell_refuses_every_lever_of_that_model(mod):
    """The full arm is the common reference for every lever on a model. If it is absent, no lever on
    that model has a with-the-lever ratio, and none may be attributed."""
    full = _row("m", "full", status="refused", refusal="compile ceiling exceeded at 7000s")
    rows = [full, _row("m", "drop_A", ours=100.0, et=100.0, dropped="A"),
            _row("m", "drop_B", ours=100.0, et=100.0, dropped="B")]
    s = mod.attribute(rows, ["m"], ["A", "B"])
    assert s["counts"]["attributed"] == 0
    for lever in ("A", "B"):
        e = s["per_model"]["m"]["levers"][lever]
        assert e["contribution"] is None
        assert "compile ceiling" in e["reason"]


# --- the shared tree ---------------------------------------------------------------------------

def test_a_source_digest_mismatch_between_the_two_cells_refuses_the_contribution(mod):
    """This tree is shared and other sessions commit mid-run, so two cells of one pair really can be
    built from different compilers. The instrument stamps ``source_digest`` over the bytes it
    actually READ; a pair that disagrees on it is not a lever measurement and must not produce a
    number."""
    full = _row("m", "full", ours=100.0, et=100.0, digest="a" * 64)
    drop = _row("m", "drop_L", ours=130.0, et=100.0, dropped="L", digest="b" * 64)
    c = mod.contribution(full, drop)
    assert c["contribution"] is None
    assert c["status"] == "source_mismatch"
    assert "DIFFERENT compiler sources" in c["reason"]

    # Matching digests over the very same walls DO attribute -- so the refusal above is about the
    # digest and nothing else.
    ok = mod.contribution(full, _row("m", "drop_L", ours=130.0, et=100.0, dropped="L",
                                     digest="a" * 64))
    assert ok["contribution"] == pytest.approx(0.30)


@pytest.mark.parametrize("digest", ["", None, "UNKNOWN:OSError"])
def test_an_unusable_source_digest_refuses_as_firmly_as_a_mismatch(mod, digest):
    """Fail closed: a digest that could not be taken cannot show the two arms agree, so it is not a
    reason to proceed. Both cells carry the SAME unusable value, so an equality test alone would
    wave the pair through -- two cells that agree on having no provenance agree on nothing."""
    full = _row("m", "full", ours=100.0, et=100.0, digest=digest)
    drop = _row("m", "drop_L", ours=130.0, et=100.0, dropped="L", digest=digest)
    assert mod._digest_of(full) == mod._digest_of(drop)      # an equality check would pass here
    c = mod.contribution(full, drop)
    assert c["contribution"] is None
    assert c["status"] == "source_mismatch"

    # And when only ONE side is unusable, it is still refused rather than compared to a real digest.
    half = mod.contribution(_row("m", "full", ours=100.0, et=100.0, digest="a" * 64),
                            _row("m", "drop_L", ours=130.0, et=100.0, dropped="L", digest=digest))
    assert half["contribution"] is None
    assert half["status"] == "source_mismatch"


def test_uncommitted_sources_are_surfaced_on_an_attributed_pair(mod):
    """A pair built from a dirty tree can still be attributed -- both arms read the same bytes -- but
    the result cannot be reproduced from the commit alone, and the report has to say so."""
    full = _row("m", "full", ours=100.0, et=100.0, dirty=["impr_features.py"])
    drop = _row("m", "drop_L", ours=110.0, et=100.0, dropped="L", dirty=["k1.py"])
    c = mod.contribution(full, drop)
    assert c["contribution"] == pytest.approx(0.10)
    assert c["source_dirty"] == ["impr_features.py", "k1.py"]


# --- feature-name validation -------------------------------------------------------------------

def test_the_two_build_path_levers_validate_even_though_the_registry_lacks_them(mod):
    """``prepack_weight_layout`` and ``cse_through_provenance`` are real levers that are NOT in
    ``impr_features._REGISTRY`` until something imports the module that registers them. Validating
    against the bare registry would reject two of the levers this tool exists to ablate -- so the
    universe is a UNION, and the source that carries them is pinned here by name. A narrowing of the
    union back to the registry snapshot has to fail this test, not merely happen to keep working
    because some other test imported the registrar first."""
    both = {"prepack_weight_layout", "cse_through_provenance"}
    sources, failures = mod.feature_name_sources()
    assert failures == [], f"a name source failed to load, narrowing the universe: {failures}"
    assert both <= sources["llvmlower module FEATURE constants"]
    # The registry source must carry them too: `feature_name_sources` eagerly imports the two
    # modules that register the lazily-bound levers before snapshotting the registry. Without those
    # imports the registry snapshot is short by exactly these two, and every OTHER consumer of the
    # registry in this process (`impr_features.normalize`, which the compiler calls) is short too.
    assert both <= sources["impr_features registry"]
    assert both <= mod.known_feature_names()
    assert mod.unknown_features(sorted(both)) == []


def test_a_name_source_that_fails_to_load_is_reported_not_swallowed(mod, monkeypatch):
    """A swallowed import failure narrows the universe silently, and a valid lever then becomes a
    startup error whose message blames the caller for a name that is real."""
    real = mod.importlib.import_module
    victim = "merlin.llvmlower.weight_prepack"

    def boom(name, *a, **k):
        if name == victim:
            raise ImportError("simulated: a half-written module on a shared tree")
        return real(name, *a, **k)

    monkeypatch.setattr(mod.importlib, "import_module", boom)
    _, failures = mod.feature_name_sources()
    assert any(victim in f and "ImportError" in f for f in failures), failures


def test_the_feature_universe_is_a_union_of_more_than_the_registry(mod):
    """Three sources, each contributing something the others do not: the registry (pipeline-edit
    features), the llvmlower ``FEATURE`` constants (build-path levers), and the ranked list. If any
    of them is dropped the universe narrows silently and a valid lever becomes a startup error whose
    message blames the caller."""
    sources, _ = mod.feature_name_sources()
    assert set(sources) == {"impr_features registry",
                            "llvmlower module FEATURE constants",
                            "wholemodel_proposer.RANKED_LEVERS"}
    for label, names in sources.items():
        assert names, f"{label} contributed no names"
        assert names <= mod.known_feature_names(), f"{label} is not in the union"


def test_every_ranked_lever_validates(mod):
    """The ranked list is the menu this tool is pointed at; nothing on it may be rejected."""
    from merlin.mining.wholemodel_proposer import RANKED_LEVERS
    assert mod.unknown_features([n for n, _ in RANKED_LEVERS]) == []


def test_an_invented_lever_name_is_rejected_before_any_board_time(mod):
    """Fail closed. An unknown name reaches the instrument as a feature the compiler never applies,
    so its cell builds the SAME code as the full arm -- a zero contribution that looks like a
    measured absence of effect."""
    assert mod.unknown_features(["prepack_weight_layoutz"]) == ["prepack_weight_layoutz"]
    assert mod.is_known_feature("no_such_lever_at_all") is False


def test_main_exits_nonzero_on_an_unknown_lever_and_runs_nothing(mod, capsys):
    """Startup error, not a silent no-op -- and it must fire under --dry-run too, which is how the
    plan gets checked before a session."""
    rc = mod.main(["--models", "small_llama",
                   "--features", "prepack_weight_layout,not_a_real_lever", "--dry-run"])
    assert rc == 2
    out = capsys.readouterr().out
    assert "unknown lever name" in out
    assert "not_a_real_lever" in out


# --- resume ------------------------------------------------------------------------------------

def test_a_recorded_cell_is_not_re_run_and_a_refusal_counts_as_recorded(mod):
    """Re-deriving a refusal costs a board slot and learns nothing. ``--retry-refused`` is the
    escape hatch for refusals that were about the SESSION rather than the cell."""
    rows = [_row("m", "full", ours=1.0, et=1.0),
            _row("m", "drop_L", dropped="L", status="refused", refusal="board went away")]
    assert mod.recorded_keys(rows) == {"m::full", "m::drop_L"}
    assert mod.recorded_keys(rows, retry_refused=True) == {"m::full"}


def test_ledger_row_of_a_verdictless_record_is_refused_with_the_reason(mod):
    """The instrument writes a record even when it cannot produce a ratio. That record's refusal
    string is what the row must carry -- not a missing status and not a zero wall."""
    class _P:
        model = "m"
        ours_bundle_id = "m_int8_consistent"
    cell = mod.plan_cells(["L"])[0]
    rec = {"source_digest": "d", "source_dirty": [],
           "verdict_qd8": {"status": "not_measured",
                           "reason": "cannot extract a warm slope: need a passing wall at BOTH N"}}
    row = mod.ledger_row(_P(), cell, rec)
    assert row["status"] == "refused"
    assert "warm slope" in row["refusal"]
    assert row["ours_ns"] is None and row["executorch_warm_ns"] is None

    ok = mod.ledger_row(_P(), cell, {"source_digest": "d", "source_dirty": [],
                                     "verdict_qd8": {"status": "measured", "ours_ns": 7.0,
                                                     "executorch_warm_ns": 5.0}})
    assert ok["status"] == "measured"
    assert (ok["ours_ns"], ok["executorch_warm_ns"]) == (7.0, 5.0)
    assert ok["key"] == "m::full"


# --- the report ---------------------------------------------------------------------------------

def test_the_report_shows_both_ratios_and_never_a_bare_wall_delta(mod):
    """A reader must be able to see the two ratios the contribution came from, and must never be
    handed an ours_ns delta -- the quantity that is not comparable across sessions."""
    rows = [_row("m", "full", ours=100.0, et=100.0),
            _row("m", "drop_A", ours=130.0, et=100.0, dropped="A"),
            _row("m", "drop_B", ours=101.0, et=100.0, dropped="B"),
            _row("m", "drop_C", dropped="C", status="refused", refusal="gate failed cos=0.31")]
    s = mod.attribute(rows, ["m"], ["A", "B", "C"])
    text = mod.format_report(s)
    assert "ratio of ratios" in text
    assert "within_noise" in text          # lever B
    assert "REFUSED" in text               # lever C
    assert "gate failed" in text
    assert s["counts"] == {"pairs": 3, "attributed": 2, "outside_noise": 1, "within_noise": 1,
                           "refused": 1, "source_mismatch": 0, "incomplete": 0}
    # The serialized product must be JSON-clean (it is written to attribution.json every cell).
    json.dumps(s)
