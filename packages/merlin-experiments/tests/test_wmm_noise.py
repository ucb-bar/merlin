"""Each machine's noise comes from its own solo repeats: the improvement margin and the batch drift
tolerance follow it, and a machine without two same-day repeats is flagged rather than guessed."""

from __future__ import annotations

from pathlib import Path

from merlin_experiments.phase2.whole_model_measured import noise as N
from merlin_experiments.phase2.whole_model_measured import objective as O
from merlin_experiments.phase2.whole_model_measured.identity import write_json_atomic

from merlin.perf import whole_model_verdict as V

STOCK, LEAN = "s" * 64, "l" * 64


def _reading(path: Path, *, device: str, cycles: int, day: str, elf: str = "e" * 64, **fields) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        path / "result.json",
        {
            "timing_status": V.TIMING_MEASURED,
            "verdict": {"whole_window_cycles": cycles, "objective_cycles": cycles},
            "objective_cycles": cycles,
            "build": {"elf_sha256": elf},
            "device": {"binary_sha256": device},
            "finished_at": f"{day}T120000Z",
            **fields,
        },
    )
    return path / "result.json"


def test_same_day_and_cross_day_spreads_are_per_device(tmp_path):
    store = tmp_path / "store"
    _reading(store / "a", device=STOCK, cycles=1000, day="20261001")
    _reading(store / "b", device=STOCK, cycles=1010, day="20261001")
    _reading(store / "c", device=STOCK, cycles=1024, day="20261006")
    _reading(store / "d", device=LEAN, cycles=2000, day="20261001")
    readings = N.solo_readings([store])
    stock = N.machine_noise(readings, device=STOCK)
    assert stock["established"] and stock["flag"] is None
    assert stock["same_day"] == {"pairs": 1, "max": 0.01, "median": 0.01}
    assert stock["cross_day"]["max"] == 0.024
    lean = N.machine_noise(readings, device=LEAN)
    assert not lean["established"] and lean["flag"] == N.NOT_ESTABLISHED


def test_batched_carried_and_unmeasured_results_are_never_solo_readings(tmp_path):
    store = tmp_path / "store"
    _reading(store / "a", device=STOCK, cycles=1000, day="20261001")
    _reading(store / "b", device=STOCK, cycles=1500, day="20261001", batch={"size": 8})
    _reading(store / "c", device=STOCK, cycles=1500, day="20261001", same_program_as="a")
    _reading(store / "d", device=STOCK, cycles=1500, day="20261001", timing_status=V.TIMING_MEASURED_INVALID)
    _reading(store / "e", device=STOCK, cycles=1500, day="20261001", elf="f" * 64)  # another program
    assert not N.machine_noise(N.solo_readings([store]), device=STOCK)["established"]
    # MUTATION: one real same-day solo repeat of the same program establishes it.
    _reading(store / "f", device=STOCK, cycles=1001, day="20261001")
    assert N.machine_noise(N.solo_readings([store]), device=STOCK)["established"]


def test_every_attempt_of_a_job_is_a_reading(tmp_path):
    store = tmp_path / "store"
    _reading(store / "job", device=STOCK, cycles=1000, day="20261001")
    _reading(store / "job" / "attempts" / "0", device=STOCK, cycles=1020, day="20261001")
    assert N.machine_noise(N.solo_readings([store]), device=STOCK)["same_day"]["max"] == 0.02


def test_the_margin_is_the_largest_measured_noise_and_says_which():
    noisy = {"established": True, "same_day": {"max": 0.024}, "flag": None}
    assert N.margin(noisy, floor=0.001, batched_vs_solo=0.004) == {
        "margin": 0.024,
        "basis": "same_day_solo_spread",
        "candidates": {"floor": 0.001, "same_day_solo_spread": 0.024, "batched_vs_solo_median": 0.004},
        "established": True,
        "flag": None,
    }
    unknown = N.margin(None, floor=0.001)
    assert unknown["margin"] == 0.001 and unknown["basis"] == "floor" and unknown["flag"] == N.NOT_ESTABLISHED


def test_the_drift_tolerance_is_never_narrower_than_declared_and_widens_to_the_machines_drift():
    noise = {"established": True, "same_day": {"max": 0.0036}, "cross_day": {"max": 0.024}}
    stale = N.drift_tolerance(noise, declared=0.02, same_day=False)
    assert stale["tolerance"] == 0.024 and stale["basis"] == "cross_day_solo_spread"
    today = N.drift_tolerance(noise, declared=0.02, same_day=True)
    assert today["tolerance"] == 0.02 and today["basis"] == "declared"
    unknown = N.drift_tolerance({"established": False, "flag": N.NOT_ESTABLISHED}, declared=0.02, same_day=True)
    assert unknown["tolerance"] == 0.02 and unknown["flag"] == N.NOT_ESTABLISHED


def _batched(path: Path, *, device: str, batch: str, ratio: float, ok: bool = True) -> None:
    _reading(
        path,
        device=device,
        cycles=1,
        day="20261006",
        batch={"batch": batch, "size": 8, "control": {"ok": ok, "ratio": ratio}},
    )


def test_the_stock_boards_measured_control_readings(tmp_path):
    """The stock board, as measured 2026-10: the vendor control's ELF alone on Oct 1 (job 1451) and on
    Oct 6 (job 1916), and inside the Oct 6 batch (job 1939, ratio 1.0108).  No two solo readings share a
    day, so the run-to-run noise is NOT established and is flagged -- but the 2.47% the machine moved
    between the two solo runs is the margin, not the 0.1% floor."""
    store = tmp_path / "store"
    _reading(store / "job1451", device=STOCK, cycles=30_214_616, day="20261001")
    _reading(store / "job1916", device=STOCK, cycles=29_486_974, day="20261006")
    _batched(store / "job1939_candidate", device=STOCK, batch="b1939", ratio=1.0108)
    noise = N.machine_noise(N.solo_readings([store]), device=STOCK, controls=N.control_readings([store]))
    assert not noise["established"] and noise["flag"] == N.NOT_ESTABLISHED
    assert noise["cross_day"]["max"] == round((30_214_616 - 29_486_974) / 29_486_974, 6) == 0.024677
    assert noise["control_in_batch"] == {"pairs": 1, "max": 0.0108, "median": 0.0108}
    margin = N.margin(noise, floor=0.001)
    assert margin["margin"] == 0.024677 and margin["basis"] == "cross_day_solo_spread"
    assert margin["flag"] == N.NOT_ESTABLISHED
    # The Oct 6 batch's control against the same-day Oct 6 solo: within the declared 2%, flagged as unestablished.
    today = N.drift_tolerance(noise, declared=0.02, same_day=True)
    assert today["tolerance"] == 0.02 and today["flag"] == N.NOT_ESTABLISHED and abs(1.0108 - 1) <= today["tolerance"]
    # Against the five-day-old Oct 1 solo reading the machine's own measured drift is the bound.
    stale = N.drift_tolerance(noise, declared=0.02, same_day=False)
    assert stale["tolerance"] == 0.024677 and stale["basis"] == "cross_day_solo_spread"


def test_a_drifted_batch_control_is_not_the_machines_noise(tmp_path):
    """MUTATION: a control that did NOT hold must not widen anything -- the drift rule exists to catch it."""
    store = tmp_path / "store"
    _batched(store / "held", device=STOCK, batch="b1", ratio=1.004)
    _batched(store / "drifted", device=STOCK, batch="b2", ratio=1.09, ok=False)
    _batched(store / "other_board", device=LEAN, batch="b3", ratio=1.05)
    noise = N.machine_noise([], device=STOCK, controls=N.control_readings([store]))
    assert noise["control_in_batch"] == {"pairs": 1, "max": 0.004, "median": 0.004}
    tolerance = N.drift_tolerance(noise, declared=0.02, same_day=True)
    assert tolerance["tolerance"] == 0.02  # control readings never set the drift bound


class _Screen:
    """The objective's view of a screen store: its root and its (empty) job list."""

    def __init__(self, root: Path):
        self.root = root
        self.machine = {"kind": "spike"}
        self.build_options = {}

    def jobs(self):
        return []

    def result(self, digest):
        return None

    def result_by_key(self, key):
        return None

    def history(self):
        return []

    def attributable(self, key):
        return True


def test_the_objective_crowns_only_beyond_its_own_machines_noise(tmp_path):
    """The same objective over two machines' stores: 2.4% same-day noise on one, 0.36% on the other."""
    margins = {}
    for device, spread in ((STOCK, 0.024), (LEAN, 0.0036)):
        root = tmp_path / device[:1]
        _reading(root / "a", device=device, cycles=100_000, day="20261001")
        _reading(root / "b", device=device, cycles=int(100_000 * (1 + spread)), day="20261001")
        objective = O.WholeModelObjective(screen=_Screen(root), screen_reference=None)
        margins[device] = objective.noise_margin()
        assert objective.summary()["noise"]["established"] is True
    assert margins == {STOCK: 0.024, LEAN: 0.0036}


def test_a_machine_without_two_same_day_repeats_is_flagged_in_the_summary(tmp_path):
    root = tmp_path / "store"
    _reading(root / "a", device=STOCK, cycles=1000, day="20261001")
    _reading(root / "b", device=STOCK, cycles=1100, day="20261005")
    objective = O.WholeModelObjective(screen=_Screen(root), screen_reference=None)
    noise = objective.summary()["noise"]
    assert noise["established"] is False and noise["flag"] == N.NOT_ESTABLISHED
    # Flagged, and still not the floor: the 10% the machine moved between the two days is the margin.
    assert noise["margin"] == 0.1 and noise["basis"] == "cross_day_solo_spread" and noise["cross_day"]["max"] == 0.1


def test_the_screen_calibration_is_validated_against_the_pairs_boards_own_margin(tmp_path):
    """The fast path's refit holds its validation to the noise margin of the board its pairs came from."""
    from merlin_experiments.phase2.whole_model_measured import fast as F

    base = tmp_path / "base"
    _reading(base / "store0" / "a", device=STOCK, cycles=1000, day="20261001")
    _reading(base / "store0" / "b", device=STOCK, cycles=1030, day="20261006")
    _reading(base / "store1" / "c", device=LEAN, cycles=2000, day="20261001")
    (base / "_structure_screens").mkdir(parents=True)
    pairs = [{"domain": {"binary_sha256": STOCK}}]
    margin = F.screen_margin(base, pairs)
    assert margin["margin"] == 0.03 and margin["basis"] == "cross_day_solo_spread"  # the stock board's own
    assert margin["machine"]["device"] == STOCK
    # Pairs from two boards, or from none, have no one margin: the ranking stays unvalidated.
    assert F.screen_margin(base, [*pairs, {"domain": {"binary_sha256": LEAN}}]) is None
    assert F.screen_margin(base, []) is None


def test_the_rendered_screen_says_when_its_ranking_is_unvalidated():
    from merlin_experiments.phase2.whole_model_measured import fast as F

    screen = {
        "status": "screened",
        "label": "STRUCTURE SCREEN",
        "calibration": {"pairs": 0, "blind_ratio": 20.0},
        "ranking": {"status": "unvalidated", "reasons": ["too few pairs"]},
        "groups": [],
    }
    text = F.render_structure(screen)
    assert "screen ranking: UNVALIDATED" in text and "too few pairs" in text
    screen["ranking"] = {"status": "validated", "reasons": []}
    assert "screen ranking: VALIDATED" in F.render_structure(screen)
