"""The structure screen: a whole-model program on the functional simulator reads correctness per group
and never feeds an objective; a partial console is refused, not read."""

from __future__ import annotations

from merlin.perf import whole_model_screen as S

CONSOLE = "\n".join(
    [
        "GM_GROUP 1 conv 100 sum=1 fnv1a=2",
        "GM_LOCAL 1 mismatches=0 of=64 first=-1",
        "GM_GROUP 2 add 50 sum=UNKNOWN fnv1a=UNKNOWN",
        "GM_BOUND 2 max_abs=3 over=2 bound=1",
        "FM full model cycles: 150",
        "GM_ARGMAX got=4 want=4 agrees=1",
        "MERLIN_WINDOW end label=m",
    ]
)


def test_each_group_is_read_by_its_own_local_check_and_never_feeds_an_objective():
    screen = S.screen_console(CONSOLE, {"1": "exact", "2": "bounded_int"})
    assert screen["status"] == "screened" and screen["feeds_objective"] is False
    rows = {r["group"]: r for r in screen["groups"]}
    assert rows[1]["local"] == "correct" and rows[2]["local"] == "wrong" and rows[2]["failure"]["over"] == 2
    assert screen["groups_not_correct"] == [2] and screen["argmax"] == [4, 4, 1]


def test_a_console_missing_an_expected_group_is_refused():
    screen = S.screen_console(CONSOLE, {"1": "exact", "2": "bounded_int", "3": "exact"})
    assert screen["status"] == "refused" and "missing ['3']" in screen["refusal"]


# --------------------------------------------------------------- the refit validates on held-out groups

DOMAIN = {"target": "toy", "rung": "board", "binary_sha256": "b" * 64}
MARGIN = {"margin": 0.02, "basis": "cross_day_solo_spread"}


def _pairs(ratios: dict[str, float], candidates: int, *, domain=DOMAIN) -> list[dict]:
    """Board-measured groups of ``candidates`` programs each: every group a conv whose board cycles are its
    simulator cycles times that group's ratio."""
    import hashlib

    rows = []
    for c in range(candidates):
        elf = hashlib.sha256(f"candidate {c}".encode()).hexdigest()
        for group, ratio in ratios.items():
            spike = 1000 * (c + 1) + int(group)
            rows.append(
                {
                    "source": f"job {c}",
                    "group": group,
                    "kind": "conv",
                    "on": "package:mm",
                    "spike": spike,
                    "board": int(round(spike * ratio)),
                    "elf_sha256": elf,
                    "domain": dict(domain),
                    "evidence_sha256s": [hashlib.sha256(f"{c}/{group}".encode()).hexdigest()],
                }
            )
    return rows


def test_the_statistical_minimum_and_rate_are_derived_from_the_margin():
    # 0.5**6 = 0.0156 <= 0.02 < 0.5**5: six unanimous pairs are the fewest that are evidence at 2%.
    assert S.minimum_decided_pairs(0.02) == 6
    assert S.minimum_rank_rate(6, 0.02) == 1.0  # on six pairs only unanimity beats a coin at 2%
    assert 0.5 < S.minimum_rank_rate(400, 0.02) < 0.6  # many pairs: a modest edge is already evidence
    for bad in (0.0, 1.0, float("nan")):
        try:
            S.minimum_decided_pairs(bad)
        except ValueError:
            continue
        raise AssertionError(f"margin {bad} was accepted")


def test_a_refit_that_predicts_held_out_groups_is_validated_and_recorded():
    calibration = S.fit_calibration(_pairs({"1": 4.0, "2": 4.0, "3": 4.0}, 5), margin=MARGIN)
    validation = calibration["validation"]
    assert validation["status"] == S.VALIDATED, validation["reasons"]
    thresholds = validation["thresholds"]
    assert thresholds["maximum_relative_error"] == MARGIN["margin"] and thresholds["minimum_predictions"] == 15
    assert thresholds["minimum_decided"] == 6 and validation["absolute_error"]["maximum_relative"] == 0.0
    screen = S.screen_console(CONSOLE, {"1": "exact", "2": "bounded_int"}, calibration=calibration)
    assert screen["ranking"] == {"status": S.VALIDATED, "reasons": []}
    assert screen["calibration"]["validation"]["status"] == S.VALIDATED


def test_a_refit_that_misses_held_out_groups_beyond_the_margin_is_unvalidated():
    calibration = S.fit_calibration(_pairs({"1": 4.0, "2": 8.0, "3": 2.0}, 5), margin=MARGIN)
    validation = calibration["validation"]
    assert validation["status"] == S.UNVALIDATED
    assert any("relative error" in reason for reason in validation["reasons"]), validation["reasons"]
    screen = S.screen_console(CONSOLE, {"1": "exact", "2": "bounded_int"}, calibration=calibration)
    assert screen["ranking"]["status"] == S.UNVALIDATED and screen["ranking"]["reasons"]


def test_too_few_measured_pairs_fail_closed_before_any_rate_is_read():
    validation = S.fit_calibration(_pairs({"1": 4.0, "2": 4.0}, 2), margin=MARGIN)["validation"]
    assert validation["status"] == S.UNVALIDATED and validation["measured_order_pairs"] == 2
    assert "at least 6 are needed" in validation["reasons"][0]


def test_without_a_margin_or_a_binding_the_ranking_is_unvalidated():
    pairs = _pairs({"1": 4.0, "2": 4.0, "3": 4.0}, 5)
    assert S.fit_calibration(pairs)["validation"]["status"] == S.UNVALIDATED  # no machine margin given
    unbound = [dict(pairs[0], domain=None), *pairs[1:]]
    assert "binding" in S.fit_calibration(unbound, margin=MARGIN)["validation"]["reasons"][0]
    mixed = [*pairs, *_pairs({"4": 4.0}, 5, domain={**DOMAIN, "binary_sha256": "c" * 64})]
    assert "more than one measurement domain" in S.fit_calibration(mixed, margin=MARGIN)["validation"]["reasons"][0]
    screen = S.screen_console(CONSOLE, {"1": "exact", "2": "bounded_int"})
    assert screen["ranking"]["status"] == S.UNVALIDATED  # no calibration at all: the gate's screen
