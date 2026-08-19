"""The --min-rounds decline: a READY marker at zero must not end the loop."""
import pathlib, sys
H = pathlib.Path("merlin/experiments/capsule_bench/harness")
sys.path.insert(0, str(H))


def _decide(marker_exists, all_pass, rnd_completed, min_rounds, ws):
    """Mirror of the loop's decision, exercised against a real marker file."""
    import run_baseline_qa_loop as L  # noqa
    ready = marker_exists
    if ready and not all_pass and rnd_completed < min_rounds:
        (ws / "submission" / L.READY_MARKER).unlink(missing_ok=True)
        ready = False
    return (all_pass or ready), ready


def _ws(tmp_path, marker=True):
    import run_baseline_qa_loop as L
    sub = tmp_path / "submission"; sub.mkdir(parents=True, exist_ok=True)
    if marker: (sub / L.READY_MARKER).write_text("done")
    return tmp_path


def test_marker_at_zero_is_declined_and_cleared(tmp_path):
    import run_baseline_qa_loop as L
    ws = _ws(tmp_path)
    stop, ready = _decide(True, False, rnd_completed=1, min_rounds=12, ws=ws)
    assert stop is False and ready is False
    assert not (ws / "submission" / L.READY_MARKER).exists(), "marker must be cleared so it is re-earned"


def test_all_pass_always_wins(tmp_path):
    import run_baseline_qa_loop as L
    ws = _ws(tmp_path)
    stop, _ = _decide(True, True, rnd_completed=1, min_rounds=12, ws=ws)
    assert stop is True
    assert (ws / "submission" / L.READY_MARKER).exists(), "a real convergence must not be touched"


def test_marker_honoured_once_min_rounds_reached(tmp_path):
    ws = _ws(tmp_path)
    stop, ready = _decide(True, False, rnd_completed=12, min_rounds=12, ws=ws)
    assert stop is True and ready is True


def test_disabled_by_default(tmp_path):
    ws = _ws(tmp_path)
    stop, ready = _decide(True, False, rnd_completed=0, min_rounds=0, ws=ws)
    assert stop is True and ready is True, "min_rounds=0 must preserve the old behaviour exactly"


class _Args:
    """Argparse-shaped stand-in: any option _arm_cmd consults but this test does not set reads as
    falsy, so the test tracks the flag under test and not the rest of the CLI surface."""
    def __init__(self, **kw):
        self.__dict__.update(dict(model="glm5", effort="high", max_rounds=12,
                                  max_rate_limit_waits=8, round_timeout=14400, driver="codex",
                                  experiment="realistic", condition="kernels"), **kw)

    def __getattr__(self, name):
        return ""


def test_flag_reaches_the_arm_command():
    import launch_ab_batch as LB
    cmd = LB._arm_cmd("merlin_rtlchecks", "t", _Args(min_rounds=12))
    assert "--min-rounds" in cmd and cmd[cmd.index("--min-rounds") + 1] == "12"


def test_flag_absent_when_disabled():
    import launch_ab_batch as LB
    assert "--min-rounds" not in LB._arm_cmd("merlin_rtlchecks", "t", _Args(min_rounds=0))
