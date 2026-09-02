"""A cycle-accurate console grows with the CYCLE CAP, not with the capsule — so it must be bounded.

Measured on GSIM at an 8M-cycle cap: one capsule's `gsim_console.log` was 646 MB / 5.19M lines of
per-cycle fetch trace, ~7 GB per graded run, and 17,398 orphaned scratch copies of the same shape had
reached 207 GB on a shared filesystem. Nothing reads the middle: the adapters read the head (boot +
config echo) and the tail (fault / halt marker / verdict), and so does a human debugging it.

The elision must be STATED. A silently truncated console cannot be reasoned about — you cannot tell a
sim that stopped early from one whose middle was discarded, and that distinction is the artifact's whole
diagnostic value.
"""
from __future__ import annotations

from merlin.targetgen.capsule_runner import _CONSOLE_MAX_BYTES, _write_console


def test_a_small_console_is_written_whole(tmp_path):
    p = tmp_path / "sim_console.log"
    text = "boot\nrun\nhalt\n"
    _write_console(p, text)
    assert p.read_text() == text, "a console under the cap must be byte-identical"


def test_an_oversized_console_is_capped(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_CONSOLE_MAX_BYTES", "4096")
    p = tmp_path / "sim_console.log"
    _write_console(p, "x" * 200_000)
    assert p.stat().st_size < 20_000, "the cap must actually bound the file"


def test_both_ends_survive(tmp_path, monkeypatch):
    """The head names the configuration, the tail carries the verdict. Neither may be the one dropped."""
    monkeypatch.setenv("MERLIN_CONSOLE_MAX_BYTES", "4096")
    p = tmp_path / "sim_console.log"
    _write_console(p, "BOOT-MARKER\n" + ("cyc\n" * 100_000) + "HALT-VERDICT\n")
    got = p.read_text()
    assert "BOOT-MARKER" in got, "the head (boot/config) must survive"
    assert "HALT-VERDICT" in got, "the tail (fault/halt/verdict) must survive"


def test_the_elision_says_how_much_it_dropped(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_CONSOLE_MAX_BYTES", "4096")
    p = tmp_path / "sim_console.log"
    _write_console(p, "head\n" + ("cyc\n" * 100_000) + "tail\n")
    got = p.read_text()
    assert "elided" in got, "a truncated console must SAY it was truncated"
    assert "bytes" in got and "lines" in got, "it must quantify what was dropped"
    assert "not a sim that stopped early" in got, \
        "the reader must be able to tell elision from an early halt"


def test_a_zero_or_negative_cap_disables_bounding(tmp_path, monkeypatch):
    """An operator who wants the whole trace can have it — the cap is a default, not a policy."""
    monkeypatch.setenv("MERLIN_CONSOLE_MAX_BYTES", "0")
    p = tmp_path / "sim_console.log"
    text = "a" * 50_000
    _write_console(p, text)
    assert p.read_text() == text


def test_a_malformed_cap_falls_back_to_the_default_rather_than_raising(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_CONSOLE_MAX_BYTES", "not-a-number")
    p = tmp_path / "sim_console.log"
    _write_console(p, "ok\n")
    assert p.read_text() == "ok\n"


def test_the_default_cap_is_small_enough_to_matter():
    """4 MiB against a measured 646 MB: the default has to be the fix, not a formality."""
    assert _CONSOLE_MAX_BYTES <= 16 << 20


def test_a_non_string_console_does_not_crash_the_grade(tmp_path):
    """Writing the console is bookkeeping beside a verdict; it must never take the verdict down."""
    p = tmp_path / "sim_console.log"
    _write_console(p, 12345)          # type: ignore[arg-type]
    assert p.read_text() == "12345"


def test_every_console_write_goes_through_the_bounded_writer():
    """A raw write_text on a console path is the unbounded growth coming back."""
    from merlin.common.paths import merlin_dir
    src = (merlin_dir() / "python/merlin/targetgen/capsule_runner.py").read_text()
    assert 'console.log").write_text(' not in src, \
        "console writes must route through _write_console, not write_text"
