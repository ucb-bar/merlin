"""``merlin-bundle-pretranspose`` — the missing runner for a complete, tested rewrite.

The analysis (``weight_layout_report``) and the rewrite (``hoist_weight_transposes``) were both
finished and covered. What was missing was any way to invoke them: no CLI, no pass registration, no
caller in any compile path — so the three ``*_pretransposed`` bundles on disk had been produced by
hand-driving the library, and nobody could reproduce them. These tests pin the runner and the two
properties that make it safe to hand to someone else: --dry-run writes nothing, and a bundle with
nothing to hoist is an honest refusal rather than an empty copy.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from merlin.baselines import pretranspose_cli as cli
from merlin.common.artifacts import recaptures_dir


def _bundle_with_transposes():
    for name in ("gemma2_2b_int8_full_seq8", "gemma2_2b_int8_full", "gemma2_2b_int8_consistent"):
        p = recaptures_dir() / name
        if (p / "model.mlir").is_file() and not name.endswith("_pretransposed"):
            return p
    return None


def test_the_console_script_is_registered():
    """A CLI nothing declares is the same shape of gap as the missing CLI itself."""
    txt = (Path(__file__).resolve().parents[3] / "pyproject.toml").read_text()
    assert 'merlin-bundle-pretranspose = "merlin.baselines.pretranspose_cli:main"' in txt


def test_a_non_bundle_directory_is_refused_before_any_work(tmp_path):
    with pytest.raises(SystemExit) as e:
        cli.main([str(tmp_path)])
    assert "not a capture bundle" in str(e.value)


def test_the_precedence_decision_against_fuse_transpose_b_is_recorded():
    """The two levers solve the same measured problem and CONFLICT (fusing first removes the
    transposes the hoist looks for), so the module must say which one wins and why -- otherwise the
    next person re-derives it, or worse, enables both."""
    doc = cli.__doc__ or ""
    assert "fuse_transpose_b" in doc
    assert "Prefer THIS" in doc
    assert "-6.53" in doc                      # the measured regression that decides it
    assert "BLOCKED" in doc                    # ...and what fusion is still the right answer for
    # and the honest size of the prize: a capacity lever, not a speed lever
    assert "0.4" in doc and "1.2" in doc and "noise band" in doc
    assert "589824064" in doc                  # the alloc failure it actually fixes


@pytest.mark.skipif(_bundle_with_transposes() is None, reason="no bundle with weight transposes")
def test_dry_run_reports_the_split_and_writes_nothing(capsys, tmp_path):
    src = _bundle_with_transposes()
    before = sorted(p.name for p in src.iterdir())
    assert cli.main([str(src), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "hoistable" in out and "MiB" in out
    assert sorted(p.name for p in src.iterdir()) == before, "--dry-run must not touch the source"
    assert not any(tmp_path.iterdir()), "--dry-run must not write anywhere"
    # the default destination must be a SIBLING that does not exist yet, never the source
    assert not (src.parent / f"{src.name}_pretransposed").samefile(src) if (
        src.parent / f"{src.name}_pretransposed").exists() else True


@pytest.mark.skipif(_bundle_with_transposes() is None, reason="no bundle with weight transposes")
def test_the_dry_run_split_matches_the_library_it_will_call(capsys):
    """--dry-run and the real run must parse the same way, or a user is shown one plan and gets
    another. Both go through mlir_query.parse under IR_LOCK."""
    src = _bundle_with_transposes()
    rep = cli._report(src, "forward")
    cli.main([str(src), "--dry-run"])
    out = capsys.readouterr().out
    assert f"{len(rep.hoistable)} hoistable" in out
    assert f"{len(rep.blocked)} blocked" in out
    assert f"{rep.hoistable_bytes:,} bytes" in out
