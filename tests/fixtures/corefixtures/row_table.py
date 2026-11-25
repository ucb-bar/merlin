"""Run a table of independent checks inside one collected case and report every failed row."""

import traceback

import pytest


def run_rows(tmp_path, monkeypatch, rows):
    """Run each `check(directory, patch)` in `rows`, a mapping from row name to check.

    Every row gets its own directory under `tmp_path` and its own monkeypatch context, so a row
    cannot see another row's files or patches. Every row runs; the case fails once, naming each
    failed row with its traceback.
    """
    failures = []
    for name, check in rows.items():
        directory = tmp_path / name
        directory.mkdir()
        try:
            with monkeypatch.context() as patch:
                check(directory, patch)
        except (Exception, pytest.fail.Exception):
            failures.append((name, traceback.format_exc()))
    if failures:
        detail = "\n".join(f"--- row {name} ---\n{text}" for name, text in failures)
        pytest.fail(f"failed rows: {', '.join(name for name, _ in failures)}\n{detail}", pytrace=False)
