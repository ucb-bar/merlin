"""The performance-feedback-levels map names the files, origins and schemas the writers actually use."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_CHECKS = importlib.util.spec_from_file_location("author_skill_perf_checks",
                                                 Path(__file__).with_name("author_skill_perf_checks.py"))
C = importlib.util.module_from_spec(_CHECKS)
_CHECKS.loader.exec_module(C)

NAME = "performance-feedback-levels"


def _text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in C.files(NAME) if p.suffix == ".md")


def test_skill_layout_is_generic_and_its_fields_and_commands_exist():
    C.check_layout(NAME)
    C.check_generic(NAME)
    assert C.missing_fields(NAME) == []
    assert C.bad_commands(NAME) == []
    assert {"roofline-and-utilization", "cycle-model-and-gsim", "ab-library-pass", "pass-library",
            "read-command-buffer", "feedback-files"} <= C.skills_named(NAME)


def test_named_files_origins_and_schemas_match_their_writers():
    from merlin_experiments.phase1.brokers import selfcheck_tuning
    from merlin_experiments.phase1.feedback import (
        completed_grade,
        idle_performance,
        idle_tuning,
        performance_brief,
        performance_headroom,
    )

    text = _text()
    for value in (f"qa/{completed_grade.FILENAME}", f"qa/{completed_grade.SUMMARY_FILENAME}",
                  idle_tuning.LEDGER.as_posix(), idle_tuning.SCHEMA, idle_performance.ORIGIN,
                  idle_performance.TUNING_ORIGIN, idle_performance.REQUEST_PREFIX,
                  f"submission/{selfcheck_tuning.DECLARATION}", selfcheck_tuning.SCHEMA,
                  performance_brief.SCHEMA, performance_headroom.SCHEMA):
        assert value in text, value


def test_idle_tuning_record_keys_are_the_ones_described():
    import inspect

    from merlin_experiments.phase1.feedback import idle_tuning

    source = inspect.getsource(idle_tuning.results)
    text = _text()
    for key in ("measured_cycles_before", "measured_cycles_after", "delta_pct", "numerics", "how_to_apply",
                "not_compared", "source_sha256"):
        assert f'"{key}"' in source and key in text
    # The "measured" keys hold Spike model estimates: the skill says so.
    assert "Spike timing model estimates" in source
    assert "Spike model estimates" in text
