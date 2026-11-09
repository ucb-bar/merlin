"""Checked consumers record admit/refuse decisions without changing results or refusals."""

import json
from dataclasses import replace

import pytest
from test_dense_panel_epochs import law, original

from merlin.compiler.schedule import library_sources as L
from merlin.compiler.schedule.dense_panel_epochs import schedule_dense_panel_epochs

API = "merlin.compile.scheduling.dense_panel_epochs.schedule_dense_panel_epochs"


@pytest.fixture(autouse=True)
def fresh_records(monkeypatch):
    monkeypatch.setattr(L, "_ADMISSIONS", {})
    monkeypatch.setattr(L, "_ADMISSION_STATE", {"dropped": 0, "exit_hook": True})
    monkeypatch.delenv(L.ADMISSION_ENV, raising=False)


def test_real_consumer_records_admit_and_refusal_and_keeps_its_answers(capsys):
    p, r, s, b, a, adapter, result = original(batch=True)
    checked = schedule_dense_panel_epochs(result, p, b, r, s, adapter, a, law(adapter))
    unrecorded = schedule_dense_panel_epochs.__wrapped__(result, p, b, r, s, adapter, a, law(adapter))
    assert checked.kernel.digest() == unrecorded.kernel.digest()
    assert checked.realization == unrecorded.realization
    with pytest.raises(ValueError, match="original overlap admission") as refused:
        schedule_dense_panel_epochs(result, p, b, r, s, adapter, replace(a, max_issued_loads=65), law(adapter))
    with pytest.raises(ValueError) as direct:
        schedule_dense_panel_epochs.__wrapped__(result, p, b, r, s, adapter, replace(a, max_issued_loads=65),
                                                law(adapter))
    assert str(refused.value) == str(direct.value)
    rows = [row for row in L.admission_records() if row["api"] == API]
    assert [(row["api"], row["admitted"]) for row in rows] == [(API, True), (API, False)]
    refusal = rows[1]
    assert refusal["failed_premise"] == str(refused.value)
    assert "unchanged input checked result" in refusal["fallback_route"]
    assert "(input route: two_window)" in refusal["fallback_route"]
    assert rows[0]["shape"] == {"m": p.m, "k": p.k, "n": p.n}
    assert schedule_dense_panel_epochs.admission_api == API
    # One line per distinct decision reaches stderr, and the host parser reads it back.
    parsed, dropped = L.parse_admission_stream(capsys.readouterr().err)
    assert dropped == 0
    assert [(row["api"], row["admitted"], row["failed_premise"]) for row in parsed if row["api"] == API] == [
        (API, True, None), (API, False, str(refused.value))]


def test_repeats_are_counted_and_summarized(capsys):
    p, r, s, b, a, adapter, result = original()
    for _ in range(3):
        with pytest.raises(ValueError):
            schedule_dense_panel_epochs(result, p, b, r, s, adapter, replace(a, max_issued_loads=65), law(adapter))
    L._admission_summary()
    parsed, _ = L.parse_admission_stream(capsys.readouterr().err)
    parsed = [row for row in parsed if row["api"] == API]
    assert len(parsed) == 1 and parsed[0]["count"] == 3 and parsed[0]["admitted"] is False


def test_none_refusals_opt_out_and_bounded_records(monkeypatch, capsys):
    @L.admission_point("example.plan", fallback_route="ordinary route", none_refuses="does not fit")
    def plan(fit):
        return {"fit": fit} if fit else None

    @L.admission_point("example.consumer", fallback_route="ordinary route")
    def consumer(text):
        raise TypeError(text)

    monkeypatch.setenv(L.ADMISSION_ENV, "0")
    assert plan(True) == {"fit": True}
    assert plan(False) is None
    with pytest.raises(TypeError, match="bad input"):
        consumer("bad input")
    assert capsys.readouterr().err == ""
    rows = L.admission_records()
    assert [(r["api"], r["admitted"], r["failed_premise"]) for r in rows] == [
        ("example.plan", True, None), ("example.plan", False, "does not fit"),
        ("example.consumer", False, "TypeError: bad input")]
    for index in range(L.ADMISSION_LIMIT + 5):
        with pytest.raises(TypeError):
            consumer(f"premise {index}")
    assert len(L.admission_records()) == L.ADMISSION_LIMIT
    assert L._ADMISSION_STATE["dropped"] == 8


def test_parser_ignores_foreign_and_malformed_lines():
    text = "\n".join((
        "ordinary compiler output",
        L.ADMISSION_PREFIX + "{not json",
        L.ADMISSION_PREFIX + '{"schema": "other", "id": 0, "api": "x", "admitted": true}',
        L.ADMISSION_PREFIX + json.dumps({"schema": L.ADMISSION_SCHEMA, "id": 0, "api": "x", "admitted": "yes"}),
        L.ADMISSION_PREFIX + json.dumps({"schema": L.ADMISSION_SCHEMA, "id": 1, "api": "y", "admitted": False,
                                         "failed_premise": "p", "fallback_route": "r"}),
    ))
    rows, dropped = L.parse_admission_stream(text)
    assert dropped == 0
    assert rows == [{"api": "y", "admitted": False, "failed_premise": "p", "fallback_route": "r", "shape": None,
                     "partial_refusals": None, "count": 1}]
