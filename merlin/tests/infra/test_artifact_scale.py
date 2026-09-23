"""An artifact that grows with its payload is refused where it arrives, with the usual cause named."""

from __future__ import annotations

from merlin.common.paths import merlin_dir
from merlin.targetgen import artifact_scale as AS


def test_an_ordinary_artifact_is_not_refused_and_a_payload_scale_one_says_why(monkeypatch) -> None:
    monkeypatch.setenv(AS.LIMIT_ENV, "1000")
    assert AS.refusal("x" * 1000) is None
    message = AS.refusal("x" * 4000, elements=100)
    assert "about 40 bytes for each of its 100 payload elements" in message
    assert "unrolled over its DATA" in message and "Emit a loop" in message


def test_the_limit_is_declared_once_and_a_bad_override_falls_back(monkeypatch) -> None:
    monkeypatch.delenv(AS.LIMIT_ENV, raising=False)
    assert AS.limit_bytes() == AS.DEFAULT_LIMIT_BYTES == 64 * 1024 * 1024
    for bad in ("", "0", "-5", "lots"):
        monkeypatch.setenv(AS.LIMIT_ENV, bad)
        assert AS.limit_bytes() == AS.DEFAULT_LIMIT_BYTES


def test_the_capsule_pipeline_checks_the_artifact_before_anything_reads_it_twice() -> None:
    # Held by source: the pipeline's failure paths are exercised by whole-suite runs, and an
    # unwired guard looks exactly like a corpus of ordinary programs.
    text = (merlin_dir() / "python/merlin/targetgen/capsule_common.py").read_text(encoding="utf-8")
    write, guard = (
        text.index("fourth_output_name).write_text(p.stdout"),
        text.index("_artifact_scale.refusal(p.stdout)"),
    )
    assert write < guard < text.index("return pkg, cb, p.stdout")
