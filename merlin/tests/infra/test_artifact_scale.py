"""An artifact that grows with its payload is refused where it arrives, with the usual cause named."""

from __future__ import annotations

from pathlib import Path

from merlin.targetgen import artifact_scale as AS
from merlin.targetgen import capsule_common


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
    # lower_interface owns the fourth entrypoint now: it writes the artifact for inspection, refuses an
    # oversized one, and only then memoizes it for the next capsule or returns it to a reader.
    text = Path(capsule_common.__file__).read_text(encoding="utf-8")
    write, guard = (
        text.index("(generated / artifact_name).write_text(_artifact"),
        text.index("_artifact_scale.refusal(_artifact)"),
    )
    assert write < guard < text.index("memo[memo_key] = ") < text.index("return cb, _artifact")
