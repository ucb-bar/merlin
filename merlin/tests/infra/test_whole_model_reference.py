"""An open model's end result is judged against its REFERENCE ARM, not the oracle.

The model's own host code with exact devices, on the same ISA, toolchain, machine and simulator: a
correct package's output is bit-identical to it (every dispatch is integer-exact), and every dispatch's
result digest agrees. The cache key names everything the reference's bytes depend on, and a part
nobody can state fails the end result closed. A model may DECLARE a statistical fallback, judged
against the same reference, and the result says which criterion it applied.
"""

from __future__ import annotations

import json
import struct

import pytest
import selected_driver

from merlin.perf import whole_model_open as WO
from merlin.perf import whole_model_reference as R

# The verdict's UART parser is the measurement service's (phase-2 port); these run once it is present.
V = pytest.importorskip("merlin.perf.whole_model_verdict")

pytestmark = pytest.mark.target("gemmini")


def _bits(values):
    return [struct.unpack("<I", struct.pack("<f", float(v)))[0] for v in values]


def _console(values, *, digest: int, words=(("18", 7), ("22", 9)), local=0) -> str:
    lines = [f"GM_LOCAL {g} mismatches={local} of=4 first=-1" for g, _ in words]
    lines += [f"GM_WORDS {g} bytes=384 digest={d}" for g, d in words]
    lines += [
        f"GM_OUTPUT_DIGEST bytes={4 * len(values)} digest={digest}",
        f"OUT {len(values)} " + " ".join(str(b) for b in _bits(values)),
    ]
    return "\n".join(lines) + "\n"


_VALUES = [0.5, -1.0, 2.0, 0.25]


def _reference(values=_VALUES, digest=1234) -> dict:
    return {
        "key": "k" * 64,
        "output_digest": {"bytes": 4 * len(values), "digest": digest},
        "output_bits": _bits(values),
        "words": {"18": [384, 7], "22": [384, 9]},
        "elf_sha256": "e" * 64,
        "simulator": {"name": "spike"},
        "provenance": {"pins": {}},
    }


@selected_driver.requires_support("gemmini")
def test_the_program_prints_its_whole_outputs_digest_and_the_verdict_reads_it() -> None:
    from merlin.runtime.backends import base as backends

    driver = backends.whole_model_driver("gemmini")
    main = driver.dispatch.render_main(driver.program.UART, [], words_helper=driver.program._WORDS_HELPER)
    assert "GM_OUTPUT_DIGEST bytes=%llu digest=%llu" in main and "words_digest(OUT" in main
    parsed = V.parse_log(_console(_VALUES, digest=99))
    assert parsed.output_digest == (16, 99)


def test_a_correct_package_is_bit_identical_to_its_reference_arm() -> None:
    verdict = R.judge(_console(_VALUES, digest=1234), _reference())
    assert verdict["passed"] and verdict["end_result_criterion"] == R.BIT_IDENTICAL and verdict["bit_identical"]
    assert verdict["words"]["agree"] == 2 and verdict["reference"]["elf_sha256"] == "e" * 64


def test_a_package_that_corrupts_one_element_fails() -> None:
    corrupted = [*_VALUES[:3], 0.2500001]
    verdict = R.judge(_console(corrupted, digest=1235), _reference())
    assert not verdict["passed"] and not verdict["bit_identical"]


def test_a_dispatch_whose_result_differs_fails_even_when_the_output_agrees() -> None:
    verdict = R.judge(_console(_VALUES, digest=1234, words=(("18", 7), ("22", 10))), _reference())
    assert not verdict["passed"] and verdict["words"]["differ"] == ["22"]


def test_the_declared_bound_is_judged_against_the_same_reference_and_says_so() -> None:
    near = [v + 1e-3 for v in _VALUES]
    declared = {"criterion": R.DECLARED_BOUND, "cosine_min": 0.999, "max_abs": 0.01}
    verdict = R.judge(_console(near, digest=1), _reference(), declaration=declared, elements=4)
    assert verdict["passed"] and verdict["end_result_criterion"] == R.DECLARED_BOUND
    assert not verdict["bit_identical"] and verdict["bound"]["max_abs"] < 0.01
    far = R.judge(_console([v + 0.5 for v in _VALUES], digest=1), _reference(), declaration=declared, elements=4)
    assert not far["passed"] and far["end_result_criterion"] == R.DECLARED_BOUND
    # A bound over part of the output is not a bound over the output.
    partial = R.judge(_console(near, digest=1), _reference(), declaration=declared, elements=8)
    assert not partial["passed"]


def test_an_unknown_key_part_fails_closed(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(R, "simulator_identity", lambda target, simulator: {"digest": "s" * 64})
    record = {"reference_identity": {"capsule": {"digest": "c" * 64}, "host_code": {"digest": R.UNKNOWN}}}
    verdict = R.end_result(_console(_VALUES, digest=1234), record, target="gemmini", simulator="spike", root=tmp_path)
    assert not verdict["passed"] and "UNKNOWN" in verdict["note"] and "host_code" in verdict["note"]
    # A digest over parts is UNKNOWN when any part is, at any depth.
    assert R.identity_digest({"a": {"b": R.UNKNOWN}}) == R.UNKNOWN
    assert R.files_identity({"missing": tmp_path / "nope"})["digest"] == R.UNKNOWN


def test_the_same_host_code_built_in_two_directories_has_one_identity(tmp_path) -> None:
    """A candidate and its reference arm are built in different directories, and the include flags name
    those directories. Measured: every host file digested equal and the identities still differed, so
    every whole-model gate refused its own reference."""
    source = tmp_path / "main.c"
    source.write_text("int main(void) { return 0; }\n")
    one, two = tmp_path / "cand" / "build", tmp_path / "ref" / "build"
    left = R.files_identity({"main.c": source}, [f"main.o:-I{one}/harness", "-O2"], root=one)
    right = R.files_identity({"main.c": source}, [f"main.o:-I{two}/harness", "-O2"], root=two)
    assert left["digest"] == right["digest"] != R.UNKNOWN
    assert left["flags"] == [f"main.o:-I{R.BUILD_ROOT}/harness", "-O2"]
    # A path outside the build, or one that merely shares the prefix, is kept as it is.
    other = R.files_identity({"main.c": source}, [f"-I{one}x/harness", "-I/opt/include"], root=one)
    assert other["flags"] == [f"-I{one}x/harness", "-I/opt/include"]
    assert R.files_identity({"main.c": source}, ["-O3"], root=one)["digest"] != left["digest"]


def _record() -> dict:
    return {
        "reference_identity": {
            "capsule": {"digest": "c" * 64},
            "host_code": {"digest": "h" * 64},
            "toolchain": {"digest": "t" * 64},
            "machine": {"machine": "m", "header_sha256": "a" * 64},
        }
    }


def test_a_cached_reference_is_used_under_its_key_and_a_missing_one_is_not_guessed(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(R, "simulator_identity", lambda target, simulator: {"digest": "s" * 64})
    key = R.reference_key(R.key_parts(_record(), {"digest": "s" * 64}))
    missing = R.end_result(
        _console(_VALUES, digest=1234),
        _record(),
        target="gemmini",
        simulator="spike",
        root=tmp_path,
        produce_missing=False,
    )
    assert not missing["passed"] and "no reference arm is cached" in missing["note"]
    (tmp_path / f"{key}.json").write_text(json.dumps({**_reference(), "key": key}), encoding="utf-8")
    found = R.end_result(_console(_VALUES, digest=1234), _record(), target="gemmini", simulator="spike", root=tmp_path)
    assert found["passed"] and found["reference"]["key"] == key
    # Another simulator build is another key.
    other = R.key_parts(_record(), {"digest": "z" * 64})
    assert R.reference_key(other) != key


def test_the_open_grade_gates_on_the_reference_and_only_reports_the_chained_check() -> None:
    expectations = {
        "groups": [18, 22],
        "golden": _VALUES,
        "oracle_output": _VALUES,
        "numeric_policy": {"atol": 0.03125, "rtol": 0.02},
        "hostin_bound": 1,
    }
    drifted = _console(_VALUES, digest=1234) + "GM_HOSTIN 18 max_abs=8 over=5 of=64 bound=1\n"
    verdict = WO.grade(drifted, expectations, reference=_reference())
    assert verdict["quotable"] and verdict["hostin"]["gates"] is False and verdict["hostin"]["worst"] == 8
    assert not WO.grade(drifted, expectations)["quotable"]  # no reference arm: not judged


def test_the_gate_judges_a_tensor_model_on_its_reference_and_keeps_the_oracle_as_a_finding(monkeypatch, tmp_path):
    from merlin.perf import whole_model_gate as G

    monkeypatch.setattr(R, "simulator_identity", lambda target, simulator: {"digest": "s" * 64})
    monkeypatch.setattr(R, "cache_root", lambda target: tmp_path / "cache")
    key = R.reference_key(R.key_parts(_record(), {"digest": "s" * 64}))
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / f"{key}.json").write_text(json.dumps({**_reference(), "key": key}), encoding="utf-8")
    (tmp_path / "run" / "screen").mkdir(parents=True)
    (tmp_path / "run" / "screen" / "console.txt").write_text(_console(_VALUES, digest=1234), encoding="utf-8")
    stated = tmp_path / "expectations.json"
    stated.write_text(
        json.dumps({"golden": _VALUES, "oracle_output": [v + 1.0 for v in _VALUES], "numeric_policy": {"atol": 0.1}})
    )
    expectations = {"argmax": None, "output": {"elements": 4}, "source": str(stated)}
    screen = {"simulator": "spike", "output": [1, 4]}
    verdict, words = G._against_reference(
        screen,
        _record(),
        expectations,
        {"name": "m"},
        target="gemmini",
        out=tmp_path / "run",
        templates=None,
        timeout=60,
        jobs=1,
    )
    assert verdict["passed"] and verdict["end_result_criterion"] == R.BIT_IDENTICAL and words["passed"]
    # The oracle's policy check fails this faithful run and is kept, as a diagnostic, beside the finding.
    assert not verdict["diagnostics"]["within_policy_of_oracle"]["passed"]
    assert verdict["diagnostics"]["oracle_vs_golden"] == {
        "oracle_within_policy_of_golden": 0,
        "of": 4,
        "policy": {"atol": 0.1, "rtol": 0.0},
    }


def test_the_reference_arm_is_cut_into_the_candidates_chunks(monkeypatch, tmp_path) -> None:
    """A chunked candidate's host code digests equal only to a reference cut the same way, so the
    reference build takes the candidate's chunk size from its record (and none when it has none)."""
    seen: list = []

    def fake_build(package, capsule, **kwargs):
        seen.append(kwargs.get("chunk_ops"))
        raise R.ReferenceUnavailable("stop after the build call")

    monkeypatch.setattr(WO, "build", fake_build)
    for record, want in (({**_record(), "forward_chunks": {"chunk_ops": 64, "chunks": 835}}, 64), (_record(), None)):
        with pytest.raises(R.ReferenceUnavailable):
            R.produce(
                record,
                target="gemmini",
                simulator="spike",
                key=f"k{want}",
                parts={},
                simulator_stated=None,
                timeout=60,
                root=tmp_path,
                wait_s=0,
            )
    assert seen == [64, None]


def test_the_reference_arm_runs_on_the_candidates_harts(monkeypatch, tmp_path) -> None:
    """A candidate whose host code runs on a vector hart is compiled for that hart's ISA; the
    reference is built on the same harts (and on one when the candidate ran on one)."""
    seen: list = []

    def fake_build(package, capsule, **kwargs):
        seen.append(kwargs.get("host_hart"))
        raise R.ReferenceUnavailable("stop after the build call")

    monkeypatch.setattr(WO, "build", fake_build)
    two = {**_record(), "program": {**(_record().get("program") or {}), "harts": {"host": 1, "unit": 0}}}
    for record in (two, _record()):
        with pytest.raises(R.ReferenceUnavailable):
            R.produce(
                record,
                target="gemmini",
                simulator="spike",
                key="k",
                parts={},
                simulator_stated=None,
                timeout=60,
                root=tmp_path,
                wait_s=0,
            )
    assert seen == [1, None]
