"""Drive the lit pass-test suite from pytest, so it runs in the normal test job and in CI.

The suite itself lives in ``merlin/tests/data/lit`` as ``.mlir`` files with ``// RUN:`` lines — the
standard MLIR pass-test shape. This module is the bridge that makes it a first-class part of
``pytest merlin/tests``.

The skip is deliberate and loud: a missing FileCheck/llvm-lit must produce a SKIP with the reason,
never a pass. A verification suite that could not run and reported green is the failure mode this
whole layer exists to prevent.
"""
from __future__ import annotations

import subprocess

import pytest

from merlin.common.paths import merlin_dir
from merlin.verify import tools

_LIT = tools.find_lit()
_FC = tools.find_filecheck()
_MO = tools.find_mlir_tool("mlir-opt")

pytestmark = pytest.mark.skipif(
    not (_LIT and _FC and _MO),
    reason=f"lit suite needs llvm-lit/FileCheck/mlir-opt (found: {tools.availability()})")


def _suite_dir():
    return merlin_dir() / "tests" / "data" / "lit"


def _run_lit(*extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([_LIT, "-sv", *extra, str(_suite_dir())],
                          capture_output=True, text=True, timeout=600)


def test_lit_suite_passes():
    r = _run_lit()
    assert r.returncode == 0, f"lit suite failed:\n{r.stdout}\n{r.stderr}"


def test_suite_is_not_empty():
    """A green run over zero tests is not evidence of anything."""
    r = _run_lit()
    out = r.stdout + r.stderr
    assert "Total Discovered Tests:" in out, out
    n = int(out.split("Total Discovered Tests:")[1].split()[0])
    assert n >= 5, f"expected the seed suite to be discovered, found {n}"


def test_negative_tests_are_present():
    """At least one test must assert a REJECTION.

    A suite made only of positive checks cannot distinguish a working verifier from one that accepts
    everything — which is exactly what the frozen grammar's IRDL was doing before the sigil
    normalization landed.

    Two spellings count, and the second is the better one. `RUN: not <tool> | FileCheck` passes when an
    error appears anywhere in the output, so it survives the constraint it names being removed as long
    as SOMETHING still errors. `-verify-diagnostics` binds each expected diagnostic to a line and a
    message and fails on unexpected ones too. The iface negatives were converted to it on 2026-09-05;
    this assertion accepts both so a file in either style still counts as a control.
    """
    negatives = [p for p in _suite_dir().rglob("*.mlir")
                 if any(mark in p.read_text(encoding="utf-8")
                        for mark in ("RUN: not ", "-verify-diagnostics"))]
    assert negatives, "the suite has no negative control"


def test_a_verify_diagnostics_file_expects_a_diagnostic_on_a_line():
    """`-verify-diagnostics` with no `expected-error` is vacuous: it asserts the input is CLEAN.

    The flag makes mlir-opt fail on any diagnostic it was not told to expect, so a file carrying the
    flag and no expectation is a positive test wearing a negative test's clothes — it would pass on the
    day the verifier stops rejecting anything. This pairs the flag with at least one expectation.
    """
    offenders = []
    for path in _suite_dir().rglob("*.mlir"):
        text = path.read_text(encoding="utf-8")
        if "-verify-diagnostics" in text and "expected-error" not in text:
            offenders.append(path.name)
    assert not offenders, (
        f"{offenders} run with -verify-diagnostics but expect no diagnostic; such a file asserts the "
        f"input is clean, which is the opposite of a negative control")


# --- mutation control: are the CHECK lines load-bearing? ------------------------------------------
# A suite that passes proves nothing unless it also FAILS on wrong output. These run the real pass,
# mutate its result the way a miscompiling backend would, and require FileCheck to reject it. If one
# of these starts passing, the corresponding CHECK lines have gone decorative.

def _filecheck(check_file, text: str) -> subprocess.CompletedProcess:
    return subprocess.run([_FC, str(check_file)], input=text, capture_output=True, text=True)


def _residency_output() -> tuple[str, object]:
    """Real output of merlin-materialize-interface on the seed input, plus its check file."""
    import sys

    src = _suite_dir() / "core" / "materialize_interface_residency.mlir"
    env = {"PYTHONPATH": str(merlin_dir() / "python"), "PATH": "/usr/bin:/bin",
           "HOME": "/tmp"}
    r = subprocess.run([sys.executable, "-m", "merlin.xdsl_dialects.opt", str(src),
                        "-p", "merlin-materialize-interface"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    return r.stdout, src


def test_unmutated_output_passes_its_checks():
    out, src = _residency_output()
    assert _filecheck(src, out).returncode == 0, "the honest baseline must pass"


@pytest.mark.parametrize("name,mutate", [
    # A weight packed twice: residency was never established, so the second pack silently re-uploads
    # and the reuse the schedule proved is gone.
    ("packed_twice",
     lambda t: t.replace('"interface.resident_pack"', '"interface.resident_pack"', 1).replace(
         "    %5 =", '    %99 = "interface.resident_pack"(%4) : (tensor<128x64xi8>) -> ()\n    %5 =', 1)),
    # Evicted before the last use: the classic use-after-evict that only shows up as a wrong number.
    ("evicted_early",
     lambda t: _move_evict_first(t)),
])
def test_mutated_output_is_rejected(name, mutate):
    out, src = _residency_output()
    mutated = mutate(out)
    assert mutated != out, f"mutation {name!r} did not change the IR — the test would be vacuous"
    r = _filecheck(src, mutated)
    assert r.returncode != 0, (
        f"FileCheck ACCEPTED the {name!r} mutation — those CHECK lines are not load-bearing\n"
        f"{mutated}")


def _move_evict_first(text: str) -> str:
    lines = text.splitlines()
    ev = [i for i, l in enumerate(lines) if "interface.resident_evict" in l]
    mm = [i for i, l in enumerate(lines) if "interface.matmul" in l]
    if not ev or not mm:
        return text
    line = lines.pop(ev[0])
    return "\n".join(lines[:mm[0]] + [line] + lines[mm[0]:]) + "\n"


def test_the_core_suite_credits_a_PRODUCTION_pass(tmp_path, monkeypatch):
    """The obligation gate must be able to read 1/4, not 0/4.

    The derived per-target suite can only credit `merlin-materialize-interface`, which lives in the
    PROTOTYPE catalog, so the gate correctly refused to count it toward production and reported
    `0 / 4 verified` — "we verify a pass", not "we verify the compiler". The core suite drives
    production passes and nothing was reading it.
    """
    from merlin.targetgen.lit_suite import record_core_verdicts
    from merlin.xdsl_dialects.lowering import passes as P

    log = tmp_path / "verify.jsonl"
    monkeypatch.setenv("MERLIN_VERIFY_LOG", str(log))
    verdicts = record_core_verdicts(lit_passed=True)

    production = {i.name for i in P.production_catalog()} if hasattr(P, "production_catalog") \
        else {i.name for i in P.CATALOG}
    credited = {n for n, v in verdicts.items() if v == "verified"} & production
    assert credited, (
        f"no PRODUCTION pass is credited by the core suite; verdicts={verdicts}. Without one the "
        f"obligation gate can only ever report 0/4 verified for production.")
    assert log.is_file(), "verdicts were computed but nothing reached the log the gate reads"


def test_a_run_line_that_stops_exercising_a_pass_stops_crediting_it(tmp_path):
    """The mapping is read from the RUN line, so it cannot drift from what the test runs."""
    from merlin.targetgen.lit_suite import passes_exercised_by

    f = tmp_path / "t.mlir"
    f.write_text("// RUN: %merlin-opt %s -p merlin-add-c-interface | %filecheck %s\n")
    assert passes_exercised_by(f) == ["merlin-add-c-interface"]

    f.write_text("// RUN: %merlin-opt %s | %filecheck %s\n")
    assert passes_exercised_by(f) == [], "a RUN line with no -p must credit no pass"


def test_a_failing_core_suite_is_refuted_never_silently_unverified():
    """A red suite is a disproof, and the gate treats `refuted` as unratchetable."""
    from merlin.targetgen.lit_suite import record_core_verdicts

    assert set(record_core_verdicts(lit_passed=False).values()) <= {"refuted", "unknown-pass"}
    assert set(record_core_verdicts(lit_passed=None).values()) <= {"unmeasured", "unknown-pass"}
