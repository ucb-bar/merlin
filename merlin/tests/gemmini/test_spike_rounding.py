"""Rounding contract for the Spike Gemmini model: accumulator scaling is true round-near-even.

Spike's libgemmini once computed the round-up candidate of ROUND_NEAR_EVEN in float, so x = pred(2^e)
rounded to 2^e + 1 instead of 2^e (664 * f32(1/166) gave 5 where the RTL gives 4). The first group of
tests is pure host arithmetic and states the contract; the second runs a tiny bare-metal case on Spike
against operator-provided extensions and is skipped when they are absent. Locations come from the
environment, never from a path baked into the test:

    MERLIN_TEST_SPIKE            spike binary
    MERLIN_TEST_RISCV_GCC        riscv bare-metal gcc
    MERLIN_TEST_ROCC_TESTS       gemmini-rocc-tests checkout (include/gemmini.h)
    MERLIN_TEST_RNE_EXTLIB       patched libgemmini.so (the registry's gemmini_spike_extension_rne)
    MERLIN_TEST_BASELINE_EXTLIB  unpatched libgemmini.so (optional; must give the wrong answer)
"""

from __future__ import annotations

import os
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from merlin.common import provenance as PROV  # noqa: E402
from merlin.common.paths import merlin_dir, repo_root  # noqa: E402


def _true_rne(x) -> int:
    """Round-half-to-even of the EXACT value of a float32 (Fraction rounds half to even)."""
    return round(Fraction(float(x)))


def _old_model(x):
    """The defective macro: the up/down candidate is computed in float32 (`x_ + 1`)."""
    x = np.float32(x)
    i = int(x)
    nxt = int(x - np.float32(1) if x < 0 else x + np.float32(1))
    rem = abs(np.float32(x - np.float32(i)))
    if rem < np.float32(0.5):
        return i
    if rem > np.float32(0.5):
        return nxt
    return i if i % 2 == 0 else nxt


def _fixed_model(x):
    """The patched macro: the candidate is the integer part plus or minus one."""
    x = np.float32(x)
    i = int(x)
    nxt = i - 1 if x < 0 else i + 1
    rem = abs(np.float32(x - np.float32(i)))
    if rem < np.float32(0.5):
        return i
    if rem > np.float32(0.5):
        return nxt
    return i if i % 2 == 0 else nxt


def _just_below_powers_of_two():
    for e in range(1, 24):
        for sign in (1.0, -1.0):
            yield sign * float(np.nextafter(np.float32(2.0**e), np.float32(0)))


def test_the_reported_case_is_just_below_four():
    x = np.float32(664) * np.float32(1.0 / 166.0)
    assert float(x) < 4.0 and _true_rne(x) == 4


def test_patched_rounding_is_true_rne_just_below_powers_of_two():
    for x in _just_below_powers_of_two():
        assert _fixed_model(x) == _true_rne(x), x


def test_old_rounding_missed_those_cases():
    """Documents the defect: the float candidate lands one past 2^e for some predecessor values."""
    bad = [x for x in _just_below_powers_of_two() if _old_model(x) != _true_rne(x)]
    assert bad, "the defect model no longer reproduces; the regression test would be vacuous"


def test_ties_still_go_to_even():
    for x, want in ((0.5, 0), (1.5, 2), (2.5, 2), (-0.5, 0), (-1.5, -2), (-2.5, -2)):
        assert _fixed_model(x) == want == _true_rne(x)


def test_registry_declares_the_patched_extension_as_a_new_identity():
    arts = PROV.load_artifacts()
    art = arts["gemmini_spike_extension_rne"]
    assert art.role == "spike_extension" and art.built_from == ("libgemmini_rne",)
    pin = PROV.load_pins()["libgemmini_rne"]
    assert pin.commit.startswith("fce5cae")
    v1_prefix = "ec6a6f47"
    assert not art.digest.startswith(v1_prefix)
    assert not any(a.digest.startswith(v1_prefix) and n == art.name for n, a in arts.items())


def test_template_lets_a_seal_declare_the_extension_by_digest():
    template = repo_root() / "examples/gemmini/phase0/execution-declaration.template.yaml"
    text = template.read_text(encoding="utf-8")
    art = PROV.load_artifacts()["gemmini_spike_extension_rne"]
    assert "MERLIN_GEMMINI_RNE_SPIKE_EXTLIB" in text and art.digest in text


# ---------------------------------------------------------------------------------------------------
# Spike
# ---------------------------------------------------------------------------------------------------
def _tool(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value and Path(value).exists() else None


_SPIKE, _GCC, _ROCC = (_tool(n) for n in ("MERLIN_TEST_SPIKE", "MERLIN_TEST_RISCV_GCC", "MERLIN_TEST_ROCC_TESTS"))
_RNE, _BASE = _tool("MERLIN_TEST_RNE_EXTLIB"), _tool("MERLIN_TEST_BASELINE_EXTLIB")
_need = pytest.mark.skipif(not all((_SPIKE, _GCC, _ROCC, _RNE)), reason="spike toolchain / extensions not provided")


@pytest.fixture(scope="module")
def probe_elf(tmp_path_factory):
    out = tmp_path_factory.mktemp("rne") / "probe.elf"
    src = merlin_dir() / "tests/data/gemmini/rne_scale_probe.c"
    cmd = [str(_GCC), "-O1", "-static", "-specs=htif_nano.specs", "-DBAREMETAL=1", f"-I{_ROCC}",
           f"-I{_ROCC}/rocc-software/src", "-o", str(out), str(src)]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    return out


def _run(elf: Path, extlib: Path) -> int:
    env = dict(os.environ, PATH=f"{_SPIKE.parent}:{os.environ.get('PATH', '')}")
    env["LD_LIBRARY_PATH"] = f"{extlib.parent}:{_SPIKE.parent.parent / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    cmd = [str(_SPIKE), "--isa=rv64gc", f"--extlib={extlib}", "--extension=gemmini", str(elf)]
    return subprocess.run(cmd, env=env, capture_output=True, timeout=120).returncode


@_need
def test_patched_extension_rounds_664_over_166_to_4(probe_elf):
    assert _run(probe_elf, _RNE) == 4


@_need
@pytest.mark.skipif(_BASE is None, reason="no unpatched baseline extension provided")
def test_unpatched_extension_fails_the_same_probe(probe_elf):
    assert _run(probe_elf, _BASE) == 5
