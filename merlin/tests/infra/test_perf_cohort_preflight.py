"""A campaign scope that cannot produce a family verdict must be refused before it is launched.

The failure this guards is silent by construction: a cohort one member short of what its analyzer
predeclares still measures every member, still writes every artifact, and simply yields no verdict
for that family. Measured 2026-09-06, a hand-assembled gemmini performance scope would have produced
family verdicts for 14 of 38 members while every family read as "fine" by inspection.
"""
from __future__ import annotations

import importlib.util
import sys

import pytest
import yaml

from merlin.common.paths import merlin_dir

_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
_SOURCE = _SCRIPTS / "perf_cohort_preflight.py"
_SPEC = importlib.util.spec_from_file_location("perf_cohort_preflight_under_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
PRE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = PRE
_SPEC.loader.exec_module(PRE)

_CAPSULES = merlin_dir() / "contract/capsules/_perf"

#: The gemmini performance corpus as the certified campaign scopes it. PC/PL/PQ deliberately share
#: one analyzer, PK and PR each predeclare an exact cohort size, and PM is a complete 4x4 grid.
_FULL = ["PC00_k64", "PC01_k128",
         "PK00_k16", "PK01_k32", "PK02_k64", "PK03_k128",
         "PL00_k16", "PL01_k16", "PL02_k32", "PL03_k32",
         *[f"PM{i:02d}_{n}" for i, n in enumerate(
             ["m16n16", "m16n32", "m16n48", "m16n64", "m32n16", "m32n32", "m32n48", "m32n64",
              "m48n16", "m48n32", "m48n48", "m48n64", "m64n16", "m64n32", "m64n48", "m64n64"])],
         "PQ01_j2_k16", "PQ03_j8_k16", "PQ04_j16_k16",
         "PQ06_j2_k32", "PQ08_j8_k32", "PQ09_j16_k32",
         "PR00_fits_double_k16", "PR01_fits_double_k2048", "PR02_fits_double_k4096",
         "PR03_fits_single_k4112", "PR04_fits_single_k6144", "PR05_fits_single_k8192"]


def _members_file(tmp_path, names):
    path = tmp_path / "members.txt"
    path.write_text(",".join(names), encoding="utf-8")
    return path


def _run(tmp_path, names, root=_CAPSULES):
    return PRE.main(["--members", str(_members_file(tmp_path, names)),
                     "--capsule-root", str(root)])


@pytest.mark.skipif(not _CAPSULES.is_dir(), reason="performance corpus is not present")
def test_the_certified_scope_preflights_ready(tmp_path) -> None:
    """CONTROL. Without this passing, every refusal below proves nothing."""
    assert _run(tmp_path, _FULL) == 0


@pytest.mark.skipif(not _CAPSULES.is_dir(), reason="performance corpus is not present")
@pytest.mark.parametrize("dropped", ["PK00_k16", "PR00_fits_double_k16", "PM00_m16n16"])
def test_one_member_short_is_refused(tmp_path, dropped: str) -> None:
    """MUTATION. Losing a single capture must refuse the launch, not quietly cost a family its
    verdict. Each of these three sits in a differently-shaped cohort: an exact-count sweep (PK), a
    band needing three depths (PR), and a complete grid (PM)."""
    assert _run(tmp_path, [n for n in _FULL if n != dropped]) == 5


@pytest.mark.skipif(not _CAPSULES.is_dir(), reason="performance corpus is not present")
def test_families_sharing_one_analyzer_are_each_decided(tmp_path) -> None:
    """PC, PL and PQ all declare `perf_paired_claim`, whose own precondition check refuses a cohort
    spanning several families. Dispatch stays on the declared analyzer; the cohort handed to it is
    the declared family. Grouping on the analyzer alone reports READY cohorts as REFUSED."""
    paired = [n for n in _FULL if n[:2] in ("PC", "PL", "PQ")]
    assert _run(tmp_path, paired) == 0


def test_a_capsule_declaring_no_analyzer_is_named_not_skipped(tmp_path) -> None:
    """An undecidable member must be REPORTED. Skipping it is how a family ships declaring a claim
    that no code path evaluates -- the defect `perf_claim_dispatch` exists to end."""
    root = tmp_path / "capsules"
    (root / "ZZ00_undeclared").mkdir(parents=True)
    (root / "ZZ00_undeclared" / "capsule.yaml").write_text(
        yaml.safe_dump({"name": "ZZ00_undeclared", "label": "dev",
                        "performance": {"family": "ZZ"}}), encoding="utf-8")
    assert _run(tmp_path, ["ZZ00_undeclared"], root=root) == 5


def test_an_absent_descriptor_refuses_rather_than_measuring_fewer(tmp_path) -> None:
    """A named member with no descriptor must stop the launch. Dropping it silently is the same
    one-member-short failure arriving through a different door."""
    root = tmp_path / "capsules"
    root.mkdir()
    with pytest.raises(SystemExit):
        _run(tmp_path, ["PM00_m16n16"], root=root)
