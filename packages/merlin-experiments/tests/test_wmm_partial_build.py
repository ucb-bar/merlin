"""A measured run refuses a PARTIAL (``only_groups``) build before any machine time.

The builder marks a partial program's expectations; the service's normalization reads them first, so a
launch config that asks for ``only_groups`` gets a refusal naming why, never a measurement of a program
that is mostly the target's library."""

from __future__ import annotations

import hashlib

import pytest
from merlin_experiments.phase2.whole_model_measured.identity import normalize_build_record

from merlin.perf import whole_model_verdict as V
from merlin.perf.whole_model_partial import MARKER


def _record(tmp_path, **expectations):
    elf = tmp_path / "program.elf"
    elf.write_bytes(b"\x7fELF fixture")
    return {
        "elf": str(elf),
        "elf_sha256": hashlib.sha256(elf.read_bytes()).hexdigest(),
        "parameter_header_sha256": "a" * 64,
        "expectations": {
            "groups": {"1": {"compare": "exact", "sum": 1, "fnv1a": 2, "inputs_from": []}},
            "argmax": 0,
            **expectations,
        },
    }


def test_a_whole_build_is_admitted_and_a_partial_one_is_refused(tmp_path):
    assert normalize_build_record(_record(tmp_path), package_sha256="p")["elf_sha256"]
    with pytest.raises(V.VerdictRefusal, match="partial build"):
        normalize_build_record(_record(tmp_path, **{MARKER: {"only_groups": ["g1"]}}), package_sha256="p")
