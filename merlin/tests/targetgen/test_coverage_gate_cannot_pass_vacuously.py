"""A conformance-coverage audit that could not run must never be spelled the same way as a clean one.

Every debt list in `check_conformance_coverage.py` filters on ``status == "ok"``, so a target whose
descriptor could not be resolved contributes no debt and the gate returned 0. Measured: the two saturn
targets reported `no_target_experiment` and the gate exited 0 for BOTH, while a reader took that for
coverage. With the descriptor found by the name it DECLARES they owe 5 uncovered items -- 0/1
composition and 0/4 host-only families.

Two independent defects, so two independent tests: the descriptor lookup was wrong (it searched by
directory name only), and the verdict for "could not audit" was wrong (0 rather than 2). Fixing either
alone still leaves a gate that can pass without asking anything.

This is the fifth instance in this repo of a check that could not run reporting success. It is spelled
2 ("cannot decide") to match `check_pass_obligations.py --fail-on-dead` with no log.
"""
from __future__ import annotations

import importlib.util
import sys

import pytest
import yaml

from merlin.common.paths import repo_root

_TARGETS = repo_root() / "merlin" / "experiments" / "capsule_bench" / "targets"


def _gate():
    p = repo_root() / "build_tools" / "scripts" / "check_conformance_coverage.py"
    spec = importlib.util.spec_from_file_location("_cov_gate2", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_cov_gate2"] = mod
    spec.loader.exec_module(mod)
    return mod


def _declared_but_not_directory() -> list[tuple[str, str]]:
    """``(declared target, directory)`` for every descriptor whose two names differ."""
    out = []
    for desc in sorted(_TARGETS.glob("*/target_experiment.yaml")):
        try:
            doc = yaml.safe_load(desc.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        declared = str(doc.get("target") or "")
        if declared and declared != desc.parent.name:
            out.append((declared, desc.parent.name))
    return out


def test_a_descriptor_is_found_by_the_name_it_declares_not_only_its_directory():
    """The configuration-qualified name is the one every other resolver uses, so it must resolve here."""
    pairs = _declared_but_not_directory()
    if not pairs:
        pytest.skip("no target declares a name differing from its directory")
    gate = _gate()
    for declared, directory in pairs:
        assert gate._target_experiment(declared) is not None, (
            f"{declared!r} (in directory {directory!r}) did not resolve; the gate would report "
            f"`no_target_experiment` and, before this was fixed, exit 0 for it")
        # The directory name must keep working -- it is the common case and the cheap path.
        assert gate._target_experiment(directory) is not None


def test_an_unauditable_target_is_cannot_decide_never_clean():
    gate = _gate()
    rc = gate.main(["--target", "definitely_not_a_target", "--fail-on-uncovered"])
    assert rc == 2, f"an unauditable target must exit 2 (cannot decide), got {rc}"
    assert gate.main(["--target", "definitely_not_a_target", "--fail-on-unverifiable"]) == 2


def test_the_saturn_targets_now_derive_a_real_requirement():
    """The regression this pair of defects hid: a target with a requirement, reported as having none."""
    pairs = _declared_but_not_directory()
    if not pairs:
        pytest.skip("no target declares a name differing from its directory")
    gate = _gate()
    for declared, _directory in pairs:
        row = gate.audit(declared)
        assert row["status"] != "no_target_experiment", (
            f"{declared}: {row.get('detail')}")
        # Establishing SOMETHING is the point; whether it is covered is the ratchet's business.
        n_cells = len(row.get("required") or row.get("cells") or ())
        assert n_cells or row.get("composition") or row.get("host_only"), (
            f"{declared} derived no requirement on any axis, so its verdict would be vacuous: {row}")
