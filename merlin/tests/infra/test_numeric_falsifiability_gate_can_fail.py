from __future__ import annotations

import importlib.util
import sys

import pytest
import yaml

from merlin.common.paths import repo_root

SCRIPTS = repo_root() / "build_tools" / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FALSIFIABILITY = _load("check_numeric_falsifiability")


def _corpus(tmp_path, atol: float):
    """A one-capsule corpus whose golden is DERIVED, like almost every capsule in the real tree.

    Deliberately not a stored ``golden.yaml``: the defect being proved against is a check that only
    looked at capsules carrying one, so a fixture that shipped an answer key would exercise the very
    path that was never the problem.
    """
    directory = tmp_path / "capsules" / "TT0_two_element_sum"
    directory.mkdir(parents=True, exist_ok=True)
    capsule = {
        "name": "TT0_two_element_sum",
        "kind": "isa",
        "source_role": "derived_sweep",
        "source_reference": "a two-element elementwise sum, small enough that its golden is obvious",
        "label": "public",
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [
            {"name": "X0", "role": "input", "shape": [1, 4], "dtype": "i8"},
            {"name": "X1", "role": "input", "shape": [1, 4], "dtype": "i8"},
        ],
        "operation": {
            "op": "residual_add",
            "attributes": {
                "lhs": "X0",
                "rhs": "X1",
                "out": "Y0",
                "epilogue": [],
                "lhs_scale": 1.0,
                "rhs_scale": 1.0,
                "bound_lsb": 0,
                "output_dtype": "i8",
            },
        },
        "numeric_policy": {"compare": "tolerance_float", "dtype": "i8", "atol": atol, "rtol": 0.0},
        "expected": {"instruction_classes": [], "modes": {"relu": False}},
        "required_oracle_tiers": ["L0"],
        "vcs": "optional",
        "firesim": "optional",
    }
    (directory / "capsule.yaml").write_text(yaml.safe_dump(capsule, sort_keys=False), encoding="utf-8")
    (directory / "capsule.interface.mlir").write_text("module {}\n", encoding="utf-8")
    return directory.parent


@pytest.fixture
def golden_spread(tmp_path):
    """The golden this corpus derives, and the half-spread a constant answer must miss it by."""
    corpus = _corpus(tmp_path, atol=0.0)
    from merlin.targetgen import capsule_common as CC
    from merlin.targetgen import capsule_golden as CG

    directory = next(corpus.iterdir())
    values = CG.golden(CC.load_capsule(directory), directory)
    flat = [float(v) for output in values.values() for row in output for v in (row if isinstance(row, list) else [row])]
    if max(flat) == min(flat):
        pytest.skip("this fixture's stimulus produced a flat golden; it cannot prove a spread rule")
    return 0.5 * (max(flat) - min(flat))


def test_the_verdict_and_not_only_the_finding_changes(tmp_path, golden_spread):
    """A finding nothing turns into an exit code is the defect one level up: the audit existed and the
    two offenders shipped anyway. So the gate's own entry point must return 1 on the mutation and 0
    on the clean corpus."""
    assert FALSIFIABILITY.main(["--root", str(_corpus(tmp_path / "bad", atol=golden_spread * 2.0))]) == 1
    assert FALSIFIABILITY.main(["--root", str(_corpus(tmp_path / "ok", atol=golden_spread * 0.5))]) == 0
