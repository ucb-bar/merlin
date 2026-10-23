"""The frontend export owns the quant vocabulary, including its pinned revision."""

import hashlib
import json
import os
import subprocess
import warnings
from pathlib import Path

import pytest

from merlin.common.paths import contract_dir, repo_root
from merlin.targetgen.core_aten_spec_int8 import quantization_schema


def test_exactness_fixture_selects_the_current_capture_revision():
    fixture = json.loads((repo_root() / "merlin/tests/data/exactness/bounded_final_gate.json").read_text())
    pin = json.loads((contract_dir() / "quant_schema_source.json").read_text())
    assert fixture["model2mlir_revision"] == pin["revision"]


def test_quant_schema_matches_pinned_frontend_and_live_export():
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("MERLIN_M2M_PYTHON is required to check the real frontend export")
    vendor = (contract_dir() / "quant_schema.json").read_bytes()
    pin = json.loads((contract_dir() / "quant_schema_source.json").read_text())
    assert hashlib.sha256(vendor).hexdigest() == pin["sha256"]
    from merlin.common.yaml import load_yaml

    capture_pin = load_yaml(contract_dir() / "software_pins.yaml")["pins"]["model2mlir_capture"]
    assert pin["revision"] == capture_pin["commit"]
    result = subprocess.run(
        [python, "-c", "import pathlib,m2m; print(pathlib.Path(m2m.__file__).resolve().parent.parent)"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    root = result.stdout.strip().splitlines()[-1]
    if (Path(root) / ".git").exists():
        observed = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        assert observed.stdout.strip() == pin["revision"], "MERLIN_M2M_PYTHON must use the pinned test snapshot"
        dirty = subprocess.run(
            ["git", "-C", root, "status", "--porcelain", "--untracked-files=normal", "--", "m2m"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert not dirty.stdout.strip(), "MERLIN_M2M_PYTHON imports a modified frontend; use a clean pinned snapshot"
        pinned = subprocess.run(["git", "-C", root, "show", pin["revision"] + ":" + pin["path"]], capture_output=True)
        if pinned.returncode:
            warnings.warn(
                "QUANT SCHEMA PIN INCOMPLETE: model2mlir export is not committed at "
                + pin["revision"]
                + "; live-export equality is checked, but revision attribution remains unqualified.",
                RuntimeWarning,
                stacklevel=1,
            )
            assert pin["status"] == "exported_uncommitted"
        else:
            assert pinned.stdout == vendor, "vendored quant schema drifts from the pinned model2mlir revision"
    else:
        from pinned_exports import export_digest

        fixture = json.loads((repo_root() / "merlin/tests/data/exactness/bounded_final_gate.json").read_text())
        exported = fixture["source_exports"]["model2mlir"]
        assert exported["revision"] == pin["revision"]
        assert export_digest(root) == exported["sha256"], "pinned frontend source export changed"
    assert (Path(root) / pin["path"]).read_bytes() == vendor, "live frontend quant schema drift"
    schema = quantization_schema()
    assert schema["schema_version"] == 1
    assert schema["unknown_keys"] == "reject"
    assert "prov.quant.host_only" in schema["attributes"]
    assert "prov.quant.scale_numerator" in schema["attributes"]
    assert "prov.quant.qmin" not in schema["attributes"]
