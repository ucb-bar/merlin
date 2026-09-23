"""The actual layout gate follows declared nested product homes with bounded scans."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from merlin.common.paths import repo_root


@pytest.mark.parametrize("relative", ["perf-studies/ledger/v2", "perf-studies/ledger/fixture/v2", "compare/fixture/v2"])
def test_declared_product_versions_require_manifest_and_relative_latest(tmp_path, relative):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    contract = tmp_path / "merlin/contract/storage.yaml"
    contract.parent.mkdir(parents=True)
    contract.write_text(
        "concerns: {perf-studies: 'Studies', compare: 'Comparisons'}\n"
        "product_roots: {perf-ledger: perf-studies/ledger}\n"
    )
    version = tmp_path / "out/artifacts" / relative
    product = version / "product"
    product.mkdir(parents=True)
    latest = version / "latest"
    latest.symlink_to(product)
    gate = repo_root() / "build_tools/scripts/check_artifact_layout.py"

    def check():
        return subprocess.run([sys.executable, str(gate)], cwd=tmp_path, capture_output=True, text=True, timeout=15)

    failed = check()
    assert failed.returncode == 1
    assert "product dir missing manifest.yaml" in failed.stderr
    assert "absolute `latest` symlink" in failed.stderr
    (product / "manifest.yaml").write_text("schema: fixture\n")
    latest.unlink()
    latest.symlink_to(product.name)
    assert check().returncode == 0
    latest.unlink()
    latest.symlink_to("missing")
    assert "dangling `latest` symlink" in check().stderr


def test_scan_does_not_descend_past_declared_axes_or_treat_verify_as_version(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    contract = tmp_path / "merlin/contract/storage.yaml"
    contract.parent.mkdir(parents=True)
    contract.write_text(
        "concerns: {perf-studies: 'Studies', probes: 'Probes'}\nproduct_roots: {perf-ledger: perf-studies/ledger}\n"
    )
    for relative in ("probes/verify/unit", "perf-studies/ledger/fixture/extra/v2/unit"):
        (tmp_path / "out/artifacts" / relative).mkdir(parents=True)
    gate = repo_root() / "build_tools/scripts/check_artifact_layout.py"
    checked = subprocess.run([sys.executable, str(gate)], cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert checked.returncode == 0, checked.stdout + checked.stderr


@pytest.mark.parametrize("document", ["[]", "broken: [", "product_roots: {perf-ledger: ../outside}"])
def test_invalid_storage_declaration_blocks_stop_hook(tmp_path, document):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    contract = tmp_path / "merlin/contract/storage.yaml"
    contract.parent.mkdir(parents=True)
    contract.write_text(document)
    (tmp_path / "out/artifacts").mkdir(parents=True)
    gate = repo_root() / "build_tools/scripts/check_artifact_layout.py"
    checked = subprocess.run(
        [sys.executable, str(gate), "--stop-hook"], cwd=tmp_path, capture_output=True, text=True, timeout=15
    )
    assert checked.returncode == 0
    assert json.loads(checked.stdout)["decision"] == "block"
