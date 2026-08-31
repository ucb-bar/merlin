"""The frozen contract ships twice, and the gate that compares the copies must be able to FAIL.

Which copy is authoritative depends on where you run: ``merlin.common.paths.data_path`` prefers the
in-repo tree when a checkout exists and falls back to the wheel-bundled ``merlin/_data/`` only when it
does not. So a divergence never shows up where the work happens -- every test passes in the checkout
while an installed package enforces a different grammar.

The first version of this gate resolved one side through ``data_path("contract")``, which in a checkout
hands back the repo copy: it compared a tree to itself and reported "35 files identical" with a
deliberate divergence sitting in the packaged copy. These tests exist so that cannot recur -- they build
both trees and inject a difference, rather than trusting the gate's own green output.
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

from merlin.common.paths import repo_root


def _gate():
    path = repo_root() / "build_tools" / "scripts" / "check_contract_copies.py"
    spec = importlib.util.spec_from_file_location("_check_contract_copies", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_repo(tmp_path, *, source: dict, packaged: dict):
    for rel, text in source.items():
        p = tmp_path.joinpath("merlin", "contract", rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    for rel, text in packaged.items():
        p = tmp_path.joinpath("merlin", "python", "merlin", "_data", "contract", rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return tmp_path


class TestTheGateCanFail:
    def test_identical_copies_pass(self, tmp_path):
        root = _fake_repo(tmp_path, source={"abi.yaml": "a: 1\n"}, packaged={"abi.yaml": "a: 1\n"})
        rep = _gate().audit(root)
        assert rep["n_compared"] == 1
        assert not (rep["differing"] or rep["only_in_source"] or rep["only_in_packaged"])

    def test_a_divergence_is_reported(self, tmp_path):
        # THE FALSIFIER. A gate that cannot produce this line establishes nothing, and the first
        # version of this one could not: it compared the checkout tree to itself.
        root = _fake_repo(tmp_path, source={"abi.yaml": "a: 1\n"}, packaged={"abi.yaml": "a: 2\n"})
        assert _gate().audit(root)["differing"] == ["abi.yaml"]

    def test_a_file_only_the_reviewer_sees_is_reported(self, tmp_path):
        # Present in the repo copy, absent from the wheel: the reviewer approves an op the installed
        # package does not have.
        root = _fake_repo(tmp_path, source={"abi.yaml": "a: 1\n", "new.yaml": "b: 1\n"},
                          packaged={"abi.yaml": "a: 1\n"})
        assert _gate().audit(root)["only_in_source"] == ["new.yaml"]

    def test_a_file_only_the_package_has_is_reported(self, tmp_path):
        # The opposite, and the more dangerous direction: nobody reviewing the repo can see it.
        root = _fake_repo(tmp_path, source={"abi.yaml": "a: 1\n"},
                          packaged={"abi.yaml": "a: 1\n", "ghost.yaml": "b: 1\n"})
        assert _gate().audit(root)["only_in_packaged"] == ["ghost.yaml"]

    def test_a_missing_packaged_tree_is_unknown_not_clean(self, tmp_path):
        # An installed wheel would ship no contract at all. Reporting that as "nothing differs" is the
        # check-that-could-not-run-reported-success failure this repo keeps hitting.
        root = _fake_repo(tmp_path, source={"abi.yaml": "a: 1\n"}, packaged={})
        rep = _gate().audit(root)
        assert rep.get("missing_packaged_tree") is True
        assert rep["n_compared"] == 0

    def test_capsules_are_exempt_and_the_reason_is_recorded(self, tmp_path):
        # The two corpora are deliberately different -- the packaged one is curated, and goldens and
        # holdouts must never ship. Requiring equality would demand the answer-key leak the
        # no-answer-keys gate forbids, so the exemption carries its reason as data.
        root = _fake_repo(tmp_path,
                          source={"abi.yaml": "a: 1\n", "capsules/x/golden.yaml": "secret\n"},
                          packaged={"abi.yaml": "a: 1\n"})
        rep = _gate().audit(root)
        assert rep["differing"] == [] and rep["only_in_source"] == []
        assert "capsules" in rep["exempt"] and rep["exempt"]["capsules"]


class TestTheRealTreeAgrees:
    def test_the_shipped_copies_are_in_sync(self):
        rep = _gate().audit()
        if rep.get("missing_packaged_tree"):
            pytest.skip("no packaged contract tree in this checkout")
        assert rep["n_compared"] > 0, "comparing zero files is not a passing gate"
        assert not rep["differing"], rep["differing"]
        assert not rep["only_in_source"], rep["only_in_source"]
        assert not rep["only_in_packaged"], rep["only_in_packaged"]
