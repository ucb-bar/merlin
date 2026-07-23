"""The 4th entrypoint is target-neutral: emit_target_artifact <-> lower_target_to_llvm resolve to
whichever spelling a package declares, so old (LLVM/RoCC) and new (SIMT/other) packages both work.
"""
from __future__ import annotations

import types
from pathlib import Path

from merlin.targetgen import oot_runner as OR


def _pkg(commands: dict):
    return types.SimpleNamespace(manifest={"commands": commands}, tool="/bin/true")


def test_new_name_resolves_a_legacy_package():
    # a package that declares the LEGACY key still runs when the runner asks for the NEW name
    pkg = _pkg({"lower_target_to_llvm": {"argv": ["{tool}", "--legacy", "{input_mlir}"]}})
    argv = OR._resolve_argv(pkg, "emit_target_artifact", Path("in.mlir"), None)
    assert argv[1] == "--legacy"


def test_legacy_name_resolves_a_new_package():
    # a package that declares the NEW key still runs when a legacy caller asks for lower_target_to_llvm
    pkg = _pkg({"emit_target_artifact": {"argv": ["{tool}", "--new", "{input_mlir}"]}})
    argv = OR._resolve_argv(pkg, "lower_target_to_llvm", Path("in.mlir"), None)
    assert argv[1] == "--new"


def test_exact_name_wins_over_alias():
    pkg = _pkg({"emit_target_artifact": {"argv": ["{tool}", "--exact", "{input_mlir}"]},
                "lower_target_to_llvm": {"argv": ["{tool}", "--other", "{input_mlir}"]}})
    assert OR._resolve_argv(pkg, "emit_target_artifact", Path("i"), None)[1] == "--exact"
