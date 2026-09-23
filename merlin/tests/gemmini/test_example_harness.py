"""Example-owned harness and retained aliases; no compiler or accelerator execution."""

import os
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.sandbox.toolchain import curated_harness_dir
from merlin.targetgen.target_experiment import load_target_experiment

pytestmark = pytest.mark.target("gemmini")


def test_descriptor_uses_example_owned_curated_harness():
    root = repo_root()
    descriptor = load_target_experiment(root / "examples/gemmini/target/descriptor.yaml")
    selected = root / "examples/gemmini/phase1/contracts/harness_curated/gemmini-rocc-tests"
    assert curated_harness_dir(descriptor) == str(selected)
    assert (selected / "include/gemmini.h").is_file()
    assert (selected / "rocc-software/LICENSE").is_file()
    for mode in ("pm", "pt", "v"):
        link = selected / f"riscv-tests/env/{mode}/link.ld"
        assert link.is_symlink() and os.readlink(link) == "../p/link.ld"
        assert link.resolve(strict=True) == selected / "riscv-tests/env/p/link.ld"


def test_legacy_and_pinned_batch_aliases_reach_same_harness():
    root = repo_root()
    selected = root / "examples/gemmini/phase1/contracts/harness_curated"
    legacy = root / "merlin/experiments/capsule_bench/targets/gemmini/contracts/harness_curated"
    assert legacy.is_symlink()
    assert legacy.resolve(strict=True) == selected
    pinned = root / "merlin/experiments/capsule_bench/targets/gemmini_g3arm97/contracts/harness_curated"
    assert pinned.resolve(strict=True) == selected


def test_universal_shared_links_move_but_parameter_override_does_not():
    root = repo_root()
    universal = root / (
        "merlin/experiments/capsule_bench/targets/gemmini_universal/contracts/harness_curated/gemmini-rocc-tests"
    )
    selected = root / "examples/gemmini/phase1/contracts/harness_curated/gemmini-rocc-tests"
    shared = [path for path in (universal / "include").iterdir() if path.name != "gemmini_params.h"]
    shared.extend(universal / name for name in ("riscv-tests", "rocc-software"))
    assert len(shared) == 17
    for link in shared:
        assert link.is_symlink()
        lexical = Path(os.path.normpath(link.parent / os.readlink(link)))
        assert lexical == selected / link.relative_to(universal)
        assert link.exists()
    override = universal / "include/gemmini_params.h"
    assert os.readlink(override) == "../../../../../../../../targets/gemmini_universal/contracts/abi/gemmini_params.h"
