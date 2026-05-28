"""IREE submodule consistency tests.

Cross-check the planner's predicted ``third_party/iree_bar/...`` paths
against the live IREE submodule:

* The submodule HEAD must be a descendant of ``IREE_UPSTREAM_BASE`` (this
  is also enforced by ``./merlin ci patch-gate`` but we mirror it here so
  pytest catches regressions before CI does).
* Every ``write_path`` the modification map predicts under
  ``third_party/iree_bar/`` must point at a directory or file that exists
  in the submodule. This catches stale planner predictions after an IREE
  rebase moves source around.
* Every ``write_path`` under the nested LLVM submodule must be inside
  ``third_party/iree_bar/third_party/llvm-project/`` (not some other
  LLVM tree). The rebase order is documented in
  ``docs/architecture/plugin_and_patch_model.md``: LLVM first, then IREE.

Marker: ``integration``. Skips cleanly if the submodule is uninitialised.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from targetgen import build_support_plan, load_capability_spec  # noqa: E402
from targetgen.stage_map import build_modification_map  # noqa: E402

MANIFEST = REPO_ROOT / "build_tools" / "patches" / "manifest.env"
IREE_ROOT = REPO_ROOT / "third_party" / "iree_bar"
LLVM_ROOT = IREE_ROOT / "third_party" / "llvm-project"

pytestmark = [pytest.mark.integration]


def _read_manifest() -> dict[str, str]:
    text = MANIFEST.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^\s*([A-Z_][A-Z0-9_]*)=\"([^\"]*)\"\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)


@pytest.fixture(scope="module")
def manifest() -> dict[str, str]:
    if not MANIFEST.exists():
        pytest.skip(f"manifest.env not found at {MANIFEST}")
    return _read_manifest()


@pytest.fixture(scope="module")
def iree_initialised() -> Path:
    if not IREE_ROOT.exists():
        pytest.skip(f"IREE submodule not present at {IREE_ROOT}")
    if not (IREE_ROOT / "CMakeLists.txt").exists():
        pytest.skip(f"IREE submodule appears uninitialised at {IREE_ROOT} (no CMakeLists.txt).")
    return IREE_ROOT


REQUIRED_MANIFEST_KEYS: tuple[str, ...] = (
    "IREE_UPSTREAM_REPO",
    "IREE_UPSTREAM_BASE",
    "IREE_FORK_REPO",
    "IREE_FORK_BRANCH",
    "LLVM_UPSTREAM_BASE",
)


def test_manifest_declares_required_keys(manifest: dict[str, str]) -> None:
    for required in REQUIRED_MANIFEST_KEYS:
        assert required in manifest, f"manifest.env missing {required}"


def test_iree_submodule_head_descends_from_upstream_base(iree_initialised: Path, manifest: dict[str, str]) -> None:
    """Mirrors `./merlin ci patch-gate` ancestry check."""
    base = manifest["IREE_UPSTREAM_BASE"]
    # Verify the base commit is reachable in the submodule (skip if the
    # local clone is shallow).
    rev_parse = _git("rev-parse", "--verify", f"{base}^{{commit}}", cwd=iree_initialised)
    if rev_parse.returncode != 0:
        pytest.skip(
            f"IREE_UPSTREAM_BASE {base[:8]} not present in local IREE clone "
            f"(probably a shallow checkout). Refresh via: "
            f"git -C {iree_initialised} fetch upstream {base}"
        )
    head = _git("rev-parse", "HEAD", cwd=iree_initialised).stdout.strip()
    descends = _git("merge-base", "--is-ancestor", base, head, cwd=iree_initialised)
    assert descends.returncode == 0, (
        f"IREE submodule HEAD {head[:8]} does not descend from {base[:8]}; "
        f"the rebase order in docs/architecture/plugin_and_patch_model.md is broken"
    )


def test_iree_patch_predictions_point_at_real_paths(iree_initialised: Path) -> None:
    """Whenever the planner predicts iree_bar paths, check they resolve."""
    examples = REPO_ROOT / "target_specs" / "examples"
    bad: list[tuple[str, str, str]] = []  # (target, stage, path)
    for cap in sorted(examples.glob("*/capability.yaml")):
        capabilities = load_capability_spec(cap)
        plan = build_support_plan(capabilities)
        modmap = build_modification_map(capabilities, targetgen_styles=plan.integration_styles)
        target = cap.parent.name
        for stage in modmap.stages:
            for write in stage.write_paths:
                if not write.startswith("third_party/iree_bar/"):
                    continue
                # Strip the submodule prefix and check the relative path.
                rel = write.removeprefix("third_party/iree_bar/")
                full = IREE_ROOT / rel
                # Predicted-new paths inside IREE end in '/' or contain a
                # variable segment; we only check that the parent directory
                # exists. Strip trailing slash.
                check_path = full
                if not check_path.exists():
                    parent = check_path.parent
                    if not parent.exists():
                        bad.append((target, stage.stage, write))
    assert not bad, "TargetGen predicts edits in IREE submodule that have no real parent:\n" + "\n".join(
        f"  [{t}/{s}] {p}" for t, s, p in bad
    )


def test_llvm_predictions_stay_inside_iree_bars_llvm(iree_initialised: Path) -> None:
    """LLVM edits the planner suggests must live inside the IREE-pinned LLVM tree."""
    examples = REPO_ROOT / "target_specs" / "examples"
    for cap in sorted(examples.glob("*/capability.yaml")):
        capabilities = load_capability_spec(cap)
        plan = build_support_plan(capabilities)
        modmap = build_modification_map(capabilities, targetgen_styles=plan.integration_styles)
        for stage in modmap.stages:
            for write in stage.write_paths:
                if "/llvm-project" not in write and "/llvm/" not in write:
                    continue
                # Any LLVM-shaped path must be under the IREE-bundled LLVM.
                assert write.startswith("third_party/iree_bar/third_party/llvm-project"), (
                    f"{cap.parent.name}/{stage.stage}: predicts LLVM edit outside "
                    f"the IREE-bundled LLVM submodule: {write!r}"
                )


def test_no_predicted_path_collides_with_manifest_pinned_files(
    iree_initialised: Path,
) -> None:
    """No write_path should target the manifest itself (operators only)."""
    examples = REPO_ROOT / "target_specs" / "examples"
    forbidden = {"build_tools/patches/manifest.env", "build_tools/patches/"}
    bad: list[tuple[str, str]] = []
    for cap in sorted(examples.glob("*/capability.yaml")):
        capabilities = load_capability_spec(cap)
        plan = build_support_plan(capabilities)
        modmap = build_modification_map(capabilities, targetgen_styles=plan.integration_styles)
        for stage in modmap.stages:
            for write in stage.write_paths:
                if any(write.startswith(p) for p in forbidden):
                    bad.append((cap.parent.name, write))
    assert not bad, "Planner predicts edits to operator-managed manifest paths:\n  " + "\n  ".join(
        f"{t}: {p}" for t, p in bad
    )
