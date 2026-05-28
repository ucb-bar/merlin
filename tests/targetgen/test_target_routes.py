from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from targetgen import build_support_plan, load_capability_spec  # noqa: E402
from targetgen.stage_map import build_modification_map  # noqa: E402

EXAMPLES = REPO_ROOT / "target_specs" / "examples"


def _modmap(target: str):
    caps = load_capability_spec(EXAMPLES / target / "capability.yaml")
    styles = build_support_plan(caps).integration_styles
    return build_modification_map(caps, targetgen_styles=styles)


def _stage(modmap, name: str):
    return next(s for s in modmap.stages if s.stage == name)


def test_riscv_route_extends_executable_blocking_questions() -> None:
    modmap = _modmap("saturn_opu_v128")
    exe = _stage(modmap, "executable_sources_llvm_intrinsics")
    joined = " ".join(exe.blocking_questions)
    assert "RVV" in joined or "RISCV" in joined or "vendor intrinsics" in joined
    assert "RISC-V extension target" in exe.reason


def test_rocc_route_adds_translation_dir_for_gemmini() -> None:
    modmap = _modmap("gemmini_mx")
    exe = _stage(modmap, "executable_sources_llvm_intrinsics")
    assert any(p.endswith("Translation/") and "GemminiMx" in p for p in exe.write_paths), exe.write_paths


def test_gpu_route_extends_radiance_hal_with_dispatch_builder() -> None:
    modmap = _modmap("radiance_muon")
    sched = _stage(modmap, "dispatch_scheduling")
    hal = _stage(modmap, "hal_driver")
    assert any(p.endswith("dispatch_builder.c") for p in sched.write_paths), sched.write_paths
    assert any("radiance_author_questions.md" in p for p in hal.read_paths), hal.read_paths
