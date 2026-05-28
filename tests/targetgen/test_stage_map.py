from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from targetgen import build_support_plan, load_capability_spec  # noqa: E402
from targetgen.model import PIPELINE_STAGES  # noqa: E402
from targetgen.stage_map import build_modification_map  # noqa: E402

EXAMPLES = REPO_ROOT / "target_specs" / "examples"


def _modmap_for(target: str):
    caps = load_capability_spec(EXAMPLES / target / "capability.yaml")
    styles = build_support_plan(caps).integration_styles
    return build_modification_map(caps, targetgen_styles=styles)


def _stage(modmap, name: str):
    matches = [s for s in modmap.stages if s.stage == name]
    assert len(matches) == 1, f"expected exactly one {name} stage"
    return matches[0]


def test_modification_map_covers_all_nine_stages() -> None:
    modmap = _modmap_for("radiance_muon")
    assert tuple(s.stage for s in modmap.stages) == PIPELINE_STAGES


def test_radiance_muon_targets_hal_driver_paths() -> None:
    modmap = _modmap_for("radiance_muon")
    assert modmap.primary_integration == "runtime_hal"
    hal_stage = _stage(modmap, "hal_driver")
    assert hal_stage.applies is True
    assert any("runtime/src/iree/hal/drivers/radiance_muon" in p for p in hal_stage.write_paths)
    assert "iree_runtime_plugin.cmake" in hal_stage.write_paths
    assert "tools/build.py" in hal_stage.write_paths


def test_gemmini_mx_anchors_post_global_plugin() -> None:
    modmap = _modmap_for("gemmini_mx")
    assert "post_global_plugin" in modmap.integration_styles
    plugin_stage = _stage(modmap, "global_optimization")
    assert plugin_stage.applies is True
    plugin_dir = "compiler/plugins/target/GemminiMx"
    assert any(p.startswith(plugin_dir) for p in plugin_stage.write_paths)
    assert "iree_compiler_plugin.cmake" in plugin_stage.write_paths


def test_saturn_opu_v128_keeps_llvm_ukernel_path() -> None:
    modmap = _modmap_for("saturn_opu_v128")
    assert modmap.primary_integration == "llvm_ukernel"
    exe_stage = _stage(modmap, "executable_sources_llvm_intrinsics")
    assert exe_stage.applies is True
    assert any("llvm/lib/Target/RISCV" in p for p in exe_stage.write_paths)
    assert any("third_party/iree_bar" in p for p in exe_stage.write_paths)
    hal_stage = _stage(modmap, "hal_driver")
    assert hal_stage.applies is False
    assert hal_stage.write_paths == []


def test_spacemit_skips_dialect_when_only_llvm_ukernel() -> None:
    modmap = _modmap_for("spacemit_x60_xsmtvdot")
    dialect = _stage(modmap, "linalg_arith_dialect")
    plugin = _stage(modmap, "global_optimization")
    assert dialect.applies is False
    assert dialect.write_paths == []
    assert plugin.applies is False
    assert plugin.write_paths == []


def test_validation_commands_use_merlin_wrapper() -> None:
    modmap = _modmap_for("radiance_muon")
    for stage in modmap.stages:
        for cmd in stage.validation_commands:
            assert cmd.startswith("./merlin "), f"{stage.stage}: {cmd!r}"


def test_ml_framework_import_always_applies() -> None:
    for target in ("radiance_muon", "gemmini_mx", "saturn_opu_v128"):
        modmap = _modmap_for(target)
        assert _stage(modmap, "ml_framework_import").applies is True
