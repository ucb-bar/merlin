"""The same compiler process can configure distinct selected SoC/Zephyr pairs."""

from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location

import pytest

from merlin.common.paths import repo_root
from merlin.compiler.build.zephyr.config import project_config, vector_max_len_bits
from merlin.compiler.build.zephyr.tree import TreeConfiguration
from merlin.runtime.boards import CONSOLE_HTIF, FLOW_ZEPHYR, Board


def tree(tmp_path, name, *, vector, width=256):
    base = tmp_path / name
    arch = base / "arch"
    arch.mkdir(parents=True)
    (arch / "Kconfig").write_text(
        f"config RISCV_VECTOR_MAX_LEN\n  int\n  default {width}\n"
        + ("config RISCV_V_KERNEL_ONLY\n  bool\nconfig RISCV_ISA_EXT_V_LAZY\n  bool\n" if vector else "")
    )
    return TreeConfiguration(base)


@pytest.fixture
def board():
    return Board(
        name="configuration-fixture",
        dram_bytes=1 << 28,
        dram_base=0x80000000,
        harts=2,
        vector_harts=2,
        console=CONSOLE_HTIF,
        flow=FLOW_ZEPHYR,
        fpu_sharing=False,
        zephyr_vector_ext=True,
        vlen=128,
    )


def test_selected_trees_do_not_share_capabilities(tmp_path, board):
    legacy = tree(tmp_path, "legacy", vector=False)
    vector = tree(tmp_path, "vector", vector=True, width=512)
    first = project_config(2, "rvv", board, tree=legacy)
    second = project_config(2, "rvv", board, tree=vector)
    assert "CONFIG_RISCV_ISA_EXT_V=y" not in first
    assert "CONFIG_RISCV_V_KERNEL_ONLY=y" in second
    assert "CONFIG_RISCV_VECTOR_MAX_LEN=512" in second
    assert project_config(2, "rvv", board, tree=legacy) == first


def test_unselected_tree_does_not_poison_later_selection(tmp_path, board):
    assert not TreeConfiguration(None).has("RISCV_V_KERNEL_ONLY")
    selected = tree(tmp_path, "selected", vector=True)
    assert "CONFIG_RISCV_ISA_EXT_V=y" in project_config(2, "rvv", board, tree=selected)


def test_stale_board_vector_policy_is_refused(tmp_path, board):
    selected = tree(tmp_path, "selected", vector=True)
    with pytest.raises(RuntimeError, match="zephyr_vector_ext=False"):
        project_config(2, "rvv", replace(board, zephyr_vector_ext=False), tree=selected)


def test_scalar_configuration_and_vector_save_floor(tmp_path, board):
    selected = tree(tmp_path, "selected", vector=True, width=512)
    assert "CONFIG_RISCV_ISA_EXT_V" not in project_config(2, "scalar", board, tree=selected)
    assert vector_max_len_bits(board, selected) == 512
    assert vector_max_len_bits(replace(board, vlen=1024), selected) == 1024


def test_conditional_defaults_remain_unknown(tmp_path):
    selected = tree(tmp_path, "selected", vector=True)
    (selected.base / "arch" / "Kconfig").write_text("config RISCV_VECTOR_MAX_LEN\n  int\n  default 512 if SOC_WIDE\n")
    assert selected.default_int("RISCV_VECTOR_MAX_LEN") is None


def test_new_build_observes_tree_changes(tmp_path):
    selected = tree(tmp_path, "selected", vector=False)
    assert not selected.has("RISCV_V_KERNEL_ONLY")
    (selected.base / "arch" / "Kconfig.extra").write_text("config RISCV_V_KERNEL_ONLY\n  bool\n")
    assert TreeConfiguration(selected.base).has("RISCV_V_KERNEL_ONLY")


def test_build_and_runtime_preflight_use_actual_selected_inputs(tmp_path, monkeypatch):
    from merlin.runtime.backends import zephyr_model as zm

    selected = tmp_path / "zephyr"
    sdk = tmp_path / "sdk"
    tools = tmp_path / "tools"
    for directory in (selected, sdk, tools):
        directory.mkdir()
    for name in ("gcc", "ld", "ar", "objcopy"):
        (tools / f"riscv64-unknown-elf-{name}").write_bytes(b"tool fixture")
    gcc = tools / "riscv64-unknown-elf-gcc"
    monkeypatch.setenv("ZEPHYR_BASE", str(selected))
    monkeypatch.setenv("ZEPHYR_SDK_INSTALL_DIR", str(sdk))
    monkeypatch.delenv("MERLIN_ZEPHYR_SW", raising=False)
    monkeypatch.setattr(zm._spike, "gcc_path", lambda: gcc)
    monkeypatch.setattr(zm, "build_tool", lambda name: tools / name)
    monkeypatch.setattr(zm.toolchain, "available", lambda: True)
    monkeypatch.setattr(zm._spike, "available", lambda: False)
    assert zm.build_available()
    assert not zm.available()

    spec = spec_from_file_location("zephyr_repro_preflight", repo_root() / "build_tools/scripts/check_repro_env.py")
    preflight = module_from_spec(spec)
    spec.loader.exec_module(preflight)
    assert preflight._probe("zephyr_spike")[0] == "unavailable"
    monkeypatch.setattr(zm._spike, "available", lambda: True)
    assert preflight._probe("zephyr_spike")[0] == "available"
    for key in ("zephyr_spike", "zephyr_multicore"):
        assert "MERLIN_ZEPHYR_SW" not in preflight._BASE_CAPABILITIES[key][2]
    assert "MERLIN_ZEPHYR_SW" not in preflight._backend_env_hints("synthetic", "zephyr")

    # The former workspace variable cannot mask a missing actual tree or SDK.
    monkeypatch.setenv("MERLIN_ZEPHYR_SW", str(tmp_path))
    for name in ("ZEPHYR_BASE", "ZEPHYR_SDK_INSTALL_DIR"):
        original = str(selected if name == "ZEPHYR_BASE" else sdk)
        monkeypatch.setenv(name, str(tmp_path / "missing"))
        assert not zm.build_available()
        assert preflight._probe("zephyr_spike")[0] == "unavailable"
        monkeypatch.setenv(name, original)
    (tools / "riscv64-unknown-elf-ld").unlink()
    assert not zm.build_available()
