import pytest

from merlin.compiler.build.zephyr.config import project_config
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
        dram_base=2147483648,
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


def test_conditional_defaults_remain_unknown(tmp_path):
    selected = tree(tmp_path, "selected", vector=True)
    (selected.base / "arch" / "Kconfig").write_text("config RISCV_VECTOR_MAX_LEN\n  int\n  default 512 if SOC_WIDE\n")
    assert selected.default_int("RISCV_VECTOR_MAX_LEN") is None
