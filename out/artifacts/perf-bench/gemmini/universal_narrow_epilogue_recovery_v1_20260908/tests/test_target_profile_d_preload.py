from pathlib import Path

import pytest

from mlir_oot.lowering.plan import LoweringDeclined
from mlir_oot.lowering.schedule import Scheduler
from mlir_oot.tables.target_profile import load


def test_universal_contract_refuses_radix_recovery_without_live_d() -> None:
    contract = Path(__file__).parents[1] / "docs/universal_target/UNIVERSAL_GEMMINI_U250.yaml"
    target = load(contract)
    assert target.d_preload_from_dram is False

    scheduler = object.__new__(Scheduler)
    scheduler.target = target
    with pytest.raises(LoweringDeclined, match="hardcodes Gemmini's D operand"):
        scheduler.recover_loop_ws_block(
            a="a", b="b", dst="c", rows=16, cols=16, depth=16,
            a_stride=16, b_stride=16, dst_stride=16,
            a_offset=0, b_offset=0, dst_offset_bytes=0,
            accumulate_dst=False,
        )
