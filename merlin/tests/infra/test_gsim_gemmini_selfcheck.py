from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from merlin.common.paths import repo_root


def _module():
    path = repo_root() / "merlin/experiments/gsim_wholemodel/run_gemmini_selfcheck.py"
    spec = importlib.util.spec_from_file_location("_gsim_gemmini_selfcheck", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gsim_result_requires_numeric_completion_and_nonvacuous_traffic():
    module = _module()
    result, axi = module._parse_result(
        "GSIM_RESULT status=pass completion=1 kernel_seen=1 entry=0x80000000 "
        "completion_cycle=52647 kernel_to_verdict_cycles=46498 "
        "gemmini_busy_cycles=223 loaded_bytes=8866\n"
        "GSIM_AXI ar=50 aw=2 w=16 base=0x80000000 size=67108864\n")
    assert result["status"] == "pass"
    assert result["entry"] == 0x80000000
    assert axi == {"ar": 50, "aw": 2, "w": 16,
                   "base": 0x80000000, "size": 67108864}


@pytest.mark.parametrize(
    "replacement",
    [
        "status=fail",
        "completion=0",
        "kernel_seen=0",
        "gemmini_busy_cycles=0",
        "ar=0",
        "aw=0",
        "w=0",
    ],
)
def test_gsim_result_fails_closed(replacement: str):
    module = _module()
    text = (
        "GSIM_RESULT status=pass completion=1 kernel_seen=1 completion_cycle=9 "
        "kernel_to_verdict_cycles=7 gemmini_busy_cycles=4\n"
        "GSIM_AXI ar=3 aw=2 w=2 base=0x80000000 size=67108864\n")
    key = replacement.partition("=")[0]
    lines = []
    for line in text.splitlines():
        fields = [replacement if field.partition("=")[0] == key else field
                  for field in line.split()]
        lines.append(" ".join(fields))
    with pytest.raises(module.SmokeError):
        module._parse_result("\n".join(lines))


def test_selfcheck_is_silent_and_has_distinct_verdict_markers():
    source = (repo_root() / "merlin/experiments/gsim_wholemodel/gemmini_selfcheck.c").read_text()
    assert "printf(" not in source
    assert "merlin_gsim_pass_marker" in source
    assert "merlin_gsim_fail_marker" in source
    assert "gemmini_kernel(T_W, T_A0, T_Y0)" in source
    assert "golden +=" in source
