"""Host resource admission is explicit, measured, and target independent."""
from __future__ import annotations

import pytest

from merlin.perf.host_resources import (HostResourcePolicy, HostResourceTripwire,
                                        parse_proc_meminfo, summarize_samples)


def _sample(*, available_kib: int = 80, swap_free_kib: int = 8):
    return parse_proc_meminfo(
        f"""MemTotal:       100 kB
MemAvailable:   {available_kib} kB
SwapTotal:       10 kB
SwapFree:        {swap_free_kib} kB
""", observed_at_unix_s=1.0)


def test_proc_sample_records_physical_bytes_and_swap_use():
    sample = _sample()
    assert sample.memory_available_bytes == 80 * 1024
    assert sample.swap_used_bytes == 2 * 1024
    assert sample.record()["observed_at_unix_s"] == 1.0


@pytest.mark.parametrize("mutation", ["missing", "unit", "inconsistent"])
def test_proc_sample_refuses_unknown_or_inconsistent_evidence(mutation):
    text = "MemTotal: 100 kB\nMemAvailable: 80 kB\nSwapTotal: 10 kB\nSwapFree: 8 kB\n"
    if mutation == "missing":
        text = text.replace("SwapFree: 8 kB\n", "")
    elif mutation == "unit":
        text = text.replace("80 kB", "80 MB")
    else:
        text = text.replace("SwapFree: 8 kB", "SwapFree: 12 kB")
    with pytest.raises(ValueError):
        parse_proc_meminfo(text)


def test_tripwire_stops_only_after_consecutive_pressure_and_recovers():
    policy = HostResourcePolicy(50 * 1024, 4 * 1024, consecutive_violations_to_stop=2)
    guard = HostResourceTripwire(policy)
    assert guard.observe(_sample(available_kib=40))["status"] == "continue"
    assert guard.observe(_sample(available_kib=80))["consecutive_violations"] == 0
    first = guard.observe(_sample(available_kib=40, swap_free_kib=4))
    second = guard.observe(_sample(available_kib=40, swap_free_kib=4))
    assert first["reasons"] == ["memory_available_below_limit", "swap_used_above_limit"]
    assert second["status"] == "stop"


def test_sample_summary_keeps_only_resource_extrema_and_endpoints():
    samples = [_sample(available_kib=80), _sample(available_kib=30, swap_free_kib=2)]
    summary = summarize_samples(samples)
    assert summary["sample_count"] == 2
    assert summary["minimum_memory_available_bytes"] == 30 * 1024
    assert summary["maximum_swap_used_bytes"] == 8 * 1024
