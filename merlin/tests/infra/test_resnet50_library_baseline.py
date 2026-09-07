"""The hand-written ResNet baseline must be attributable before it is compared to Merlin."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir


SOURCE = (merlin_dir()
          / "experiments/gemmini_perf_bench/scripts/resnet50_library_baseline.py")
SPEC = importlib.util.spec_from_file_location("resnet50_library_baseline_under_test", SOURCE)
BASELINE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASELINE)


WARM = """
MERLIN_PROFILE warmup begin
Total cycles: 900 (100%)
Matmul cycles: 300 (33%)
Im2col cycles: 50 (5%)
Conv cycles: 350 (38%)
Pooling cycles: 50 (5%)
Depthwise convolution cycles: 0 (0%)
Res add cycles: 100 (11%)
Other cycles: 50 (5%)
PASS
MERLIN_PROFILE warmup end rc=0
"""

MEASURED = """
MERLIN_PROFILE measured begin
conv 1 cycles: 12
Prediction: 7 (score: 9)

Total cycles: 1000 (100%)
Matmul cycles: 300 (30%)
Im2col cycles: 100 (10%)
Conv cycles: 400 (40%)
Pooling cycles: 50 (5%)
Depthwise convolution cycles: 0 (0%)
Res add cycles: 100 (10%)
Other cycles: 50 (5%)
PASS
MERLIN_PROFILE measured end rc=0
"""

UART = WARM + MEASURED


def test_wrapper_selects_arm_without_modifying_upstream_source():
    wrapper = BASELINE.render_wrapper("ws_matmul")
    assert '#include "../source/imagenet/resnet50_profiled.c"' in wrapper
    assert '#define main merlin_resnet50_upstream_main' in wrapper
    assert 'char arg1[] = "ws";' in wrapper
    assert 'char arg2[] = "matmul";' in wrapper
    assert wrapper.count("merlin_resnet50_upstream_main(3, argv)") == 2
    assert "MERLIN_PROFILE warmup begin" in wrapper
    assert "MERLIN_PROFILE measured begin" in wrapper


def test_profile_source_changes_only_the_terminal_success_exit():
    upstream = 'if (bad) exit(1);\n    printf("PASS\\n");\n    exit(0);\n}\n'
    got = BASELINE.render_profile_source(upstream)
    assert got == 'if (bad) exit(1);\n    printf("PASS\\n");\n    return 0;\n}\n'
    with pytest.raises(BASELINE.BaselineError, match="absent or ambiguous"):
        BASELINE.render_profile_source("int main(void) { return 0; }\n")


@pytest.mark.parametrize("arm,expected", sorted(BASELINE.ARM_ARGS.items()))
def test_all_six_upstream_modes_are_expressible(arm, expected):
    wrapper = BASELINE.render_wrapper(arm)
    assert f'char arg1[] = "{expected[0]}";' in wrapper
    assert f'char arg2[] = "{expected[1]}";' in wrapper


def test_uart_parser_requires_pass_and_an_exact_component_sum():
    parsed = BASELINE.parse_uart(UART)
    assert parsed["cycles"]["total"] == 1000
    assert parsed["component_sum"] == 1000
    assert parsed["component_percent"]["conv"] == 40.0

    with pytest.raises(BASELINE.BaselineError, match="did not report PASS"):
        BASELINE.parse_uart(UART.replace("PASS", "", 1))
    with pytest.raises(BASELINE.BaselineError, match="reported FAIL"):
        BASELINE.parse_uart(UART.replace("PASS", "FAIL", 1))
    with pytest.raises(BASELINE.BaselineError, match="sum to"):
        BASELINE.parse_uart(UART.replace("Other cycles: 50", "Other cycles: 49"))
    with pytest.raises(BASELINE.BaselineError, match="missing 'Im2col cycles'"):
        BASELINE.parse_uart(WARM + MEASURED.replace("Im2col cycles", "Unpriced cycles", 1))


def test_uart_parser_profiles_only_the_post_warmup_inference():
    parsed = BASELINE.parse_uart(UART)
    assert parsed["cycles"]["total"] == 1000
    assert parsed["profile"] == {
        "warmup_runs": 1,
        "measured_runs": 1,
        "recorded_scope": "post-warm-up compute-cycle decomposition",
    }
    with pytest.raises(BASELINE.BaselineError, match="exactly one"):
        BASELINE.parse_uart(UART + MEASURED)


def test_queue_command_pins_the_physical_design():
    command = BASELINE.queue_command(
        queue=Path("/queue"), chipyard=Path("/chipyard"), elf=Path("/model.elf"),
        hw_config="the-only-acceptable-bitstream", timeout=123, priority=5,
    )
    assert command[command.index("--hw-config") + 1] == "the-only-acceptable-bitstream"
    assert command[command.index("--stage-from") + 1] == "/model.elf"
    assert command[command.index("--timeout") + 1] == "123"
    assert command[:2] == ["/queue", "runworkload-full"]


def test_queue_client_does_not_override_a_cross_user_daemon_identity(monkeypatch):
    monkeypatch.setenv("HOME", "/private/client-home")
    monkeypatch.setenv("USER", "client")
    monkeypatch.setenv("LOGNAME", "client")
    monkeypatch.setenv("PATH", "/needed/by/queue-shim")
    env = BASELINE.queue_client_environment()
    assert "HOME" not in env
    assert "USER" not in env
    assert "LOGNAME" not in env
    path_entries = env["PATH"].split(":")
    assert path_entries[0] == str(BASELINE.QUEUE_CWD_LAUNCHER.parent)
    assert path_entries[1:] == ["/needed/by/queue-shim"]


def test_queue_cwd_launcher_forces_firesim_to_use_the_daemon_selected_deploy_dir():
    launcher = BASELINE.QUEUE_CWD_LAUNCHER
    assert launcher.is_file()
    text = launcher.read_text(encoding="utf-8")
    assert 'shadow_deploy="$shadow_firesim/deploy"' in text
    assert 'link_exact "$chipyard_root/generators" "$shadow_chipyard/generators"' in text
    assert 'link_exact "$chipyard_root/env.sh" "$shadow_chipyard/env.sh"' in text
    assert 'for name in env.sh platforms target-design utils' in text
    assert 'shadow_sim="$shadow_firesim/sim"' in text
    assert '[[ "$(basename "$entry")" == output ]] && continue' in text
    assert 'shadow_driver_dir="$shadow_sim/output/$driver_rel"' in text
    assert 'link_exact "$chipyard_root/sims/firesim-staging"' in text
    assert 'source_wrapper="$shadow_firesim/sourceme-manager.sh"' in text
    assert 'source %q "$@"' in text
    assert 'export PATH=%q:"$PATH"' in text
    assert 'link_exact "$entry" "$shadow_deploy/$(basename "$entry")"' in text
    assert 'exec "$shadow_deploy/firesim"' in text
    assert 'unlink "$overlay_dir/generated-topology-diagrams"' in text
    assert 'mkdir -p "$overlay_dir/generated-topology-diagrams"' in text
    assert "default_simulation_dir:" in text
    assert 'simulation_dir="$job_root/simulation"' in text
    assert 'export PATH="$launcher_dir:$PATH"' in text


def test_queue_make_launcher_fails_closed_around_the_exact_prebuilt_driver():
    launcher = BASELINE.FIRESIM_MAKE_LAUNCHER
    assert launcher.is_file()
    text = launcher.read_text(encoding="utf-8")
    assert "TARGET_CONFIG=FireSimGemminiAndOPUShuttleConfig" in text
    assert 'makefrag_normalized=$(realpath -m -s "$makefrag")' in text
    assert '"$real_make" -q --old-file=firesim_target_symlink_hook' in text
    assert 'exec "$real_make" --old-file=firesim_target_symlink_hook' in text
    assert "exact-config driver is stale" in text
    assert 'exec "$real_make" "$@"' in text


def test_queue_contract_requires_exact_atomic_firesim_lifecycle():
    receipt = BASELINE.validate_queue_help(
        "runworkload-full: atomic kill -> infrasetup -> runworkload -> kill sequence")
    assert receipt["firesim_lifecycle"] == [
        "firesim kill", "firesim infrasetup", "firesim runworkload", "firesim kill"]
    with pytest.raises(BASELINE.BaselineError, match="required"):
        BASELINE.validate_queue_help("runworkload-full: runworkload only")


def test_queue_daemon_phases_must_be_in_order():
    log = "\n".join(
        f"=== [firesim-queue] phase={phase} job_id=9 ==="
        for phase in ("STAGING", "INFRASETUP", "RUNNING", "TEARDOWN"))
    assert BASELINE.validate_queue_phases(log) == [
        "STAGING", "INFRASETUP", "RUNNING", "TEARDOWN"]
    with pytest.raises(BASELINE.BaselineError, match="RUNNING"):
        BASELINE.validate_queue_phases(log.replace("RUNNING", "SKIPPED"))


def test_uart_discovery_prefers_the_cross_user_queue_overlay(tmp_path):
    queue = tmp_path / "queue/bin/firesim-queue"
    queue.parent.mkdir(parents=True)
    queue.touch()
    chipyard = tmp_path / "chipyard"
    overlay_uart = (tmp_path / "queue/jobs/17/deploy_overlay/results-workload"
                    / "2026-01-01-merlin-perfbench-q17/merlin-perfbench0/uartlog")
    native_uart = (chipyard / "sims/firesim/deploy/results-workload"
                   / "2026-01-02-merlin-perfbench-q17/merlin-perfbench0/uartlog")
    overlay_uart.parent.mkdir(parents=True)
    native_uart.parent.mkdir(parents=True)
    overlay_uart.write_text("overlay", encoding="utf-8")
    native_uart.write_text("native", encoding="utf-8")
    assert BASELINE._find_uart(queue, chipyard, 17) == overlay_uart


def test_comparison_prices_native_conv_against_library_matmul_strategy():
    results = {
        "ws_conv": {"cycles": {"total": 400, "conv": 250, "im2col": 0}},
        "ws_matmul": {"cycles": {"total": 1000, "conv": 0, "im2col": 300}},
        "cpu_matmul": {"cycles": {"total": 5000}},
    }
    got = BASELINE.comparison(results)
    assert got == {
        "ws_matmul_over_ws_conv": 2.5,
        "ws_conv_minus_ws_matmul_cycles": -600,
        "ws_native_conv_cycles": 250,
        "ws_explicit_im2col_cycles": 300,
        "cpu_matmul_over_ws_matmul": 5.0,
    }


def test_aggregate_selects_one_exact_median_decomposition():
    rows = [
        {"status": "pass", "arm": "ws_conv", "repetition": 1,
         "cycles": {"total": 110, "conv": 90}, "component_sum": 110,
         "component_percent": {"conv": 81.8}},
        {"status": "pass", "arm": "ws_conv", "repetition": 2,
         "cycles": {"total": 100, "conv": 80}, "component_sum": 100,
         "component_percent": {"conv": 80.0}},
        {"status": "pass", "arm": "ws_conv", "repetition": 3,
         "cycles": {"total": 120, "conv": 99}, "component_sum": 120,
         "component_percent": {"conv": 82.5}},
    ]
    got = BASELINE.aggregate_arm(rows)
    assert got["representative_repetition"] == 1
    assert got["cycles"] == rows[0]["cycles"]
    assert got["total_cycle_distribution"] == {
        "values": [110, 100, 120], "min": 100, "max": 120, "median": 110}
