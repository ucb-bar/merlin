"""The chip examples' FireSim stage (examples/lib/firesim_flow.py), with no FPGA, queue or toolchain.

What is pinned: a planned job always names its bitstream; a returned console is graded against the
W8A8 golden and refused when it ran on another bitstream or is another image; a run outside the queue
can be graded from its uartlog alone; and `chip_flow op` only plans FireSim jobs for images that passed
their spike gate.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from merlin.common.paths import repo_root

HW = "u250_test_bitstream"


def _load(name: str):
    path = repo_root() / "examples" / "lib" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def flow(monkeypatch, tmp_path):
    """firesim_flow with a fake FireSim board and a host configured only from this dict."""
    module = _load("firesim_flow")
    host_env: dict[str, str] = {}
    monkeypatch.setattr(module, "_env", lambda key: host_env.get(key, ""))
    monkeypatch.setattr(module, "_board", lambda name: SimpleNamespace(name=name, bitstream=HW, vlen=512))
    module.host_env = host_env
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    return module


def _bundle(tmp_path: Path) -> tuple[Path, np.ndarray]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    w8 = np.linspace(-2.0, 3.0, 24, dtype=np.float32)
    np.save(bundle / "golden_w8a8.npy", w8)
    np.save(bundle / "golden.npy", w8 * 1.0001)
    return bundle, w8


def _uart(values: np.ndarray, *, cycles: int, build_hash: str, done: bool = True) -> str:
    from merlin.runtime.backends.zephyr_model import out_hash_fnv1a32

    bits = [struct.unpack("<I", struct.pack("<f", float(v)))[0] for v in values]
    lines = [
        "*** Booting Zephyr OS build 852bb170cc56 ***",
        "=== merlin_zephyr hart=0 ===",
        f"METRIC build_hash {build_hash}",
        "STAGE boot hart=0 t=73",
    ]
    if done:
        lines += [
            f"METRIC cycles {cycles}",
            f"OUT {len(bits)} {' '.join(map(str, bits))}",
            f"HASH fnv1a32 {len(bits)} {out_hash_fnv1a32(values)}",
            "DONE",
        ]
        lines += ["Simulation complete.", f"*** PASSED *** after {cycles * 2} cycles"]
    lines += [
        "Emulation Performance Summary",
        "Wallclock Time Elapsed: 573.6 s",
        "Host Frequency: 15.000 MHz",
        "Effective Target Frequency: 14.918 MHz",
    ]
    return "\n".join(lines) + "\n"


def _results_dir(tmp_path: Path, name: str, uart: str, hw: str | None = HW) -> Path:
    d = tmp_path / name / "merlin-zephyr0"
    d.mkdir(parents=True)
    (d / "uartlog").write_text(uart)
    if hw is not None:
        (d / "HW_CFG_SUMMARY").write_text(f"RuntimeHWConfig: {hw}\nBitstream tar: file:///nowhere/{hw}.tar.gz\n")
    return d


def _plan(flow, tmp_path: Path, bundle: Path) -> Path:
    job = tmp_path / "job"
    legs = {}
    for leg, h in (("device", "aaaa"), ("control", "bbbb")):
        elf = tmp_path / f"{leg}.elf"
        elf.write_bytes(leg.encode())
        legs[leg] = {"elf": elf, "build_hash": h}
    flow.plan(job, board="b", name="op_x", bundle=bundle, legs=legs, timeout_s=1800)
    return job


@pytest.mark.parametrize(
    ("bit_exact", "control_has_unit", "tier"),
    [(True, False, "w8a8"), (False, False, "w8a8"), (True, True, "w8a8"), (True, False, "fp32")],
)
def test_whole_model_legs_stage_only_exact_spike_gated_images(
    flow, tmp_path, monkeypatch, bit_exact, control_has_unit, tier
):
    from merlin.common import artifacts

    model = "test_model"
    bundle = tmp_path / f"{model}_int8_full"
    bundle.mkdir()
    (bundle / "model.mlir").write_text("module {}")
    np.save(bundle / "golden_w8a8.npy", np.zeros(1, dtype=np.float32))
    monkeypatch.setattr(artifacts, "recaptures_dir", lambda: tmp_path)
    binaries = {}
    for leg in flow.LEGS:
        binaries[leg] = tmp_path / f"source-{leg}.elf"
        binaries[leg].write_bytes(leg.encode())

    def compile_image(cmd, **kwargs):
        assert cmd[cmd.index("--package") + 1] == str(flow.REPO / "out/artifacts/targets/rvv/test")
        leg = "device" if "--matrix-unit" in cmd else "control"
        matrix = (
            {
                "unit_instruction_counts": {"OPMACC": 1},
                "shipped_on_simulator": {
                    "status": "verified",
                    "bit_exact_vs_twin": bit_exact,
                    "verify": {"gate_ok": True, "tier_ok": "w8a8"},
                },
            }
            if leg == "device"
            else None
        )
        result = {
            "status": "verified",
            "binary": str(binaries[leg]),
            "build_hash": leg,
            "verify": {"gate_ok": True, "tier_ok": tier},
            "matrix": matrix,
            "output_identity": {"out_hash": {"algo": "fnv1a32", "elems": 1, "value": 1}, "prefix_sha256": "x"},
        }
        if matrix:
            matrix["shipped_on_simulator"]["output_identity"] = result["output_identity"]
        return SimpleNamespace(returncode=0, stdout="build log\n" + json.dumps(result, indent=2), stderr="")

    monkeypatch.setattr(flow.subprocess, "run", compile_image)
    monkeypatch.setattr(
        flow,
        "_unit_counts",
        lambda elf, *args: {"OPMACC": 1} if "device" in str(elf) or control_has_unit else {},
    )
    args = SimpleNamespace(
        bundle=f"{model}_int8_full",
        board="fs",
        package="rvv/test",
        harts=1,
        matrix_unit="opu",
        matrix_config="cfg",
        matrix_support_target="kodiak",
        matrix_kernel=None,
        spike_timeout=40,
        timeout=7200,
    )
    if not bit_exact:
        with pytest.raises(SystemExit, match="bit-exact twin gate"):
            flow.build_legs(args)
        assert not (tmp_path / "out" / "artifacts" / "probes" / "fs" / f"{model}_firesim_h1" / flow.MANIFEST).exists()
        return
    if tier != "w8a8":
        with pytest.raises(SystemExit, match="W8A8 gate"):
            flow.build_legs(args)
        return
    if control_has_unit:
        with pytest.raises(SystemExit, match="control leg: staged ELF contains"):
            flow.build_legs(args)
        return
    job = flow.build_legs(args)
    assert job.parent.name == f"{model}_firesim_h1" and job.name.startswith("run_")
    manifest = flow.load(job)
    assert manifest["timeout_s"] == 7200
    for leg in flow.LEGS:
        staged = Path(manifest["legs"][leg]["elf"])
        assert staged.read_bytes() == binaries[leg].read_bytes()
        assert manifest["legs"][leg]["spike_gate"]["verify"]["gate_ok"] is True
        assert manifest["legs"][leg]["spike_output_identity"]["out_hash"]["elems"] == 1


def test_whole_model_legs_require_w8a8_reference_before_compile(flow, tmp_path, monkeypatch):
    from merlin.common import artifacts

    bundle = tmp_path / "test_model_int8_full"
    bundle.mkdir()
    (bundle / "model.mlir").write_text("module {}")
    monkeypatch.setattr(artifacts, "recaptures_dir", lambda: tmp_path)
    args = SimpleNamespace(bundle=bundle.name, board="fs", harts=1)
    with pytest.raises(SystemExit, match="requires the model's W8A8 reference"):
        flow.build_legs(args)


def test_failed_rebuild_cannot_leave_a_prior_firesim_plan_submittable(flow, tmp_path, monkeypatch):
    from merlin.common import artifacts

    bundle = tmp_path / "test_model_int8_full"
    bundle.mkdir()
    (bundle / "model.mlir").write_text("module {}")
    np.save(bundle / "golden_w8a8.npy", np.zeros(1, dtype=np.float32))
    monkeypatch.setattr(artifacts, "recaptures_dir", lambda: tmp_path)
    job = tmp_path / "out/artifacts/probes/fs/test_model_firesim_h1"
    elf = tmp_path / "old_nan.elf"
    elf.write_bytes(b"old")
    flow.plan(job, board="fs", name="test_model", bundle=bundle, legs={"device": {"elf": elf}})
    monkeypatch.setattr(
        flow.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="compile failed"),
    )
    args = SimpleNamespace(
        bundle=bundle.name,
        board="fs",
        package="rvv/test",
        harts=1,
        matrix_unit="opu",
        matrix_config="cfg",
        matrix_support_target="kodiak",
        matrix_kernel=None,
        spike_timeout=40,
        timeout=7200,
    )
    with pytest.raises(SystemExit, match="returned no result"):
        flow.build_legs(args)
    assert not (job / flow.MANIFEST).exists()
    assert len(list(job.glob("firesim_job.previous_*.json"))) == 1
    assert len(list(job.glob("run_*"))) == 1
    assert elf.read_bytes() == b"old"


# ------------------------------------------------------------------------------------------- parsing


def test_the_simulator_footer_and_hw_summary_are_read_structurally(flow, tmp_path):
    foot = flow.read_sim_footer(_uart(np.zeros(2), cycles=10, build_hash="x"))
    assert foot == {
        "sim_passed": True,
        "target_cycles_total": 20,
        "fpga_wall_s": 573.6,
        "host_mhz": 15.0,
        "effective_mhz": 14.918,
    }
    d = _results_dir(tmp_path, "r", "x")
    assert flow.read_hw_summary(d)["RuntimeHWConfig"] == HW
    assert flow.read_hw_summary(tmp_path / "missing") == {}


# ------------------------------------------------------------------------------------------- plan


def test_a_plan_records_the_elfs_their_digests_and_the_workload(flow, tmp_path):
    bundle, _ = _bundle(tmp_path)
    job = _plan(flow, tmp_path, bundle)
    manifest = flow.load(job)
    assert manifest["hw_config"] == HW and manifest["workload"] == flow.DEFAULT_WORKLOAD
    assert set(manifest["legs"]) == {"device", "control"}
    assert len(manifest["legs"]["device"]["elf_sha256"]) == 64
    wl = json.loads((job / "firesim" / "workload" / f"{flow.DEFAULT_WORKLOAD}.json").read_text())
    assert wl["common_bootbinary"] == manifest["bootbinary"]


def test_a_plan_refuses_a_missing_elf_and_a_board_without_a_bitstream(flow, tmp_path, monkeypatch):
    bundle, _ = _bundle(tmp_path)
    with pytest.raises(SystemExit, match="no ELF"):
        flow.plan(tmp_path / "j", board="b", name="n", bundle=bundle, legs={"device": {"elf": tmp_path / "no.elf"}})
    real = _load("firesim_flow")
    monkeypatch.setattr(
        "merlin.runtime.boards.board", lambda name: SimpleNamespace(name=name, bitstream=None, vlen=None)
    )
    with pytest.raises(SystemExit, match="not a FireSim board"):
        real._board("spike_like")


def test_every_submission_names_its_bitstream_and_the_hwdb_entry(flow, tmp_path):
    bundle, _ = _bundle(tmp_path)
    chipyard = tmp_path / "chipyard"
    hwdb = chipyard / "sims/firesim/deploy/built-hwdb-entries" / HW
    hwdb.parent.mkdir(parents=True)
    hwdb.write_text(f"{HW}:\n  bitstream_tar: file:///x\n")
    flow.host_env.update({"MERLIN_CHIPYARD": str(chipyard), "MERLIN_EXT_FIRESIM_QUEUE": str(tmp_path / "q")})
    job = _plan(flow, tmp_path, bundle)
    host = flow.host_config()
    argv = flow.submission_argv(flow.load(job), "device", host)
    assert argv[:2] == [str(tmp_path / "q" / "bin" / "firesim-queue"), "runworkload-full"]
    assert argv[argv.index("--hw-config") + 1] == HW
    assert argv[argv.index("--hwdb-config-artifact") + 1] == str(hwdb)
    assert argv[argv.index("--stage-from") + 1].endswith("device.elf")
    assert argv[argv.index("--timeout") + 1] == "1800"
    with pytest.raises(SystemExit, match="priority"):
        flow.submission_argv(flow.load(job), "device", host, priority=3)


def test_describe_prints_queue_commands_or_the_no_queue_recipe(flow, tmp_path):
    bundle, _ = _bundle(tmp_path)
    job = _plan(flow, tmp_path, bundle)
    text = flow.describe(job, flow.load(job), flow.host_config())
    assert "No queue configured" in text and f"default_hw_config: {HW}" in text
    assert "firesim infrasetup" in text and "--uartlog" in text
    flow.host_env.update(
        {"MERLIN_EXT_FIRESIM_QUEUE": str(tmp_path / "q"), "MERLIN_FIRESIM_CLIENT_DROP_ENV": "HOME,USER"}
    )
    text = flow.describe(job, flow.load(job), flow.host_config())
    assert text.count("runworkload-full") == 2 and "env -u HOME -u USER " in text
    assert f"--hw-config {HW}" in text


def test_the_installed_workload_names_the_boot_binary(flow, tmp_path):
    chipyard = tmp_path / "cy"
    wl = chipyard / "sims/firesim/deploy/workloads/mine.json"
    wl.parent.mkdir(parents=True)
    wl.write_text(json.dumps({"benchmark_name": "mine", "common_bootbinary": "zephyr0-zephyr.elf"}))
    flow.host_env.update({"MERLIN_CHIPYARD": str(chipyard), "MERLIN_FIRESIM_WORKLOAD": "mine"})
    host = flow.host_config()
    assert host["workload_installed"] and host["bootbinary"] == "zephyr0-zephyr.elf"


def test_a_staged_binary_the_daemon_does_not_own_is_refused(flow, tmp_path, monkeypatch):
    deploy, queue = tmp_path / "deploy", tmp_path / "queue"
    queue.mkdir()
    (queue / "daemon.pid").write_text("1\n")
    assert flow.staging_refusal(deploy, "mine", "boot.elf", queue) is None  # absent: the daemon creates it
    dst = deploy / "workloads" / "mine" / "boot.elf"
    dst.parent.mkdir(parents=True)
    dst.write_bytes(b"old")
    assert flow.staging_refusal(deploy, "mine", "boot.elf", queue) is None  # same owner as the daemon
    owners = {str(queue / "daemon.pid"): 1001, str(dst): 1000}
    monkeypatch.setattr(flow, "_owner_uid", lambda p: owners[str(p)])
    why = flow.staging_refusal(deploy, "mine", "boot.elf", queue)
    assert why and "Operation not permitted" in why and "uid 1000" in why


def test_submit_refuses_without_a_configured_queue(flow, tmp_path):
    bundle, _ = _bundle(tmp_path)
    job = _plan(flow, tmp_path, bundle)
    with pytest.raises(SystemExit, match="not configured"):
        flow.submit(job)


def test_submit_refuses_a_changed_spike_gated_elf_before_contacting_queue(flow, tmp_path):
    bundle, _ = _bundle(tmp_path)
    job = _plan(flow, tmp_path, bundle)
    Path(flow.load(job)["legs"]["device"]["elf"]).write_bytes(b"changed")
    with pytest.raises(SystemExit, match="planned ELF is missing or changed"):
        flow.submit(job)


# ------------------------------------------------------------------------------------------- collect


def test_a_faithful_run_is_verified_and_the_speedup_reported(flow, tmp_path):
    bundle, w8 = _bundle(tmp_path)
    job = _plan(flow, tmp_path, bundle)
    dev = _results_dir(tmp_path, "d", _uart(w8, cycles=1000, build_hash="aaaa"))
    ctl = _results_dir(tmp_path, "c", _uart(w8, cycles=1580, build_hash="bbbb"))
    res = flow.collect(job, uartlogs={"device": dev / "uartlog", "control": ctl / "uartlog"}, measurement=False)
    assert res["all_verified"], res
    assert res["device"]["cycles"] == 1000 and res["device"]["grade"]["w8a8_cos"] > 0.999999
    assert res["device"]["grade"]["tier_ok"] == "w8a8" and res["device"]["hw_config"] == HW
    assert res["speedup_control_over_device"] == pytest.approx(1.58)
    assert (job / "firesim" / "results.json").is_file()
    assert res["provenance"]["bitstream_artifact"]["matched"] is False  # an undeclared tar is not pinned


def test_collect_requires_the_full_spike_output_even_when_the_printed_prefix_passes(flow, tmp_path):
    from merlin.runtime.backends import output_identity
    from merlin.runtime.backends import zephyr_model as zm

    bundle, w8 = _bundle(tmp_path)
    job = _plan(flow, tmp_path, bundle)
    good = zm._parse_console(_uart(w8, cycles=10, build_hash="aaaa"), 0)
    manifest = flow.load(job)
    manifest["legs"]["device"]["spike_output_identity"] = output_identity.output_identity(good)
    flow._save(job, manifest)
    uart = _uart(w8, cycles=10, build_hash="aaaa")
    uart = uart.replace(
        f"HASH fnv1a32 {w8.size} {good['out_hash']['value']}",
        f"HASH fnv1a32 {w8.size} {good['out_hash']['value'] ^ 1}",
    )
    device = _results_dir(tmp_path, "d", uart)
    result = flow.collect(job, uartlogs={"device": device / "uartlog"}, measurement=False)
    assert result["device"]["grade"]["tier_ok"] == "w8a8"
    assert result["device"]["out_hash_matches_spike"] is False
    assert any("whole-output HASH differs" in p for p in result["device"]["problems"])


def test_a_run_on_another_bitstream_or_of_another_image_is_not_verified(flow, tmp_path):
    bundle, w8 = _bundle(tmp_path)
    job = _plan(flow, tmp_path, bundle)
    other_hw = _results_dir(tmp_path, "d", _uart(w8, cycles=1, build_hash="aaaa"), hw="someone_elses_design")
    stale = _results_dir(tmp_path, "c", _uart(w8, cycles=1, build_hash="cafe"))
    res = flow.collect(job, uartlogs={"device": other_hw / "uartlog", "control": stale / "uartlog"}, measurement=False)
    assert not res["all_verified"]
    assert any("someone_elses_design" in p for p in res["device"]["problems"])
    assert any("not this leg's image" in p for p in res["control"]["problems"])


def test_unverifiable_and_wrong_runs_say_why(flow, tmp_path):
    bundle, w8 = _bundle(tmp_path)
    job = _plan(flow, tmp_path, bundle)
    no_summary = _results_dir(tmp_path, "d", _uart(-w8, cycles=1, build_hash="aaaa"), hw=None)
    hung = _results_dir(tmp_path, "c", _uart(w8, cycles=1, build_hash="bbbb", done=False))
    res = flow.collect(job, uartlogs={"device": no_summary / "uartlog", "control": hung / "uartlog"}, measurement=False)
    dev, ctl = res["device"], res["control"]
    assert any("HW_CFG_SUMMARY" in p for p in dev["problems"])
    assert any("fails the gate" in p for p in dev["problems"])
    assert ctl["status"] == "no_output" and ctl["last_progress"] == ["STAGE boot hart=0 t=73"]


def test_an_elf_rebuilt_since_planning_is_flagged(flow, tmp_path):
    bundle, w8 = _bundle(tmp_path)
    job = _plan(flow, tmp_path, bundle)
    (tmp_path / "device.elf").write_bytes(b"rebuilt")
    dev = _results_dir(tmp_path, "d", _uart(w8, cycles=5, build_hash="aaaa"))
    res = flow.collect(job, uartlogs={"device": dev / "uartlog"}, measurement=False)
    assert any("changed on disk" in p for p in res["device"]["problems"])
    assert res["control"]["status"] == "not_run"


def test_a_queue_job_is_found_in_the_daemons_overlay(flow, tmp_path):
    queue = tmp_path / "q"
    leaf = queue / "jobs/42/deploy_overlay/results-workload/2026-01-01--00-00-00-wl-q42/wl0"
    leaf.mkdir(parents=True)
    (leaf / "uartlog").write_text("x")
    host = {"queue_root": queue, "deploy": None}
    assert flow.results_dir_for_job(42, "wl", host) == leaf
    assert flow.results_dir_for_job(43, "wl", host) is None


def test_collect_writes_a_measurement_under_the_bitstreams_substrate(flow, tmp_path):
    bundle, w8 = _bundle(tmp_path)
    job = _plan(flow, tmp_path, bundle)
    dev = _results_dir(tmp_path, "d", _uart(w8, cycles=7, build_hash="aaaa"))
    res = flow.collect(job, uartlogs={"device": dev / "uartlog"})
    path = Path(res["measurement"])
    assert path.parts[-4:-1] == ("measurements", f"firesim_{HW}", "op_x")
    assert json.loads((path / "perf_results.json").read_text())["device"]["cycles"] == 7


# ------------------------------------------------------------------------------------------- chip_flow


def _op_results(tmp_path: Path, status: str) -> dict:
    files = {}
    for name in ("rvv", "ship", "twin"):
        files[name] = tmp_path / f"{name}.elf"
        files[name].write_bytes(name.encode())
    return {
        "rvv": {"status": status, "binary": str(files["rvv"]), "build_hash": "r1", "cycles": 9},
        "matrix": {
            "status": status,
            "binary": str(files["ship"]),
            "twin_binary": str(files["twin"]),
            "build_hash": "m1",
            "matrix": {"unit_instruction_counts": {"X": 1}},
        },
    }


@pytest.mark.parametrize("status, planned", [("verified", True), ("run_mismatch", False)])
def test_op_on_a_firesim_board_plans_only_gated_images(flow, tmp_path, monkeypatch, capsys, status, planned):
    chip = _load("chip_flow")
    bundle, _ = _bundle(tmp_path)
    monkeypatch.setitem(sys.modules, "firesim_flow", flow)
    monkeypatch.setattr(chip, "bundle_dir", lambda name: bundle)
    monkeypatch.setattr(chip, "bundle_on_pin", lambda bundle: True)  # captured by model2MLIR at its pin
    monkeypatch.setattr(chip, "capture", lambda *a, **k: pytest.fail("bundle exists; no capture"))
    results = iter([_op_results(tmp_path, status)["rvv"], _op_results(tmp_path, status)["matrix"]])
    monkeypatch.setattr(chip, "_compile", lambda name, args, matrix: next(results))
    monkeypatch.setattr(chip, "_firesim_board", lambda name: True)
    rc = chip.main(
        ["op", "gemm", "2", "2", "2", "--board", "fs", "--matrix-unit", "u", "--matrix-config", "c",
         "--matrix-support-target", "t"]
    )  # fmt: skip
    out = capsys.readouterr().out
    kept = Path(out.split("kept in: ")[1].split()[0])
    assert (kept / flow.MANIFEST).is_file() is planned
    if planned:
        assert rc == 0 and "not submitted" in out
        manifest = flow.load(kept)
        assert manifest["legs"]["device"]["build_hash"] == "m1"
        assert manifest["legs"]["control"]["elf"] == str(kept / "rvv.elf")
        assert manifest["timeout_s"] == 1800
    else:
        assert rc == 1 and "not planning FireSim jobs" in out
