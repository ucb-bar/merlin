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


def _strict_host_cache(flow, tmp_path: Path, monkeypatch) -> dict:
    cache_dir = tmp_path / "host-cache"
    cache_dir.mkdir()
    key = "a" * 16
    obj = cache_dir / f"{key}.so"
    obj.write_bytes(b"compiled host kernel")
    identity = {"sha256": "b" * 64, "inputs": {"schema": "synthetic-host-inputs"}}
    monkeypatch.setattr(flow, "_current_host_cache_identity", lambda: identity)
    manifest = {
        "schema": "merlin.host_cache_object.v1",
        "key": key,
        "kernel_text_sha256": "c" * 64,
        "compiler_identity_sha256": identity["sha256"],
        "object_sha256": flow._sha256(obj),
        "object_bytes": obj.stat().st_size,
        "lowering_recipe_sha256": "d" * 64,
        "lowering_commands": [{"argv": ["test-lowerer"]}],
        "builder": "merlin.llvmlower.kernel_backend.compile_host",
        "native_codegen": "test native build",
    }
    manifest_path = cache_dir / f"{key}.manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return {
        "schema": "merlin.host_cache.v1",
        "mode": "cache_only",
        "cache_dir": str(cache_dir),
        "identity": identity,
        "loaded": [
            {
                "key": key,
                "path": str(obj),
                "sha256": flow._sha256(obj),
                "bytes": obj.stat().st_size,
                "manifest_path": str(manifest_path),
                "manifest_sha256": flow._sha256(manifest_path),
                "manifest": manifest,
            }
        ],
        "miss_count": 0,
        "verified_after_execution": True,
    }


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
    routing = tmp_path / "routing_report.json"
    routing.write_text("{}")
    (tmp_path / "opu_signatures.json").write_text("[]")

    def compile_image(cmd, **kwargs):
        assert cmd[cmd.index("--package") + 1] == str(flow.REPO / "out/artifacts/targets/rvv/test")
        leg = "device" if "--matrix-unit" in cmd else "control"
        matrix = (
            {
                "unit_instruction_counts": {"OPMACC": 1},
                "routing_report": str(routing),
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
    assert Path(manifest["legs"]["device"]["routing_sidecars"]["routing_report.json"]["path"]).read_text() == "{}"


def test_whole_model_legs_require_w8a8_reference_before_compile(flow, tmp_path, monkeypatch):
    from merlin.common import artifacts

    bundle = tmp_path / "test_model_int8_full"
    bundle.mkdir()
    (bundle / "model.mlir").write_text("module {}")
    monkeypatch.setattr(artifacts, "recaptures_dir", lambda: tmp_path)
    args = SimpleNamespace(bundle=bundle.name, board="fs", harts=1)
    with pytest.raises(SystemExit, match="requires the model's W8A8 reference"):
        flow.build_legs(args)


@pytest.mark.parametrize(
    "alteration", ["none", "missing", "read_write", "unverified", "object", "manifest", "identity"]
)
def test_source_host_cache_receipt_is_required_even_for_diagnostic_plans(flow, tmp_path, monkeypatch, alteration):
    from merlin.targetgen import application_inventory

    bundle = tmp_path / "source"
    bundle.mkdir()
    (bundle / "model.mlir").write_text("module {}")
    np.save(bundle / "golden.npy", np.zeros(2, np.float32))
    receipt = {"status": "verified_materialized"}
    monkeypatch.setattr(application_inventory, "verify_capture_receipt", lambda _path: receipt)
    cache = _strict_host_cache(flow, tmp_path, monkeypatch)
    framework = {
        "run": "host",
        "dtype": "int8_w8a8",
        "bundle": str(bundle),
        "status": "verified",
        "verify": {"tier_ok": "w8a8", "comparison_complete": True, "n_compared": 2, "n_reference": 2},
        "reference": {"kind": "torchao_w8a8_framework", "sha256": flow._sha256(bundle / "golden.npy")},
        "source_quantization": {
            "source_scheme": "int8_dyn_act_int8_weight",
            "source_mlir_sha256": flow._sha256(bundle / "model.mlir"),
            "reference_kind": "torchao_w8a8_framework",
            "integer_matmuls": 1,
            "capture_receipt": receipt,
        },
        "host_cache": cache,
    }
    if alteration == "missing":
        framework.pop("host_cache")
    elif alteration == "read_write":
        cache["mode"] = "read_write"
    elif alteration == "unverified":
        cache["verified_after_execution"] = False
    elif alteration == "object":
        (tmp_path / "host-cache" / ("a" * 16 + ".so")).write_bytes(b"tampered object")
    elif alteration == "manifest":
        (tmp_path / "host-cache" / ("a" * 16 + ".manifest.json")).write_text("{}")
    elif alteration == "identity":
        monkeypatch.setattr(flow, "_current_host_cache_identity", lambda: {"sha256": "f" * 64})
    framework_path = tmp_path / "framework.json"
    framework_path.write_text(json.dumps(framework))
    args = SimpleNamespace(
        dtype="int8_w8a8",
        capture_bundle=bundle,
        gate="twin",
        harts=1,
        matrix_support_target="kodiak",
        matrix_unit="saturn_opu",
        matrix_config="KodiakOPUConfig",
        framework_result=framework_path,
        allow_framework_mismatch=True,
    )
    board = SimpleNamespace(name="firesim_kodiak_opu", bitstream="alveo_u250_firesim_kodiak_opu_1core")
    if alteration != "none":
        with pytest.raises(SystemExit, match="host-cache"):
            flow._source_gate(args, bundle, board)
        return
    source = flow._source_gate(args, bundle, board)
    assert source["host_cache_identity_sha256"] == "b" * 64
    assert source["host_cache_loaded_unique"] == 1
    flow._validate_source_plan({"source": source, "bundle": str(bundle)})
    source["host_cache_identity_sha256"] = "e" * 64
    with pytest.raises(SystemExit, match="host-cache receipt changed"):
        flow._validate_source_plan({"source": source, "bundle": str(bundle)})


@pytest.mark.parametrize(
    ("framework_ok", "tier_mismatch", "spike_complete", "host_partial"),
    [
        (True, False, True, False),
        (True, False, False, False),
        (False, False, False, False),
        (False, True, False, False),
        (False, False, False, True),
    ],
)
def test_source_torchao_pair_binds_capture_and_refuses_diagnostic_submit(
    flow, tmp_path, monkeypatch, framework_ok, tier_mismatch, spike_complete, host_partial
):
    from merlin.targetgen import application_inventory

    flow._board = lambda name: SimpleNamespace(name=name, bitstream="alveo_u250_firesim_kodiak_opu_1core", vlen=512)
    bundle = tmp_path / "spectformer_torchao_w8a8"
    bundle.mkdir()
    (bundle / "model.mlir").write_text("module {}")
    np.save(bundle / "golden.npy", np.zeros(2, dtype=np.float32))
    monkeypatch.setattr(
        application_inventory, "verify_capture_receipt", lambda path: {"status": "verified_materialized"}
    )
    framework = {
        "run": "host",
        "dtype": "int8_w8a8",
        "bundle": str(bundle),
        "status": "verified" if framework_ok or tier_mismatch or host_partial else "run_mismatch",
        "verify": {
            "tier_ok": "w8a8" if framework_ok or host_partial else None,
            "comparison_complete": not host_partial,
            "n_compared": 1 if host_partial else 2,
            "n_reference": 2,
        },
        "reference": {"kind": "torchao_w8a8_framework", "sha256": flow._sha256(bundle / "golden.npy")},
        "source_quantization": {
            "source_scheme": "int8_dyn_act_int8_weight",
            "source_mlir_sha256": flow._sha256(bundle / "model.mlir"),
            "reference_kind": "torchao_w8a8_framework",
            "integer_matmuls": 41,
            "capture_receipt": {"status": "verified_materialized"},
        },
        "host_cache": _strict_host_cache(flow, tmp_path, monkeypatch),
    }
    framework_path = tmp_path / "framework.json"
    framework_path.write_text("host build progress\n" + json.dumps(framework, indent=2))
    args = SimpleNamespace(
        bundle="source_spectformer",
        capture_bundle=bundle,
        dtype="int8_w8a8",
        framework_result=framework_path,
        allow_framework_mismatch=False,
        board="firesim_kodiak_opu",
        harts=1,
        package="rvv/test",
        matrix_support_target="kodiak",
        matrix_unit="saturn_opu",
        matrix_config="KodiakOPUConfig",
        matrix_kernel=None,
        gate="twin",
        spike_timeout=30,
        timeout=60,
    )
    if not framework_ok:
        with pytest.raises(SystemExit, match="diagnostic-only plan"):
            flow.build_legs(args)
        args.allow_framework_mismatch = True
    binaries = {leg: tmp_path / f"{leg}.elf" for leg in flow.LEGS}
    for leg, elf in binaries.items():
        elf.write_bytes(leg.encode())
    routing = tmp_path / "routing_report.json"
    routing.write_text("{}")
    (tmp_path / "opu_signatures.json").write_text("[]")
    identity = {
        "out_hash": {"algo": "fnv1a32", "elems": 2, "value": 1},
        "prefix_sha256": "x",
        "prefix_elems": 2 if spike_complete else 1,
    }

    def compile_image(cmd, **kwargs):
        assert cmd[cmd.index("--capture-bundle") + 1] == str(bundle)
        assert cmd[cmd.index("--dtype") + 1] == "int8_w8a8"
        assert ("--no-verify" in cmd) == (not framework_ok)
        leg = "device" if "--matrix-unit" in cmd else "control"
        result = {
            "status": "verified" if framework_ok else "ran",
            "binary": str(binaries[leg]),
            "build_hash": leg,
            "verify": {
                "tier_ok": "w8a8",
                "comparison_complete": spike_complete,
                "n_compared": 2 if spike_complete else 1,
                "n_reference": 2,
            }
            if framework_ok
            else {},
            "output_identity": identity,
        }
        if leg == "device":
            result["matrix"] = {
                "unit_instruction_counts": {"OPMACC": 1},
                "routing_report": str(routing),
                "shipped_on_simulator": {
                    "status": "verified",
                    "verify": {
                        "tier_ok": "w8a8",
                        "comparison_complete": spike_complete,
                        "n_compared": 2 if spike_complete else 1,
                        "n_reference": 2,
                    }
                    if framework_ok
                    else {},
                    "bit_exact_vs_twin": True,
                    "output_identity": identity,
                },
            }
        return SimpleNamespace(returncode=0, stdout="log\n" + json.dumps(result, indent=2), stderr="")

    monkeypatch.setattr(flow.subprocess, "run", compile_image)
    monkeypatch.setattr(flow, "_unit_counts", lambda elf, *a: {"OPMACC": 1} if "device" in str(elf) else {})
    job = flow.build_legs(args)
    manifest = flow.load(job)
    assert manifest["bundle"] == str(bundle)
    assert manifest["source"]["reference"]["file"] == "golden.npy"
    assert manifest["source"]["host_cache_identity_sha256"] == "b" * 64
    assert manifest["source"]["host_cache_loaded_unique"] == 1
    assert manifest["source"]["framework_gate_ok"] is framework_ok
    assert manifest["source"]["target_output_complete"] == (spike_complete if framework_ok else None)
    assert Path(manifest["source"]["framework_result"]) == job / "framework_result.json"
    assert (job / "framework_result.json").read_bytes() == framework_path.read_bytes()
    assert manifest["diagnostic_only"] == (not framework_ok or not spike_complete)
    assert manifest["legs"]["device"]["spike_output_identity"] == identity
    assert manifest["legs"]["control"]["spike_output_identity"] == identity
    if manifest["diagnostic_only"]:
        with pytest.raises(SystemExit, match="requires explicit --submit-diagnostic"):
            flow.submit(job)
        with pytest.raises(SystemExit, match="cannot submit"):
            flow.submit(job, submit_diagnostic=True)
        uart = tmp_path / "uartlog"
        uart.write_text("diagnostic output")
        monkeypatch.setattr(
            flow,
            "grade_leg",
            lambda *a, **kw: {
                "status": "verified",
                "out_hash_matches_spike": True,
                "prefix_matches_spike": True,
            },
        )
        monkeypatch.setattr(
            flow,
            "_result",
            lambda manifest, legs: {
                "all_verified": True,
                "provenance": {"bitstream_artifact": {"matched": True, "ok": True}},
            },
        )
        result = flow.collect(job, uartlogs={leg: uart for leg in flow.LEGS}, measurement=False)
        assert result["target_pair_spike_identity"] is True
        assert result["framework_fidelity"] == ("passed" if framework_ok else "failed")
        assert result["acceptance"] is False
        assert result["all_verified"] is False


def test_diagnostic_opu_utilization_uses_verified_hardware_identity(tmp_path):
    utilization = _load("opu_utilization")
    run = tmp_path / "run"
    (run / "firesim").mkdir(parents=True)
    (run / "device" / "routing").mkdir(parents=True)
    signatures = {
        "count": 1,
        "routed": [{"b": 1, "m": 64, "n": 64, "k": 64}],
        "macs": {"routed": 64**3, "not_routed": 0},
    }
    sidecar = run / "device" / "routing" / "opu_signatures.json"
    sidecar.write_text(json.dumps(signatures))
    (run / "device" / "leg.json").write_text(
        json.dumps(
            {
                "board": "firesim_kodiak_opu",
                "harts": 1,
                "vlen": 512,
                "routing_sidecars": {
                    "opu_signatures.json": {"sha256": utilization.hashlib.sha256(sidecar.read_bytes()).hexdigest()}
                },
            }
        )
    )
    result = {
        "all_verified": False,
        "diagnostic_only": True,
        "target_pair_hardware_verified": True,
        "acceptance": False,
        "framework_fidelity": "failed",
        "board": "firesim_kodiak_opu",
        "harts": 1,
        "bitstream": "alveo_u250_firesim_kodiak_opu_1core",
        "device": {"status": "graded_with_problems", "cycles": 1024, "queue_job": 1},
        "control": {"status": "graded_with_problems", "cycles": 2048, "queue_job": 2},
    }
    path = run / "firesim" / "results.json"
    path.write_text(json.dumps(result))
    measured = utilization.measure(run)
    assert measured["whole_inference_effective_peak_fraction"] == pytest.approx(0.25)
    assert measured["diagnostic_only"] is True
    assert measured["acceptance"] is False
    result["target_pair_hardware_verified"] = False
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="hardware identity verification"):
        utilization.measure(run)


@pytest.mark.parametrize(("same", "capture_drift"), [(True, False), (False, False), (True, True)])
def test_independent_rvv_gate_requires_exact_board_output_identity(flow, tmp_path, monkeypatch, same, capture_drift):
    from merlin.common import artifacts

    bundle = tmp_path / "test_model_int8_full"
    bundle.mkdir()
    (bundle / "model.mlir").write_text("module {}")
    np.save(bundle / "golden_w8a8.npy", np.zeros(1, dtype=np.float32))
    monkeypatch.setattr(artifacts, "recaptures_dir", lambda: tmp_path)
    binaries = {leg: tmp_path / f"{leg}.elf" for leg in flow.LEGS}
    for leg, binary in binaries.items():
        binary.write_bytes(leg.encode())
    routing = tmp_path / "routing_report.json"
    routing.write_text("{}")
    (tmp_path / "opu_signatures.json").write_text("[]")

    def compile_image(cmd, **kwargs):
        assert cmd[cmd.index("--board") + 1] == "fs"
        leg = "device" if "--matrix-unit" in cmd else "control"
        assert ("--matrix-gate" in cmd) == (leg == "device")
        if leg == "device":
            assert cmd[cmd.index("--matrix-gate") + 1] == "shipped"
        elif capture_drift:
            (bundle / "model.mlir").write_text("module { changed }")
        identity = {
            "out_hash": {"algo": "fnv1a32", "elems": 1, "value": 1 if same or leg == "control" else 2},
            "prefix_sha256": "same-prefix",
            "prefix_elems": 1,
        }
        shipped = {"status": "verified", "verify": {"gate_ok": True, "tier_ok": "w8a8"}, "output_identity": identity}
        result = {
            "status": "verified",
            "binary": str(binaries[leg]),
            "build_hash": leg,
            "verify": {"gate_ok": True, "tier_ok": "w8a8"},
            "matrix": {
                "unit_instruction_counts": {"OPMACC": 1},
                "routing_report": str(routing),
                "shipped_on_simulator": shipped,
            }
            if leg == "device"
            else None,
            "output_identity": identity,
        }
        return SimpleNamespace(returncode=0, stdout="build log\n" + json.dumps(result, indent=2), stderr="")

    monkeypatch.setattr(flow.subprocess, "run", compile_image)
    monkeypatch.setattr(flow, "_unit_counts", lambda elf, *args: {"OPMACC": 1} if "device" in str(elf) else {})
    args = SimpleNamespace(
        bundle=bundle.name,
        board="fs",
        package="rvv/test",
        harts=1,
        matrix_unit="opu",
        matrix_config="cfg",
        matrix_support_target="kodiak",
        matrix_kernel=None,
        gate="independent-rvv",
        spike_timeout=40,
        timeout=7200,
    )
    if not same:
        with pytest.raises(SystemExit, match="do not match bit for bit"):
            flow.build_legs(args)
        return
    if capture_drift:
        with pytest.raises(SystemExit, match="capture bundle changed"):
            flow.build_legs(args)
        return
    job = flow.build_legs(args)
    plan = flow.load(job)
    assert plan["capture_sha256"] == flow._capture_sha256(bundle)
    assert plan["legs"]["device"]["spike_gate"]["method"] == "independent-rvv"
    assert plan["legs"]["device"]["spike_gate"]["independent_rvv_bit_exact"] is True
    sidecar = Path(plan["legs"]["device"]["routing_sidecars"]["routing_report.json"]["path"])
    sidecar.write_text("changed")
    with pytest.raises(SystemExit, match="routing sidecar changed"):
        flow.collect(job)


def test_collect_rejects_capture_changed_since_pair_build(flow, tmp_path):
    bundle, _ = _bundle(tmp_path)
    elf = tmp_path / "device.elf"
    elf.write_bytes(b"image")
    job = tmp_path / "job"
    flow.plan(
        job,
        board="fs",
        name="test_model",
        bundle=bundle,
        legs={"device": {"elf": elf}},
        capture_sha256=flow._capture_sha256(bundle),
    )
    np.save(bundle / "golden_w8a8.npy", np.ones(24, dtype=np.float32))
    with pytest.raises(SystemExit, match="capture bundle changed"):
        flow.collect(job)


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

    # Early toolchain failures emit a JSON object at stdout offset zero. Preserve its reason.
    monkeypatch.setattr(
        flow.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1,
            stdout='{"status": "not_run", "reason": "Zephyr toolchain unavailable"}',
            stderr="",
        ),
    )
    with pytest.raises(SystemExit, match="not_run"):
        flow.build_legs(args)


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


def test_source_grade_refuses_a_partial_torchao_prefix_despite_a_whole_spike_hash(flow):
    from merlin.runtime.backends import output_identity
    from merlin.runtime.backends import zephyr_model as zm

    golden = np.linspace(-2.0, 3.0, 24, dtype=np.float32)
    prefix = golden[:4]
    uart = _uart(prefix, cycles=10, build_hash="source")
    prefix_hash = zm.out_hash_fnv1a32(prefix)
    whole_hash = zm.out_hash_fnv1a32(golden)
    uart = uart.replace(f"HASH fnv1a32 {prefix.size} {prefix_hash}", f"HASH fnv1a32 {golden.size} {whole_hash}")
    parsed = zm._parse_console(uart, 0)
    row = {"build_hash": "source", "spike_output_identity": output_identity.output_identity(parsed)}
    hw = {"RuntimeHWConfig": HW}
    legacy = flow.grade_leg(uart, row=row, refs={"w8a8": golden}, hw_config=HW, hw_summary=hw)
    assert legacy["grade"]["tier_ok"] == "w8a8"
    source = flow.grade_leg(uart, row=row, refs={"w8a8": golden}, hw_config=HW, hw_summary=hw, require_full_output=True)
    assert source["out_hash_matches_spike"] is True
    assert source["prefix_matches_spike"] is True
    assert source["source_output_complete"] is False
    assert source["status"] == "graded_with_problems"
    assert any("complete TorchAO output" in p for p in source["problems"])
    assert not flow._complete_source_output(
        {"comparison_complete": True, "n_compared": golden.size, "n_reference": golden.size},
        row["spike_output_identity"],
        golden.size,
    )

    full_uart = _uart(golden, cycles=10, build_hash="source")
    full_identity = output_identity.output_identity(zm._parse_console(full_uart, 0))
    full = flow.grade_leg(
        full_uart,
        row={"build_hash": "source", "spike_output_identity": full_identity},
        refs={"w8a8": golden},
        hw_config=HW,
        hw_summary=hw,
        require_full_output=True,
    )
    assert full["source_output_complete"] is True
    assert full["status"] == "verified"


def test_source_partial_output_keeps_hardware_identity_diagnostic(flow, tmp_path, monkeypatch):
    from merlin.runtime.backends import output_identity
    from merlin.runtime.backends import zephyr_model as zm

    bundle, _ = _bundle(tmp_path)
    golden = np.load(bundle / "golden.npy")
    prefix = golden[:4]
    uart = _uart(prefix, cycles=10, build_hash="aaaa")
    uart = uart.replace(
        f"HASH fnv1a32 {prefix.size} {zm.out_hash_fnv1a32(prefix)}",
        f"HASH fnv1a32 {golden.size} {zm.out_hash_fnv1a32(golden)}",
    )
    identity = output_identity.output_identity(zm._parse_console(uart, 0))
    job = _plan(flow, tmp_path, bundle)
    manifest = flow.load(job)
    for leg in flow.LEGS:
        manifest["legs"][leg]["spike_output_identity"] = identity
    framework = tmp_path / "framework.json"
    cache = _strict_host_cache(flow, tmp_path, monkeypatch)
    framework.write_text(json.dumps({"host_cache": cache}))
    manifest["source"] = {
        "dtype": "int8_w8a8",
        "framework_gate_ok": False,
        "reference": {"sha256": flow._sha256(bundle / "golden.npy")},
        "reference_elems": int(golden.size),
        "capture_sha256": flow._capture_sha256(bundle),
        "framework_result": str(framework),
        "framework_result_sha256": flow._sha256(framework),
        "host_cache_identity_sha256": cache["identity"]["sha256"],
        "host_cache_loaded_unique": 1,
        "target_output_complete": None,
        "diagnostic_reasons": ["framework_fidelity_failed_or_incomplete"],
        "diagnostic_only": True,
    }
    manifest["diagnostic_only"] = True
    flow._save(job, manifest)
    monkeypatch.setattr(flow, "_bitstream_artifact", lambda uri: {"matched": True, "ok": True})
    dev = _results_dir(tmp_path, "d", uart)
    ctl = _results_dir(tmp_path, "c", uart.replace("METRIC build_hash aaaa", "METRIC build_hash bbbb"))
    result = flow.collect(job, uartlogs={"device": dev / "uartlog", "control": ctl / "uartlog"}, measurement=False)
    assert result["target_pair_hardware_verified"] is True
    assert result["target_pair_spike_identity"] is True
    assert result["device"]["source_output_complete"] is False
    assert result["acceptance"] is False
    assert result["all_verified"] is False


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
