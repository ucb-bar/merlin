from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.compare.host_experiment import (
    _calibration_private_capsule,
    _grader_package_tree_identity,
    _validate_calibration_semantics,
)


SCRIPT = (repo_root() /
          "merlin/experiments/cpu_host_compiler_v0/calibrate_search_costs.py")
NOISE_SCRIPT = SCRIPT.with_name("calibrate_search_noise.py")
SPACE = SCRIPT.with_name("optimization_space_v1.yaml")


def _module():
    spec = importlib.util.spec_from_file_location("cpu_host_search_cost_calibrator", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Clock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        self.value += 1_000
        return self.value


class _Runner:
    @staticmethod
    def _candidate(actions):
        assert actions == []
        return {"candidate_sha256": "c" * 64, "actions": []}

    @staticmethod
    def select_semantic_sample(rows, *, per_family, families):
        buckets = {}
        for row in rows:
            family = row["family"]
            if family == "contraction":
                key = family, row["operation"], row["dtype"], row["layout"]
            elif family == "runtime_parallel":
                key = family, row["operation"], str(row["core_count"])
            else:
                key = family, row["operation"], row["dtype"]
            buckets.setdefault(key, []).append(row)
        representatives = [min(values, key=lambda row: row["sha256"])
                           for _, values in sorted(buckets.items())]
        selected = []
        for family in sorted(families):
            selected.extend(sorted(
                (row for row in representatives if row["family"] == family),
                key=lambda row: row["sha256"])[:per_family])
        return selected


class _Grader:
    def __init__(self) -> None:
        self.prebuild_calls = 0
        self.evaluation_submissions = []
        self.evaluation_public_rows = []
        self.current_k1 = None

    def prepare_prebuilt_search_package(self, *, submission, destination, build_override):
        self.prebuild_calls += 1
        assert build_override == ["/bin/true"]
        before_manifest = hashlib.sha256((submission / "manifest.yaml").read_bytes()).hexdigest()
        before_tree = _grader_package_tree_identity(submission)
        shutil.copytree(submission, destination)
        manifest = yaml.safe_load((destination / "manifest.yaml").read_text())
        manifest["build"] = {"command": ["/bin/true"]}
        (destination / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        private_manifest = hashlib.sha256((destination / "manifest.yaml").read_bytes()).hexdigest()
        sealed_tree = _grader_package_tree_identity(destination)
        return {
            "version": 1, "authority": "driver_private_prebuild",
            "submitted_manifest_sha256": before_manifest,
            "private_manifest_sha256": private_manifest,
            "real_build_commands": [["cmake", "--build", "build"]],
            "real_build_logs": [{"returncode": 0}],
            "prebuild_tree_sha256": before_tree,
            "built_tree_sha256": "2" * 64,
            "sealed_prebuilt_tree_sha256": sealed_tree,
            "submitted_entrypoint_identity": None,
            "built_entrypoint_identity": [0o755, "4" * 64],
            "private_build_override": ["/bin/true"],
            "policy_sha256": "5" * 64,
        }

    def _build(self, _submission):
        return {"version": 1}, {"stdout": "built"}

    def _compile_one(self, *_args, **_kwargs):
        return {"ok": True, "source_sha256": "d" * 64}

    def _grade_spike(self, *_args, **_kwargs):
        return {
            "status": "pass", "kernel_text_sha256": "e" * 64,
            "checks": {
                "rvv_correctness": True, "instruction_evidence": True,
                "vlen_256": True, "cycle_measurement": True,
            },
        }

    @staticmethod
    def _search_spike_correct(record):
        return record.get("status") == "pass" and all(record["checks"].values())

    def _grade_k1(self, *_args, **_kwargs):
        assert self.current_k1 is not None
        return {
            "capsule": self.current_k1["id"], "family": self.current_k1["family"],
            "mode": "rvv", "harts": 1,
            "status": "pass", "checks": {"numeric_correctness": True},
            "kernel_text_sha256": "f" * 64, "seed": 7,
            "metrics": {"wall_ns": 100, "calls": 1},
            "monitor": {"returncode": 0}, "receipt_nonce": 11,
            "local_sha256": "6" * 64, "remote_sha256": "6" * 64,
            "board_wall_seconds": 0.1, "ssh_returncode": 0,
        }

    def evaluate_public_policy_k1(self, **kwargs):
        assert kwargs["repeats"] == 6
        self.evaluation_submissions.append(kwargs["submission"])
        self.evaluation_public_rows.append(list(kwargs["public_rows"]))
        observations = []
        for row in kwargs["capsules"]:
            self.current_k1 = row
            orders = ["parent_candidate", "candidate_parent", "candidate_parent",
                      "parent_candidate", "parent_candidate", "candidate_parent"]
            state = {
                "authority": "driver_ssh_sysfs_procfs",
                "controller_monotonic_ns": 1, "returncode": 0,
                "state": {
                    "online": "0-7",
                    "governors": {str(index): "performance" for index in range(8)},
                    "frequencies_khz": {str(index): "1600000" for index in range(8)},
                    "temperatures_millic": {"0": "40000"},
                    "loadavg": "0.1 0.1 0.1 1/1 1",
                },
            }
            condition_pairs = []
            for pair_index, order in enumerate(orders):
                evidence = {side: self._grade_k1() for side in ("parent", "candidate")}
                # The curated space is bound to the current live K1-derived 16.0% margin.
                # Keep this synthetic receipt internally self-consistent with that
                # derivation instead of pretending an exact 1.0 ratio can produce it.
                elapsed = {"parent": 1154, "candidate": 1000}
                for side, value in evidence.items():
                    value["metrics"]["wall_ns"] = elapsed[side]
                condition_pairs.append({
                    "pair_id": pair_index, "attempt_id": pair_index, "order": order,
                    "seed": 7, "settle_probes": [deepcopy(state)],
                    "measurements": {side: {
                        "elapsed_ns": elapsed[side], "calls": 1, "seed": 7,
                        "evidence": value,
                    } for side, value in evidence.items()},
                    "before": deepcopy(state), "after": deepcopy(state), "valid": True,
                })
            observations.append({
                "capsule_id": row["id"], "family": row["family"],
                "correctness_ok": True, "board_condition_pairs": condition_pairs,
                "excluded_board_condition_pairs": [], "pair_orders": orders,
                "k1_program_count": 12,
                "baseline_elapsed_ns": [1154] * 6, "baseline_calls": [1] * 6,
                "candidate_elapsed_ns": [1000] * 6, "candidate_calls": [1] * 6,
                "baseline_code_sha256": "f" * 64,
                "candidate_code_sha256": "f" * 64,
                "spike_gates": {side: {
                    "compile_ok": True, "k1_compile_ok": True, "passed": True,
                    "kernel_text_sha256": "f" * 64,
                    "checks": {"rvv_correctness": True, "instruction_evidence": True,
                               "vlen_256": True, "cycle_measurement": True},
                } for side in ("parent", "candidate")},
            })
        return observations

    def evaluate_public_policy_spike(self, **kwargs):
        self.evaluation_submissions.append(kwargs["submission"])
        self.evaluation_public_rows.append(list(kwargs["public_rows"]))
        observations = []
        for row in kwargs["capsules"]:
            for _label in ("parent", "candidate"):
                self._build(kwargs["submission"])
                self._compile_one(kwargs["submission"], {}, row, "rvv")
                self._grade_spike()
            observations.append({
                "capsule_id": row["id"], "family": row["family"],
                "correctness_ok": True,
                "baseline_code_sha256": "e" * 64,
                "candidate_code_sha256": "e" * 64,
            })
        return observations

    def evaluate_public_policy_confirmation_stages(self, **kwargs):
        self.evaluation_submissions.append(kwargs["submission"])
        self.evaluation_public_rows.append(list(kwargs["public_rows"]))
        observations = []
        for row in kwargs["capsules"]:
            for _label in ("parent", "candidate"):
                self._build(kwargs["submission"])
                self._compile_one(kwargs["submission"], {}, row, "rvv")
                if row["family"] == "runtime_parallel":
                    self._compile_one(kwargs["submission"], {}, row, "rvv_multicore")
                self._grade_spike()
            observations.append({
                "capsule_id": row["id"], "family": row["family"],
                "correctness_ok": True,
                "baseline_code_sha256": "e" * 64,
                "candidate_code_sha256": "e" * 64,
                "calibration_authority": "exact_confirmation_pre_k1_stages_without_k1",
            })
        return observations


def _inputs(tmp_path: Path):
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "manifest.yaml").write_text("version: 1\n", encoding="utf-8")
    space = yaml.safe_load(SPACE.read_text(encoding="utf-8"))
    rows = []
    family_inputs = {
        "contraction": (("matmul", "batch_matmul"), "fp32", "row_row",
                        lambda index: {"M": 8 + index, "N": 16 + index, "K": 4 + index}, {}),
        "elementwise_map": (("add", "multiply"), "fp32", "contiguous",
                            lambda index: {"length": 64 + index}, {}),
        "reduction": (("sum", "max"), "fp32", "contiguous",
                      lambda index: {"length": 64 + index}, {}),
        "movement_layout": (("copy", "transpose2d"), "fp32", "contiguous",
                            lambda index: {"working_set_bytes": 256 + 4 * index}, {}),
        "fusion_epilogue": (("matmul_bias", "matmul_bias_relu"), "fp32", "row_row",
                            lambda index: {"M": 8 + index, "N": 16 + index,
                                           "K": 4 + index}, {}),
        "runtime_parallel": (("producer_consumer", "producer_consumer"),
                             "fp32", "contiguous",
                             lambda index: {"work_items": 64 + index}, {"reuse_count": 2}),
    }
    for family in space["confirmation_families"]:
        for index in range(2):
            operations, dtype, layout, shape, state = family_inputs[family]
            rows.append({
                "id": f"{family}-{index}", "sha256": hashlib.sha256(
                    f"{family}-{index}".encode()).hexdigest(),
                "split": "train", "family": family,
                "operation": operations[index], "dtype": dtype,
                "layout": layout, "core_count": index + 1,
                "shape": shape(index), "state": deepcopy(state),
            })
    public_train = tmp_path / "train.jsonl"
    public_train.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return submission, public_train, space


def _write_noise_predecessor(module, submission: Path, space: dict, path: Path) -> Path:
    protocol_sha = "a" * 64
    raw_tree = module._tree_sha256(submission)
    raw_package = module._package_tree_sha256(submission)
    value = {
        "version": 2,
        "kind": "cpu_host_k1_order_balanced_aa_noise_calibration",
        "status": "pass",
        "calibration_protocol_sha256": protocol_sha,
        "prebuild_input_tree_sha256": raw_tree,
        "prebuild_input_package_sha256": raw_package,
        "derived_noise_margin": float(space["noise_margin"]),
        "calibration_lineage": {
            "version": 1,
            "stage": "noise_pre_result",
            "pre_result_protocol_sha256": protocol_sha,
            "raw_input_tree_sha256": raw_tree,
            "raw_input_package_sha256": raw_package,
            "output_field": "noise_margin",
        },
    }
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _produce(tmp_path: Path, kind: str):
    module = _module()
    submission, public_train, space = _inputs(tmp_path)
    grader = _Grader()
    prebuilt = tmp_path / f"prebuilt-{kind}"
    noise_authority = _write_noise_predecessor(
        module, submission, space, tmp_path / f"noise-{kind}.json")
    broker = module._load(f"test_broker_{kind}", SCRIPT.with_name("trusted_search_broker.py"))
    artifact = module.run_calibration(
        kind=kind, submission=submission, public_train=public_train, space_path=SPACE,
        prebuilt_destination=prebuilt, noise_authority=noise_authority,
        grader=grader, runner=_Runner(), broker=broker,
        calibration_nonce=b"\x12" * 32,
        toolchain_identity=_test_toolchain_identity(), monotonic_ns=_Clock())
    return module, artifact, grader, prebuilt, public_train, space


def _test_toolchain_identity():
    path = Path(sys.executable).resolve()
    identity = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "mode": path.stat().st_mode & 0o777}
    return {name: dict(identity) for name in (
        "python", "bwrap", "prebuild_command_0", "private_build_override_0",
        "spike_gcc", "spike_spike",
        "spike_objdump", "k1_clang", "k1_objcopy", "ssh", "scp")}


def _problems(artifact, label, public_train, space):
    lineage = artifact.get("calibration_lineage") if isinstance(artifact, dict) else None
    noise_path = (Path(lineage["noise_authority"]) if isinstance(lineage, dict) and
                  lineage.get("stage") == "cost_post_noise_result" else None)
    noise_value = json.loads(noise_path.read_text()) if noise_path is not None else artifact
    noise_sha = (hashlib.sha256(noise_path.read_bytes()).hexdigest()
                 if noise_path is not None else None)
    return _validate_calibration_semantics(
        label=label, value=artifact,
        train_sha256=hashlib.sha256(public_train.read_bytes()).hexdigest(),
        source_sha256=artifact["source_sha256"], space=space,
        train_rows=[json.loads(line) for line in public_train.read_text().splitlines()],
        noise_authority=noise_value, noise_authority_sha256=noise_sha)


@pytest.mark.parametrize("kind,label", [
    ("k1-program", "k1_calibration"),
    ("spike-screen", "spike_calibration"),
    ("confirmation-overhead", "confirmation_overhead_calibration"),
])
def test_mocked_producer_round_trips_exact_semantic_validator(
        tmp_path: Path, kind: str, label: str) -> None:
    _module_value, artifact, grader, prebuilt, public_train, space = _produce(
        tmp_path, kind)

    assert artifact["status"] == "pass", artifact
    assert artifact["heldout_opened"] is False
    assert artifact["paid_work"] is False
    assert artifact["public_split_sha256"] == hashlib.sha256(
        public_train.read_bytes()).hexdigest()
    assert set(artifact["source_sha256"]) == {
        "cost_calibrator", "noise_calibrator", "grader", "search_runner",
        "trusted_harness", "k1_monitor", "search_space", "trusted_evaluator",
        "trusted_broker", "k1_adapter",
    }
    assert _problems(artifact, label, public_train, space) == []
    assert grader.prebuild_calls == 1
    assert grader.evaluation_submissions
    assert all(path == prebuilt for path in grader.evaluation_submissions)
    assert all(len(rows) == 12 for rows in grader.evaluation_public_rows)
    assert yaml.safe_load((prebuilt / "manifest.yaml").read_text())["build"] == {
        "command": ["/bin/true"]}

    if kind == "k1-program":
        assert len(artifact["programs"]) == 12
        assert all(row["evidence"]["receipt_nonce"] == 11
                   for row in artifact["programs"])
    elif kind == "spike-screen":
        assert len(artifact["observations"]) == 12
        assert artifact["end_monotonic_ns"] > artifact["start_monotonic_ns"]
    else:
        assert set(artifact["stage_observations"]) == {
            "package_build", "compiler_invocation", "spike_check"}
        assert {name: len(rows) for name, rows in artifact["stage_observations"].items()} == {
            "package_build": 12, "compiler_invocation": 14, "spike_check": 12}
        assert all(all(row["evidence"] for row in rows)
                   for rows in artifact["stage_observations"].values())


def test_k1_authority_accepts_predeclared_board_condition_replacement(
        tmp_path: Path) -> None:
    module, artifact, _grader, _prebuilt, public_train, space = _produce(
        tmp_path, "k1-program")
    observation = artifact["trusted_evaluation_observations"][0]
    excluded = deepcopy(observation["board_condition_pairs"][0])
    excluded["valid"] = False
    observation["excluded_board_condition_pairs"] = [excluded]
    observation["k1_program_count"] = 14
    for _ in range(2):
        row = deepcopy(artifact["programs"][-1])
        row["index"] = len(artifact["programs"])
        artifact["programs"].append(row)
    artifact["statistics"] = module._summary([
        float(row["total_seconds"]) for row in artifact["programs"]])

    assert _problems(artifact, "k1_calibration", public_train, space) == []


def test_default_path_is_public_train_and_no_heldout_interface(tmp_path: Path) -> None:
    module = _module()
    expected = (repo_root() /
                "out/artifacts/rvv-development-corpus/k1_cpu/v2/latest/public/train.jsonl")
    assert module.DEFAULT_PUBLIC_TRAIN == expected
    assert "heldout" not in inspect.signature(module.calibrate).parameters
    assert "heldout" not in inspect.signature(module.run_calibration).parameters

    submission, public_train, _space = _inputs(tmp_path)
    rows = [json.loads(line) for line in public_train.read_text().splitlines()]
    rows[0]["split"] = "heldout"
    public_train.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    artifact = module.run_calibration(
        kind="spike-screen", submission=submission, public_train=public_train,
        space_path=SPACE, prebuilt_destination=tmp_path / "prebuilt",
        noise_authority=tmp_path / "unused-noise.json",
        grader=_Grader(), runner=_Runner(), monotonic_ns=_Clock())
    assert artifact["status"] == "fail"
    assert artifact["heldout_opened"] is False
    assert "public train rows only" in artifact["error"]


def test_main_uses_aet_run_lifecycle_and_metrics_directory(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    submission, public_train, _space = _inputs(tmp_path)
    run_dir = tmp_path / "out" / "runs" / "20260831T010203Z_cost_calibration"
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "artifacts_dir").mkdir()
    handle = SimpleNamespace(run_dir=run_dir)
    calls = []

    def fake_start_run(**kwargs):
        calls.append(("start", kwargs))
        return handle

    def fake_finish_run(received, **kwargs):
        assert received is handle
        calls.append(("finish", kwargs))

    received = {}

    def fake_run_calibration(**kwargs):
        received.update(kwargs)
        return {"status": "pass"}

    monkeypatch.setattr(module, "start_run", fake_start_run)
    monkeypatch.setattr(module, "finish_run", fake_finish_run)
    monkeypatch.setattr(module, "run_calibration", fake_run_calibration)
    result = module.main([
        "--kind", "spike-screen", "--submission", str(submission),
        "--public-train", str(public_train), "--space", str(SPACE),
        "--noise-authority", str(tmp_path / "noise.json"),
    ])

    artifact = run_dir / "metrics" / "spike_screen_calibration.json"
    assert result == 0 and artifact.is_file()
    assert calls[0][0] == "start" and calls[-1][0] == "finish"
    assert calls[0][1]["method"] == "spike-screen-calibration"
    assert calls[-1][1]["summary"]["artifact"] == str(artifact.resolve())
    retained = run_dir / "artifacts_dir" / "prebuild_input_submission"
    assert received["submission"] == retained.resolve()
    assert retained.is_dir()
    assert module._tree_sha256(retained) == module._tree_sha256(submission)
    assert calls[-1][1]["summary"]["prebuild_input_submission"] == str(retained.resolve())


def test_retained_input_survives_transient_source_and_rejects_links(tmp_path: Path) -> None:
    module = _module()
    submission, _public_train, _space = _inputs(tmp_path)
    retained = module._retain_input_submission(submission, tmp_path / "authority" / "input")
    expected = module._tree_sha256(retained)
    shutil.rmtree(submission)
    assert retained.is_dir() and module._tree_sha256(retained) == expected

    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "manifest.yaml").write_text("version: 1\n", encoding="utf-8")
    (linked / "escape").symlink_to(retained / "manifest.yaml")
    with pytest.raises(ValueError, match="symlinks are forbidden"):
        module._retain_input_submission(linked, tmp_path / "authority" / "linked")


def test_alternate_optimization_space_fails_before_prebuild(tmp_path: Path) -> None:
    module = _module()
    submission, public_train, _space = _inputs(tmp_path)
    alternate = tmp_path / "relaxed-space.yaml"
    alternate.write_bytes(SPACE.read_bytes())
    grader = _Grader()
    artifact = module.run_calibration(
        kind="spike-screen", submission=submission, public_train=public_train,
        space_path=alternate, prebuilt_destination=tmp_path / "prebuilt",
        noise_authority=tmp_path / "unused-noise.json",
        grader=grader, runner=_Runner(), monotonic_ns=_Clock())
    assert artifact["status"] == "fail"
    assert "exact frozen optimization_space_v1.yaml path" in artifact["error"]
    assert grader.prebuild_calls == 0


def test_validator_private_shape_mirror_equals_broker(tmp_path: Path) -> None:
    module = _module()
    _submission, public_train, _space = _inputs(tmp_path)
    broker = module._load("test_broker_equivalence", SCRIPT.with_name("trusted_search_broker.py"))
    nonce = b"\xa5" * 32
    for row in (json.loads(line) for line in public_train.read_text().splitlines()):
        assert _calibration_private_capsule(row, nonce=nonce, phase="confirm") == \
            broker._private_capsule(row, secret=nonce, phase="confirm", split="train")


def test_real_grader_confirmation_api_has_exact_broker_stage_multiplicity(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    grader = module._load("real_grader_confirmation_api", SCRIPT.with_name("grader.py"))
    submission, public_train, space = _inputs(tmp_path)
    (submission / "policy.yaml").write_text("{}\n", encoding="utf-8")
    (submission / "compiler.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    (submission / "compiler.py").chmod(0o755)
    (submission / "manifest.yaml").write_text(yaml.safe_dump({
        "version": 1, "build": {"command": ["/bin/true"]},
        "compiler": {"command": [
            "compiler.py", "--input", "{input_mlir}", "--output-dir", "{output_dir}",
            "--mode", "{mode}", "--harts", "{harts}", "--vlen-bits", "{vlen_bits}"]},
        "policy": "policy.yaml",
    }, sort_keys=False), encoding="utf-8")
    public_rows = [json.loads(line) for line in public_train.read_text().splitlines()]
    panel = _Runner.select_semantic_sample(
        public_rows, per_family=1, families=list(space["confirmation_families"]))
    counts = {"build": 0, "compile": [], "spike": 0, "public_context": []}
    original_codes = grader._codes

    def fake_build(package):
        counts["build"] += 1
        return yaml.safe_load((package / "manifest.yaml").read_text()), {"commands": []}

    def fake_compile(_package, _manifest, row, mode, _operation_codes, _root):
        counts["compile"].append((row["id"], mode))
        return {"ok": True, "mode": mode}

    def fake_spike(row, _compiled, _operation_codes, _root):
        counts["spike"] += 1
        return {
            "status": "pass", "kernel_text_sha256": "8" * 64,
            "checks": {"rvv_correctness": True, "instruction_evidence": True,
                       "vlen_256": True, "cycle_measurement": True},
        }

    def observed_codes(rows, key):
        counts["public_context"].append([row["id"] for row in rows])
        return original_codes(rows, key)

    monkeypatch.setattr(grader, "_build", fake_build)
    monkeypatch.setattr(grader, "_compile_one", fake_compile)
    monkeypatch.setattr(grader, "_grade_spike", fake_spike)
    monkeypatch.setattr(grader, "_codes", observed_codes)
    monkeypatch.setattr(
        grader, "_grade_k1", lambda *_args, **_kwargs: pytest.fail("K1 must not run"))
    candidate = {"candidate_sha256": "9" * 64, "actions": []}
    observations = []
    for capsule in panel:
        observations.extend(grader.evaluate_public_policy_confirmation_stages(
            submission=submission, capsules=[capsule], parent=candidate,
            candidate=candidate, public_rows=public_rows))

    assert len(observations) == 6
    assert counts["build"] == 12
    assert len(counts["compile"]) == 14
    assert counts["spike"] == 12
    assert sum(mode == "rvv_multicore" for _capsule, mode in counts["compile"]) == 2
    assert counts["public_context"] == [[row["id"] for row in public_rows]] * 6


def test_noise_calibration_uses_prebuilt_private_panel_and_full_public_context(
        tmp_path: Path) -> None:
    cost_module = _module()
    noise = cost_module._load("noise_calibrator_under_test", NOISE_SCRIPT)
    submission, public_train, space = _inputs(tmp_path)
    grader = _Grader()
    broker = cost_module._load("noise_broker_under_test", SCRIPT.with_name(
        "trusted_search_broker.py"))
    artifact = noise.calibrate(
        submission=submission, public_train=public_train, space_path=SPACE,
        prebuilt_destination=tmp_path / "noise-prebuilt", grader=grader, runner=_Runner(),
        broker=broker, calibration_nonce=b"\x34" * 32,
        toolchain_identity=_test_toolchain_identity())
    rows = [json.loads(line) for line in public_train.read_text().splitlines()]

    assert artifact["status"] == "pass"
    assert grader.prebuild_calls == 1
    assert len(grader.evaluation_public_rows) == 1
    assert grader.evaluation_public_rows[0] == rows
    assert all(observation["capsule_id"].startswith("private-")
               for observation in artifact["observations"])
    assert _validate_calibration_semantics(
        label="noise_calibration", value=artifact,
        train_sha256=hashlib.sha256(public_train.read_bytes()).hexdigest(),
        source_sha256=artifact["source_sha256"], space=space, train_rows=rows) == []


@pytest.mark.parametrize("tamper", [
    "public_context", "private_shape", "toolchain", "raw_elapsed", "raw_calls",
    "pair_ratio", "maximum_pair_deviation", "lineage", "consistent_summary",
    "pair_receipt", "attempt_transcript", "early_ready_probe",
])
def test_noise_calibration_binding_tampering_is_rejected(
        tmp_path: Path, tamper: str) -> None:
    cost_module = _module()
    noise = cost_module._load(f"noise_tamper_{tamper}", NOISE_SCRIPT)
    submission, public_train, space = _inputs(tmp_path)
    broker = cost_module._load(f"noise_tamper_broker_{tamper}", SCRIPT.with_name(
        "trusted_search_broker.py"))
    artifact = noise.calibrate(
        submission=submission, public_train=public_train, space_path=SPACE,
        prebuilt_destination=tmp_path / "noise-prebuilt", grader=_Grader(), runner=_Runner(),
        broker=broker, calibration_nonce=b"\x34" * 32,
        toolchain_identity=_test_toolchain_identity())
    damaged = deepcopy(artifact)
    if tamper == "public_context":
        damaged["public_context"]["capsule_ids"].pop()
    elif tamper == "private_shape":
        shape = damaged["private_shape_calibration"]["records"][0]["private"]["shape"]
        shape[next(iter(shape))] += 1
    elif tamper == "toolchain":
        damaged["toolchain_identity"]["k1_clang"]["sha256"] = "0" * 64
    elif tamper == "raw_elapsed":
        damaged["observations"][0]["baseline_elapsed_ns"][0] += 1
    elif tamper == "raw_calls":
        damaged["observations"][0]["candidate_calls"][0] += 1
    elif tamper == "pair_ratio":
        damaged["pairs"][0]["speedup_ratio"] = 1.01
    elif tamper == "maximum_pair_deviation":
        damaged["maximum_absolute_pair_deviation"] = 0.5
    elif tamper == "lineage":
        damaged["calibration_lineage"]["stage"] = "cost_post_noise_result"
    elif tamper == "consistent_summary":
        damaged["observations"][0]["baseline_elapsed_ns"][0] = 101
        ratio = 1.01
        damaged["pairs"][0]["speedup_ratio"] = ratio
        damaged["pairs"][0]["absolute_unit_deviation"] = ratio - 1.0
        maximum_log = abs(math.log(ratio))
        damaged["maximum_absolute_pair_deviation"] = ratio - 1.0
        damaged["maximum_absolute_log_ratio"] = maximum_log
        damaged["padded_log_half_width"] = maximum_log + 0.005
    elif tamper == "pair_receipt":
        damaged["observations"][0]["board_condition_pairs"][0]["measurements"][
            "parent"]["evidence"]["metrics"]["wall_ns"] += 1
    elif tamper == "attempt_transcript":
        damaged["observations"][0]["board_condition_pairs"][1]["attempt_id"] = 0
    else:
        pair = damaged["observations"][0]["board_condition_pairs"][0]
        pair["settle_probes"].insert(0, deepcopy(pair["settle_probes"][0]))
    rows = [json.loads(line) for line in public_train.read_text().splitlines()]
    assert _validate_calibration_semantics(
        label="noise_calibration", value=damaged,
        train_sha256=hashlib.sha256(public_train.read_bytes()).hexdigest(),
        source_sha256=artifact["source_sha256"], space=space, train_rows=rows)


@pytest.mark.parametrize("tamper", [
    "noise_sha", "raw_tree", "final_space", "derived_margin", "predecessor_stage",
])
def test_cost_calibration_lineage_tampering_is_rejected(
        tmp_path: Path, tamper: str) -> None:
    _module_value, artifact, _grader, _prebuilt, public_train, space = _produce(
        tmp_path, "spike-screen")
    damaged = deepcopy(artifact)
    lineage = damaged["calibration_lineage"]
    if tamper == "noise_sha":
        lineage["noise_authority_sha256"] = "0" * 64
    elif tamper == "raw_tree":
        lineage["raw_input_tree_sha256"] = "0" * 64
    elif tamper == "final_space":
        lineage["final_space_sha256"] = "0" * 64
    elif tamper == "derived_margin":
        lineage["derived_noise_margin"] += 0.001
    else:
        lineage["predecessor_stage"] = "cost_pre_noise_result"
    assert _problems(damaged, "spike_calibration", public_train, space)


def test_cost_producer_rejects_noise_from_a_different_raw_input_tree(
        tmp_path: Path) -> None:
    module = _module()
    submission, public_train, space = _inputs(tmp_path)
    different = tmp_path / "different"
    shutil.copytree(submission, different)
    (different / "extra").write_text("different\n", encoding="utf-8")
    noise_authority = _write_noise_predecessor(
        module, different, space, tmp_path / "noise.json")
    grader = _Grader()
    artifact = module.run_calibration(
        kind="spike-screen", submission=submission, public_train=public_train,
        space_path=SPACE, prebuilt_destination=tmp_path / "prebuilt",
        noise_authority=noise_authority, grader=grader, runner=_Runner(),
        toolchain_identity=_test_toolchain_identity(), monotonic_ns=_Clock())
    assert artifact["status"] == "fail"
    assert "share one raw compiler input tree" in artifact["error"]
    assert grader.prebuild_calls == 0


def test_cost_producer_rejects_mode_only_raw_package_mismatch(tmp_path: Path) -> None:
    module = _module()
    submission, public_train, space = _inputs(tmp_path)
    different = tmp_path / "mode-different"
    shutil.copytree(submission, different)
    (different / "manifest.yaml").chmod(0o400)
    assert module._tree_sha256(different) == module._tree_sha256(submission)
    assert module._package_tree_sha256(different) != module._package_tree_sha256(submission)
    noise_authority = _write_noise_predecessor(
        module, different, space, tmp_path / "noise-mode.json")
    grader = _Grader()
    artifact = module.run_calibration(
        kind="spike-screen", submission=submission, public_train=public_train,
        space_path=SPACE, prebuilt_destination=tmp_path / "prebuilt-mode",
        noise_authority=noise_authority, grader=grader, runner=_Runner(),
        toolchain_identity=_test_toolchain_identity(), monotonic_ns=_Clock())
    assert artifact["status"] == "fail"
    assert "mode-sensitive raw package" in artifact["error"]
    assert grader.prebuild_calls == 0


def test_noise_calibration_rejects_alternate_space_before_prebuild(tmp_path: Path) -> None:
    cost_module = _module()
    noise = cost_module._load("noise_calibrator_space_test", NOISE_SCRIPT)
    submission, public_train, _space = _inputs(tmp_path)
    alternate = tmp_path / "space.yaml"
    alternate.write_bytes(SPACE.read_bytes())
    with pytest.raises(ValueError, match="exact frozen optimization_space_v1.yaml path"):
        noise.calibrate(
            submission=submission, public_train=public_train, space_path=alternate,
            prebuilt_destination=tmp_path / "prebuilt", grader=_Grader(), runner=_Runner())


def test_noise_main_retains_unbuilt_input_in_aet_run(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cost_module = _module()
    noise = cost_module._load("noise_main_retention", NOISE_SCRIPT)
    submission, public_train, _space = _inputs(tmp_path)
    run_dir = tmp_path / "noise-run"
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "artifacts_dir").mkdir()
    handle = SimpleNamespace(run_dir=run_dir)
    received = {}
    summaries = []

    monkeypatch.setattr(noise, "start_run", lambda **_kwargs: handle)
    monkeypatch.setattr(
        noise, "finish_run", lambda _handle, **kwargs: summaries.append(kwargs))

    def fake_calibrate(**kwargs):
        received.update(kwargs)
        return {"status": "pass"}

    monkeypatch.setattr(noise, "calibrate", fake_calibrate)
    assert noise.main([
        "--submission", str(submission), "--public-train", str(public_train),
        "--space", str(SPACE),
    ]) == 0
    retained = run_dir / "artifacts_dir" / "prebuild_input_submission"
    assert received["submission"] == retained.resolve()
    assert cost_module._tree_sha256(retained) == cost_module._tree_sha256(submission)
    assert summaries[-1]["summary"]["prebuild_input_submission"] == str(retained.resolve())


@pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit])
def test_noise_main_finalizes_aet_without_converting_control_flow(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signal) -> None:
    cost_module = _module()
    noise = cost_module._load(f"noise_interrupt_{signal.__name__}", NOISE_SCRIPT)
    submission, public_train, _space = _inputs(tmp_path)
    run_dir = tmp_path / "noise-run"
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "artifacts_dir").mkdir()
    handle = SimpleNamespace(run_dir=run_dir)
    finishes = []
    monkeypatch.setattr(noise, "start_run", lambda **_kwargs: handle)
    monkeypatch.setattr(
        noise, "finish_run", lambda received, **kwargs: finishes.append((received, kwargs)))
    monkeypatch.setattr(noise, "calibrate", lambda **_kwargs: (_ for _ in ()).throw(signal()))
    with pytest.raises(signal):
        noise.main([
            "--submission", str(submission), "--public-train", str(public_train),
            "--space", str(SPACE),
        ])
    assert len(finishes) == 1
    assert finishes[0][1]["status"] == "error"
    assert not (run_dir / "metrics" / "k1_aa_noise_calibration.json").exists()


@pytest.mark.parametrize("tamper", [
    "prebuild_receipt", "prebuilt_package", "toolchain", "private_shape", "k1_capsule",
    "k1_evidence", "program_count",
])
def test_k1_authority_tampering_is_rejected(tmp_path: Path, tamper: str) -> None:
    _module_value, artifact, _grader, prebuilt, public_train, space = _produce(
        tmp_path, "k1-program")
    damaged = deepcopy(artifact)
    if tamper == "prebuild_receipt":
        damaged["prebuild_receipt"]["private_build_override"] = ["cmake"]
    elif tamper == "prebuilt_package":
        manifest = yaml.safe_load((prebuilt / "manifest.yaml").read_text())
        manifest["build"] = {"command": ["cmake"]}
        (prebuilt / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    elif tamper == "toolchain":
        damaged["toolchain_identity"]["k1_clang"]["sha256"] = "0" * 64
    elif tamper == "private_shape":
        shape = damaged["private_shape_calibration"]["records"][0]["private"]["shape"]
        shape[next(iter(shape))] += 1
    elif tamper == "k1_capsule":
        damaged["programs"][0]["capsule_id"] = "bogus"
    elif tamper == "program_count":
        damaged["trusted_evaluation_observations"][0]["k1_program_count"] += 2
    else:
        damaged["programs"][0]["evidence"].pop("receipt_nonce")
    assert _problems(damaged, "k1_calibration", public_train, space)


@pytest.mark.parametrize("tamper", ["duplicate", "family", "monotonic"])
def test_spike_authority_tampering_is_rejected(tmp_path: Path, tamper: str) -> None:
    _module_value, artifact, _grader, _prebuilt, public_train, space = _produce(
        tmp_path, "spike-screen")
    damaged = deepcopy(artifact)
    if tamper == "duplicate":
        damaged["observations"][1]["capsule_id"] = damaged["observations"][0]["capsule_id"]
    elif tamper == "family":
        damaged["observations"][0]["family"] = "bogus"
    else:
        damaged["end_monotonic_ns"] = damaged["start_monotonic_ns"]
    assert _problems(damaged, "spike_calibration", public_train, space)


@pytest.mark.parametrize("tamper", [
    "delta", "capsule", "stage", "side_duplicate", "mean", "median", "p95", "max",
])
def test_confirmation_authority_tampering_is_rejected(tmp_path: Path, tamper: str) -> None:
    _module_value, artifact, _grader, _prebuilt, public_train, space = _produce(
        tmp_path, "confirmation-overhead")
    damaged = deepcopy(artifact)
    row = damaged["stage_observations"]["package_build"][0]
    if tamper == "delta":
        row["wall_seconds"] *= 2
    elif tamper == "capsule":
        row["capsule_id"] = "bogus"
    elif tamper == "stage":
        row["stage"] = "spike_check"
    elif tamper == "side_duplicate":
        rows = damaged["stage_observations"]["package_build"]
        same_capsule = [item for item in rows if item["capsule_id"] == row["capsule_id"]]
        same_capsule[1]["side"] = same_capsule[0]["side"]
    else:
        damaged["package_build"][f"{tamper}_seconds"] += 0.001
    assert _problems(damaged, "confirmation_overhead_calibration", public_train, space)


@pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit])
def test_main_finalizes_aet_without_converting_control_flow_to_artifact(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signal) -> None:
    module = _module()
    submission, public_train, _space = _inputs(tmp_path)
    run_dir = tmp_path / "out" / "runs" / "interrupted"
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "artifacts_dir").mkdir()
    handle = SimpleNamespace(run_dir=run_dir)
    finishes = []
    monkeypatch.setattr(module, "start_run", lambda **_kwargs: handle)
    monkeypatch.setattr(
        module, "finish_run", lambda received, **kwargs: finishes.append((received, kwargs)))

    def interrupt(**_kwargs):
        raise signal()

    monkeypatch.setattr(module, "run_calibration", interrupt)
    with pytest.raises(signal):
        module.main([
            "--kind", "spike-screen", "--submission", str(submission),
            "--public-train", str(public_train), "--space", str(SPACE),
            "--noise-authority", str(tmp_path / "noise.json"),
        ])
    assert len(finishes) == 1
    assert finishes[0][1]["status"] == "error"
    assert not (run_dir / "metrics" / "spike_screen_calibration.json").exists()
