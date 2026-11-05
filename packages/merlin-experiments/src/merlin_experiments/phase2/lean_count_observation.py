"""Experimental fixed-model counts without correspondence or archive admission.

Reuse the original boxed observer and full native numerical result. This adapter
does not construct source, calibration, hardware or promotion owners.
"""

import json
import math
from pathlib import Path

from merlin.runtime import _native_files as F
from merlin.targetgen import native_model_execution as NATIVE
from merlin.targetgen.target_experiment import TargetExperiment, load_target_experiment

from . import archive_count_observation as OBS
from . import campaign as PC
from .archive_spike_runtime import COUNT_OBSERVATION_900S_PROTOCOL, freeze_runtime
from .contracts import write_json

MODEL_SHA256 = "dc413b315b21b0bb2874098c0421db5900d541aef99cf20f76fa8eaf02f3fc0d"
SCHEMA = "merlin.phase2.lean_count_observation.v1"


def prepare_runtime(spec_path, *, expected_model_sha256=MODEL_SHA256):
    """Freeze existing runtime bytes directly; no qualification replay or refit."""
    if expected_model_sha256 != MODEL_SHA256:
        raise ValueError("lean exploration retains the exact frozen dc413 timing model")
    spec_path = F.canonical(Path(spec_path))
    spec = json.loads(spec_path.read_bytes())
    spec = spec.get("runtime", spec)
    model = F.canonical(Path(spec["model"]))
    if F.digest(model) != MODEL_SHA256:
        raise ValueError("lean exploration model differs from frozen dc413 coefficients")
    result = freeze_runtime(
        review_id="experimental whole-native counts; fixed dc413; no correspondence qualification",
        functional_argv=tuple("--instructions=1000000000" if item.startswith("--instructions=") else item
                              for item in spec["functional_argv"]),
        observer=Path(spec["observer"]), disassembler=Path(spec["disassembler"]),
        model_path=model, source_manifest=Path(spec["source_manifest"]),
        libraries=tuple(map(Path, spec["libraries"])), dependencies=tuple(map(Path, spec["dependencies"])),
        entry_symbol=spec["entry_symbol"], completion_symbol=spec["completion_symbol"],
        timing_selection=spec["timing_selection"], observation_resource_protocol=COUNT_OBSERVATION_900S_PROTOCOL,
    )
    if F.digest(model) != MODEL_SHA256:
        raise ValueError("lean exploration model changed during runtime selection")
    return result


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def observe_evaluation(evaluation, *, runtime, target_experiment, package, directory,
                       floor_cycles=None, target_fraction=0.3, remaining_s):
    """Count the exact ELF only after its existing complete native output pass."""
    if not _positive(remaining_s) or not _positive(target_fraction) or target_fraction > 1:
        raise ValueError("lean count observation requires positive bounded time and target fraction")
    if floor_cycles is not None and not _positive(floor_cycles):
        raise ValueError("lean count structural floor must be positive or explicitly unknown")
    directory = F.canonical(Path(directory), exists=False)
    directory.mkdir(parents=True, exist_ok=False)
    result = dict(schema=SCHEMA, status="UNKNOWN", qualification="UNQUALIFIED", source_effects="PENDING",
                  formal_admission=False, promotion_authority="NONE", roofline_convergence=False,
                  member=evaluation.get("member", evaluation.get("id")), model_sha256=MODEL_SHA256,
                  structural_floor_cycles=floor_cycles, target_fraction=target_fraction,
                  estimated_cycles=None, estimated_attainment=None, attainment_fraction=None,
                  feature_complete=False, timing_complete=False, complete_counts=False, native_outputs_complete=False,
                  scope_sha256=None, native_event_timing_qualified=False, hardware_execution_qualified=False,
                  estimate_below_structural_bound=False, comparison_consistent=None,
                  uncertainty_interval_cycles=None,
                  scope="experimental fixed-model counts versus structural floor; no measured hardware latency")
    try:
        native_path = NATIVE._pinned_native_file(evaluation, "native_result")
        native = json.loads(native_path.read_bytes())
        numeric = native.get("numeric") or {}
        outputs = numeric.get("per_output")
        if (native.get("schema") != "merlin_candidate_native_model_execution_v1"
                or numeric.get("status") != "pass" or numeric.get("policy") != "exact_int"
                or numeric.get("mismatch_count") != 0 or not isinstance(outputs, dict) or not outputs
                or any(row.get("status") != "pass" or row.get("mismatch_count") != 0
                       or type(row.get("n_elements")) is not int or row["n_elements"] <= 0
                       for row in outputs.values())):
            raise ValueError("lean counts require the original complete exact native output pass")
        elf = NATIVE._pinned_native_file(native, "elf")
        workspace = F.canonical(Path(evaluation["work_root"]))
        if not elf.is_relative_to(workspace) or not directory.is_relative_to(workspace):
            raise ValueError("lean count ELF and output must remain inside the native member workspace")
        if F.digest(runtime.model_path) != MODEL_SHA256:
            raise ValueError("lean count runtime changed its frozen model")
        target = (target_experiment if isinstance(target_experiment, TargetExperiment)
                  else load_target_experiment(Path(target_experiment)))
        policy = PC.package_sandbox_policy(target, workspace, Path(package))
        observed = OBS.observe(elf, runtime=runtime, policy=policy, directory=directory/"observation",
            input_sha256=native["source"]["inputs"]["sha256"], remaining_s=remaining_s, numerical_pass=True)
        result.update(native_result=NATIVE._digest(native_path), elf=NATIVE._digest(elf),
                      input_sha256=native["source"]["inputs"]["sha256"],
                      source_interface_sha256=native["source"]["interface"]["sha256"],
                      runtime=runtime.record(), runtime_sha256=runtime.sha256, count_observation=observed,
                      native_outputs_complete=True)
        if observed.get("status") != "observed":
            result["reason"] = observed.get("reason", "original count observation unavailable")
        else:
            report_path = F.canonical(Path(observed["report"]["path"]))
            if F.digest(report_path) != observed["report"]["sha256"]:
                raise ValueError("lean count report changed before projection")
            report = json.loads(report_path.read_bytes())
            windows = report.get("counter_windows")
            cycles = report.get("estimated_cycles")
            if (report.get("model_sha256") != MODEL_SHA256 or observed.get("elf_sha256") != native["elf"]["sha256"]
                    or not isinstance(windows, list) or len(windows) != 1
                    or windows[0].get("estimated_cycles") != cycles
                    or any(windows[0].get(key) != observed["scope_binding"].get(key)
                           for key in ("begin_pc", "end_pc"))):
                raise ValueError("lean count report lost its exact fixed-model/ELF/call scope")
            result.update(status="observed", report=NATIVE._digest(report_path),
                          estimated_cycles=cycles if _positive(cycles) else None,
                          feature_complete=observed.get("complete_counts") is True,
                          complete_counts=observed.get("complete_counts") is True,
                          timing_complete=report.get("timing_complete") is True,
                          unresolved_events=report.get("unresolved", []), scope_binding=observed["scope_binding"],
                          scope_sha256=report.get("scope_sha256"))
            if _positive(cycles) and floor_cycles is not None:
                below = cycles < floor_cycles
                result.update(estimate_below_structural_bound=below, comparison_consistent=not below)
                if below:
                    result["reason"] = "estimate_below_structural_bound"
            if (_positive(cycles) and floor_cycles is not None and result["feature_complete"]
                    and result["timing_complete"] and result["unresolved_events"] == []
                    and result["comparison_consistent"] is True):
                result.update(estimated_attainment=floor_cycles/cycles, attainment_fraction=floor_cycles/cycles,
                              target_cycle_budget=floor_cycles/target_fraction)
        if NATIVE._digest(native_path) != evaluation["native_result"] or F.digest(runtime.model_path) != MODEL_SHA256:
            raise ValueError("lean observation changed original native/model bytes")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        result.update(status="UNKNOWN", reason=f"{type(error).__name__}: {error}",
                      estimated_cycles=None, estimated_attainment=None, attainment_fraction=None,
                      feature_complete=False, timing_complete=False, complete_counts=False)
    write_json(directory/"result.json", result)
    return result
