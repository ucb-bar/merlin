"""Descriptive original queued counts, separate from native timing predictions."""

import json
from pathlib import Path

from merlin_timing import observations as O

from merlin.common import invocation_record as I
from merlin.runtime import _native_files as F

from . import archive_native_screen as NATIVE
from .contracts import StageGateError, canonical_json, document_sha256

PROTOCOL = "merlin.public_native_observed_counts.v1"
SCHEMA = PROTOCOL
SCOPE = (
    "original selected counter-PC window; descriptive subset, overlapping categories; "
    "encoded requested movement is not physical memory traffic"
)
_BYTES = {
    "cpu_data.requested_read_bytes",
    "cpu_data.requested_write_bytes",
    "cpu_fetch.requested_read_bytes",
    "cpu_fetch.requested_write_bytes",
    "device.requested_read_bytes",
    "device.requested_write_bytes",
    "target.requested_read_bytes",
    "target.requested_write_bytes",
}


def unit(name):
    if name == "scope_instructions" or name.startswith("cpu."):
        return "instructions"
    if name.startswith("target.class."):
        return "events"
    if name == "target.commands":
        return "commands"
    if name == "target.fences":
        return "fences"
    if name in _BYTES:
        return "encoded requested bytes"
    if name in {"target.local_read_bytes", "target.local_write_bytes"}:
        return "encoded local operand bytes"
    return None


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate(value, *, receipt_sha256, runtime_sha256, numerical=None, capsule_sha256=None):
    """Validate the public transport; it never issues execution or timing authority."""
    if (
        type(value) is not dict
        or set(value) != {"schema", "status", "provenance", "metrics", "scope", "unknown", "authority"}
        or value["schema"] != SCHEMA
        or value["status"] != "observed"
        or value["scope"] != SCOPE
        or value["authority"] != "descriptive counts only; no performance ranking or promotion"
        or value["unknown"] != {
            "physical_memory_traffic": None, "overlap": None, "cycles": None, "attainment": None
        }
    ):
        raise StageGateError("public observed counts changed their descriptive-only contract")
    provenance = value["provenance"]
    fields = {
        "receipt_sha256", "report_sha256", "runtime_sha256", "numerical_receipt_sha256",
        "elf_sha256", "input_sha256", "capsule_sha256", "scope_binding",
    }
    if (
        type(provenance) is not dict or set(provenance) != fields
        or any(not _digest(provenance[k]) for k in fields - {"scope_binding"})
        or provenance["receipt_sha256"] != receipt_sha256
        or provenance["runtime_sha256"] != runtime_sha256
    ):
        raise StageGateError("public observed counts lost their original receipt/runtime joins")
    binding = provenance["scope_binding"]
    if (
        type(binding) is not dict
        or set(binding) != {
            "schema", "elf_sha256", "input_sha256", "scope_sha256", "begin_pc", "end_pc",
            "occurrence", "evidence_sha256",
        }
        or binding["schema"] != "merlin.spike_scope_binding.v1"
        or any(not _digest(binding[k]) for k in ("elf_sha256", "input_sha256", "scope_sha256", "evidence_sha256"))
        or binding["elf_sha256"] != provenance["elf_sha256"]
        or binding["input_sha256"] != provenance["input_sha256"]
        or any(type(binding[k]) is not int or binding[k] < 0 for k in ("begin_pc", "end_pc", "occurrence"))
        or binding["begin_pc"] >= binding["end_pc"]
    ):
        raise StageGateError("public observed counts lost their original counter-PC scope")
    if numerical is not None and (
        numerical.get("functional_status") != "PASS"
        or numerical.get("evidence_sha256") != provenance["numerical_receipt_sha256"]
        or numerical.get("compiled", {}).get("elf_sha256") != provenance["elf_sha256"]
    ):
        raise StageGateError("public observed counts differ from the original complete numerical arm")
    if capsule_sha256 is not None and provenance["capsule_sha256"] != capsule_sha256:
        raise StageGateError("public observed counts differ from the original public member")
    metrics = value["metrics"]
    if type(metrics) is not dict or not 1 <= len(metrics) <= 256 or "scope_instructions" not in metrics:
        raise StageGateError("public observed counts require a bounded original metric roster")
    for name, metric in metrics.items():
        if (
            type(name) is not str or unit(name) is None
            or type(metric) is not dict or set(metric) != {"value", "unit"}
            or metric["unit"] != unit(name)
            or type(metric["value"]) is not int or not 0 <= metric["value"] < 2**64
        ):
            raise StageGateError("public observed count metric has unknown units or modeled activity")
    return value


def project(owner, policy, member, workspace, numerical, receipt_path):
    """Reopen original private products; no saved numerical or count verdict is accepted."""
    workspace = F.canonical(workspace)
    receipt_path = F.canonical(receipt_path)
    if receipt_path != workspace / "public_native_counts_receipt.json":
        raise StageGateError("public observed count receipt escaped its original arm workspace")
    original, numeric_pin = NATIVE.original_numerical_observation(
        workspace, numerical,
        checked_source=(policy._checked_source_diagnostic, policy._compiler_pass_selection),
    )
    owner.selection.runtime.verify()
    receipt_pin = F.pin(receipt_path, max_bytes=16 << 20)
    receipt = json.loads(receipt_path.read_bytes())
    if (
        type(receipt) is not dict or type(receipt.get("report")) is not dict
        or set(receipt["report"]) != {"path", "sha256"}
        or type(receipt.get("processes")) is not list or len(receipt["processes"]) != 2
        or any(type(p) is not dict or set(p) != {"path", "sha256", "stdout_sha256"}
               or not _digest(p["sha256"]) or not _digest(p["stdout_sha256"])
               for p in receipt["processes"])
        or receipt["processes"][0]["stdout_sha256"] != receipt["processes"][1]["stdout_sha256"]
    ):
        raise StageGateError("public observed counts require their original complete count receipt")
    report_pin = F.pin(receipt["report"]["path"], max_bytes=16 << 20)
    if not Path(report_pin["path"]).is_relative_to(workspace / "public_native_counts"):
        raise StageGateError("public observed count report escaped its original count workspace")
    for path in (receipt_path, Path(report_pin["path"]), workspace / "ordinary_observation.json"):
        if path.stat().st_nlink != 1:
            raise StageGateError("public observed count products require original unaliased regular files")
    report = json.loads(F.canonical(receipt["report"]["path"]).read_bytes())
    model = json.loads(owner.selection.runtime.model_json)
    input_sha256 = document_sha256({"kind": "original_frozen_public_capsule", "sha256": member.source_sha256})
    if (
        receipt.get("schema") != "merlin.phase2.archive_count_observation.v1"
        or receipt.get("status") != "observed"
        or receipt.get("functional_identical") is not True
        or receipt.get("complete_counts") is not True
        or receipt.get("native_event_timing_qualified") is not False
        or receipt.get("runtime_sha256") != owner.selection.runtime.sha256
        or receipt.get("elf_sha256") != original.get("compiled", {}).get("elf_sha256")
        or receipt.get("input_sha256") != input_sha256
        or report_pin["sha256"] != receipt["report"]["sha256"]
        or report.get("schema") != O.REPORT_V2
        or not O.feature_complete(report)
        or report.get("scope_binding") != receipt.get("scope_binding")
        or report.get("elf_sha256") != receipt["elf_sha256"]
        or report.get("input_sha256") != receipt["input_sha256"]
        or report.get("model_sha256") != F.digest(owner.selection.runtime.model_path)
        or report.get("feature_engine_sha256") != model["domain"]["feature_engine_sha256"]
        or report.get("device_sha256") != model["domain"]["device_sha256"]
        or report.get("scope_sha256") != model["domain"]["scope_sha256"]
        or receipt.get("scope_binding", {}).get("scope_sha256") != model["domain"]["scope_sha256"]
        or report.get("feature_counts") != receipt.get("feature_counts")
        or report.get("loaded_library_sha256") != model["libraries"]
        or report["feature_observation"]["scope_kind"] != "counter_pc_pair"
    ):
        raise StageGateError("public observed counts lost their original numerical/report/runtime/window joins")
    for arm, process in zip(("functional", "annotated"), receipt["processes"], strict=True):
        path = F.canonical(process["path"])
        if not path.is_relative_to(workspace / "public_native_counts" / arm) or F.digest(path) != process["sha256"]:
            raise StageGateError("public observed counts lost their original process custody")
        actual = I.verify(path)
        scopes = [p for p in actual["inputs"] if Path(p["path"]).name == "scope.json"]
        if (
            actual.get("kind") != "subprocess" or actual.get("status") != "completed"
            or actual.get("returncode") != 0 or actual.get("stage") != "archive_screen_count"
            or actual.get("stdout", {}).get("sha256") != process["stdout_sha256"]
            or not any(p["sha256"] == receipt["elf_sha256"] for p in actual["inputs"])
            or not all({"path": p, "sha256": d} in actual["dependencies"] for p, d in owner.selection.runtime.files)
            or len(scopes) != 1 or json.loads(Path(scopes[0]["path"]).read_bytes()) != receipt["scope_binding"]
            or arm == "annotated" and not any(p["sha256"] == report_pin["sha256"] for p in actual["outputs"])
        ):
            raise StageGateError("public observed counts differ from their original complete process")
    metrics = {
        name: {"value": count, "unit": unit(name)}
        for name, count in report["feature_counts"].items() if unit(name) is not None
    }
    metrics["scope_instructions"] = {
        "value": report["feature_observation"]["scope_instructions"], "unit": "instructions"
    }
    result = {
        "schema": SCHEMA, "status": "observed", "scope": SCOPE,
        "provenance": {
            "receipt_sha256": receipt_pin["sha256"], "report_sha256": report_pin["sha256"],
            "runtime_sha256": receipt["runtime_sha256"], "numerical_receipt_sha256": numeric_pin["sha256"],
            "elf_sha256": receipt["elf_sha256"], "input_sha256": receipt["input_sha256"],
            "capsule_sha256": member.source_sha256, "scope_binding": receipt["scope_binding"],
        },
        "metrics": metrics,
        "unknown": {"physical_memory_traffic": None, "overlap": None, "cycles": None, "attainment": None},
        "authority": "descriptive counts only; no performance ranking or promotion",
    }
    validate(result, receipt_sha256=receipt_pin["sha256"], runtime_sha256=owner.selection.runtime.sha256,
             numerical=numerical, capsule_sha256=member.source_sha256)
    owner.selection.runtime.verify()
    if (
        F.pin(receipt_path, max_bytes=16 << 20) != receipt_pin
        or F.pin(receipt["report"]["path"], max_bytes=16 << 20) != report_pin
        or F.pin(workspace / "ordinary_observation.json", max_bytes=64 << 20) != numeric_pin
    ):
        raise StageGateError("public observed count/numerical products changed during projection")
    return result


def recheck_child_result(owner, *, policy, document, work_root, selected_members, deadline):
    """Rebuild from the parent's trusted job paths before completed feedback."""
    from merlin.common.cpu_pool import shared_compute_pool

    from . import ordinary_checked_source as CHECKED
    from .public_native_exploration import OBSERVED_COUNT_FEEDBACK_PROTOCOL, PublicNativeExploration

    if type(owner) is not PublicNativeExploration or owner.count_feedback_protocol != OBSERVED_COUNT_FEEDBACK_PROTOCOL:
        raise StageGateError("public observed count parent check requires its exact original explicit owner")
    limit = CHECKED._deadline(deadline)
    root = F.canonical(work_root)
    if not root.is_relative_to(F.canonical(policy.feedback_evaluator.work_root)):
        raise StageGateError("public observed count job escaped its original private owner")
    members = sorted(policy.search_corpus.capsules, key=lambda row: (row.family, row.capsule))
    if (
        type(document) is not dict or document.get("selected_members") != [list(k) for k in selected_members]
        or type(document.get("cells")) is not list or len(document["cells"]) != len(members)
    ):
        raise StageGateError("public observed count feedback changed its original admitted roster")
    with shared_compute_pool(timeout_s=CHECKED._queue_wait(limit)):
        owner.verify_for(policy)
        for index, (member, row) in enumerate(zip(members, document["cells"], strict=True)):
            if type(row) is not dict or (row.get("family"), row.get("capsule"), row.get("source_sha256")) != (
                member.family, member.capsule, member.source_sha256
            ):
                raise StageGateError("public observed count feedback changed its original member order")
            if (member.family, member.capsule) not in selected_members:
                continue
            for arm in ("baseline", "candidate"):
                CHECKED._completed(limit)
                value = row.get(arm)
                if type(value) is not dict or type(value.get("native_exploration")) is not dict:
                    raise StageGateError("public observed count feedback lost an original arm")
                projection = value["native_exploration"]
                workspace = F.canonical(root / f"m{index:03d}_{arm}")
                receipt_path = workspace / "public_native_counts_receipt.json"
                if projection.get("observed_counts") is None:
                    if projection.get("receipt_sha256") is not None or (
                        receipt_path.exists() and json.loads(receipt_path.read_bytes()).get("status") == "observed"
                    ):
                        raise StageGateError("public observed count feedback omitted its original observed products")
                    continue
                numerical = {k: v for k, v in value.items() if k != "native_exploration"}
                expected = project(owner, policy, member, workspace, numerical, receipt_path)
                if owner.exploratory_estimate_protocol is not None:
                    from . import _public_native_exploratory_estimate as ESTIMATE

                    report_path = json.loads(receipt_path.read_bytes())["report"]["path"]
                    estimate = ESTIMATE.project(owner, expected, report_path)
                    if canonical_json(estimate) != canonical_json(projection.get("exploratory_estimate")):
                        raise StageGateError("exploratory estimate differs from its original fixed-model report")
                if canonical_json(expected) != canonical_json(projection["observed_counts"]):
                    raise StageGateError("public observed count feedback differs from its fresh original parent check")
        owner.verify_for(policy)
    CHECKED._completed(limit)
