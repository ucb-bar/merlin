"""Whole-model workflow: scope-bound analyses and distinct global receipt authority."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .broker import BrokerAction
from .broker_evidence import _is_sha256, workflow_binding
from .broker_policy import (
    ACTION_PROVIDER_REQUIREMENTS,
    ANALYSIS_ACTION,
    CHANGED_REGION_ACTION,
    CONTROLLED_CONTEXT_ACTION,
    DEVELOPMENT_FEEDBACK_ACTION,
    E2E_ANALYSIS_ACTION,
    INVENTORY_ACTION,
    OCCUPANCY_PROFILE_ACTION,
    PAIRED_CONTEXT_ACTION,
    SOURCE_CONTRACTION_PREPARATION_ACTION,
    SOURCE_CONTRACTION_QUALIFICATION_ACTION,
    SOURCE_CONVOLUTION_PREPARATION_ACTION,
    WHOLE_MODEL_V1,
    WorkflowPolicy,
    _build_action_registry,
    _record_host_refusal,
    agent_visible_refusal,
    unavailable_global_actions,
)
from .contracts import StageGateError
from .contracts import canonical_json as _canonical_json
from .contracts import sha256_file as _sha256_file
from .contracts import write_json as _write_json


def verify_global_broker_receipts(
    path: Path, *, actions: Sequence[BrokerAction], audit: Mapping[str, Any]
) -> dict[str, Any]:
    """Join macro agent tool calls without importing the legacy mandatory micro timing rule."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("global broker receipts are absent or linked")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    # Requests receive indices at admission, but concurrent requests finish out of order.
    rows.sort(key=lambda row: row.get("index", -1))
    registry = {action.name: action for action in actions}
    for index, row in enumerate(rows):
        if (
            row.get("index") != index
            or row.get("receipt_schema_version") != 1
            or row.get("action") not in registry
            or row.get("state") not in ("complete", "rejected")
            or any(
                not _is_sha256(row.get(key))
                for key in ("bindings_command_sha256", "stdout_sha256", "stderr_sha256", "argv_sha256")
            )
        ):
            raise ValueError("global broker receipt schema is invalid")
    binding = workflow_binding(rows, WHOLE_MODEL_V1)
    observed = [(row["action"], row["bindings_command_sha256"]) for row in rows]
    invocations = audit.get("broker_invocations")
    if not isinstance(invocations, list) or observed != [
        (row.get("action"), row.get("bindings_sha256")) for row in invocations
    ]:
        raise ValueError("global broker receipts do not match exact agent invocations")
    required = {action.name for action in actions if action.required}
    succeeded = {row["action"] for row in rows if row.get("state") == "complete" and row.get("returncode") == 0}
    if required - succeeded:
        raise ValueError("global broker required actions did not complete: " + str(sorted(required - succeeded)))
    if DEVELOPMENT_FEEDBACK_ACTION in succeeded:
        raise ValueError("a legacy micro sweep cannot be macro experiment evidence")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "count": len(rows),
        "required_actions": sorted(required),
        "successful_actions": sorted(succeeded),
        "all_required_succeeded": True,
        "workflow_policy": binding,
    }


class WholeModelPolicy(WorkflowPolicy):
    @property
    def workflow_id(self) -> str:
        return WHOLE_MODEL_V1

    @property
    def unavailable(self):
        return unavailable_global_actions(
            {attribute: getattr(self, attribute) for _action, attribute, _reason in ACTION_PROVIDER_REQUIREMENTS}
        )

    def build_registry(self):
        return _build_action_registry(
            self.candidate, self.target_experiment, global_optimization=True, unavailable=self.unavailable
        )

    verify_receipts = staticmethod(verify_global_broker_receipts)

    def execute(self, request, action_name, rendered, call_index, timeout_s, started):
        if action_name != E2E_ANALYSIS_ACTION:
            try:
                self.global_experiment.revision_session.validate_candidate_scope(self.candidate)
            except ValueError as exc:
                # Scope/input-integrity checks use ValueError for a denied submission. Normalize
                # only that check's expected refusal to the broker gate protocol so execute()
                # closes its allocated receipt before HTTP reports the denial. Unexpected faults
                # still propagate; none of these paths may reach candidate execution.
                _record_host_refusal(self, exc, round_index=self.feedback_round, call_index=call_index)
                raise StageGateError("compiler edit authority or input integrity refused candidate execution") from exc
        handlers = {
            E2E_ANALYSIS_ACTION: self._analyze,
            INVENTORY_ACTION: self._inventory,
            OCCUPANCY_PROFILE_ACTION: self._profile,
            ANALYSIS_ACTION: self._command_buffers,
            DEVELOPMENT_FEEDBACK_ACTION: self._feedback,
            PAIRED_CONTEXT_ACTION: self._paired_context,
            CONTROLLED_CONTEXT_ACTION: self._controlled_context,
            SOURCE_CONTRACTION_PREPARATION_ACTION: self._contraction,
            SOURCE_CONTRACTION_QUALIFICATION_ACTION: self._contraction,
            SOURCE_CONVOLUTION_PREPARATION_ACTION: self._convolution,
            CHANGED_REGION_ACTION: self._changed_region,
        }
        handler = handlers.get(action_name)
        if handler is None:
            return None
        return handler(request, action_name, rendered, call_index, timeout_s, started)

    def _analyze_document(self, request, rendered, call_index, timeout_s):
        document = self.global_experiment.analysis.analyze(
            self.candidate,
            hypothesis=str(request.get("hypothesis") or "Agent-requested complete-model compiler revision"),
            timeout_s=timeout_s,
        )
        if self.services.global_analysis_view is None:
            raise StageGateError("global analysis projection service is unavailable")

        detail = self.receipt_path.parent / f"full_model_analysis_{call_index:04d}.json"
        _write_json(detail, document)
        detail.chmod(0o444)
        document = self.services.global_analysis_view(
            document,
            complete_evidence=f"/perf-control/{detail.name}",
            context_provider_installed=self.global_context_provider is not None,
        )
        return document

    def _profile_document(self, request, rendered, call_index, timeout_s):
        if self.global_probe_provider is None:
            raise StageGateError("host mechanism-equivalent probe extraction is unavailable")
        prepared = self.global_probe_provider(
            candidate=self.candidate, probes=self.global_experiment.probes, timeout_s=timeout_s
        )
        document = self.global_experiment.probes.measure_probe(
            self.candidate,
            admission_inputs=prepared["admission_inputs"],
            execute=prepared["execute"],
            timeout_s=timeout_s,
        )
        return document

    def _inventory_document(self, request, rendered, call_index, timeout_s):
        return self.global_experiment.revision_session.inspect_optimization_surfaces(self.candidate)

    def _paired_context(self, request, action_name, rendered, call_index, timeout_s, started):
        feedback_document = None
        try:
            if self.global_experiment is None or self.global_paired_context_provider is None:
                raise StageGateError("paired fixed-work context extraction is unavailable")
            document = self.global_experiment.probes.compare_controlled_context(
                self.candidate, provider=self.global_paired_context_provider, timeout_s=timeout_s
            )
            result = {
                "returncode": 0,
                "stdout": _canonical_json(document).decode("utf-8"),
                "stderr": "",
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        except Exception as exc:
            _record_host_refusal(self, exc, round_index=self.feedback_round, call_index=call_index)
            result = {
                "returncode": 125,
                "stdout": "",
                "stderr": agent_visible_refusal("paired context comparison refused", exc),
                "elapsed_s": round(time.monotonic() - started, 3),
            }

        return result, feedback_document

    def _controlled_context(self, request, action_name, rendered, call_index, timeout_s, started):
        feedback_document = None
        try:
            if self.global_experiment is None or self.global_context_provider is None:
                raise StageGateError("controlled source-prefix extraction is unavailable")
            document = self.global_experiment.probes.profile_controlled_context(
                self.candidate, provider=self.global_context_provider, timeout_s=timeout_s
            )
            result = {
                "returncode": 0,
                "stdout": _canonical_json(document).decode("utf-8"),
                "stderr": "",
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        except Exception as exc:
            _record_host_refusal(self, exc, round_index=self.feedback_round, call_index=call_index)
            result = {
                "returncode": 125,
                "stdout": "",
                "stderr": agent_visible_refusal("controlled source-prefix profile refused", exc),
                "elapsed_s": round(time.monotonic() - started, 3),
            }

        return result, feedback_document

    def _contraction(self, request, action_name, rendered, call_index, timeout_s, started):
        feedback_document = None
        try:
            if self.global_experiment is None:
                raise StageGateError("source contraction actions require a global experiment")
            if action_name == SOURCE_CONTRACTION_PREPARATION_ACTION:
                values = {}
                for key in ("source_op_index", "max_m", "max_n", "max_k"):
                    value = rendered[key]
                    if not isinstance(value, str) or not value.isascii() or not value.isdecimal() or len(value) > 10:
                        raise StageGateError("source contraction indices and bounds must be decimal integers")
                    values[key] = int(value)
                document = self.global_experiment.probes.prepare_source_contraction(
                    self.candidate, comparison_arm=rendered["comparison_arm"], **values, timeout_s=timeout_s
                )
            else:
                if self.global_source_pair_provider is None:
                    raise StageGateError("complete source-pair runtime provider is unavailable")
                document = self.global_experiment.probes.qualify_source_contraction(
                    self.candidate,
                    preparation_sha256=rendered["preparation_sha256"],
                    provider=self.global_source_pair_provider,
                    timeout_s=timeout_s,
                )
            result = {
                "returncode": 0,
                "stdout": _canonical_json(document).decode("utf-8"),
                "stderr": "",
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        except Exception as exc:
            _record_host_refusal(self, exc, round_index=self.feedback_round, call_index=call_index)
            result = {
                "returncode": 125,
                "stdout": "",
                "stderr": agent_visible_refusal("source contraction action refused", exc),
                "elapsed_s": round(time.monotonic() - started, 3),
            }

        return result, feedback_document

    def _convolution(self, request, action_name, rendered, call_index, timeout_s, started):
        feedback_document = None
        try:
            if self.global_experiment is None:
                raise StageGateError("source-convolution preparation requires a global experiment")
            document = self.global_experiment.probes.prepare_source_convolution(
                self.candidate, comparison_arm=rendered["comparison_arm"], timeout_s=timeout_s
            )
            result = {
                "returncode": 0,
                "stdout": _canonical_json(document).decode("utf-8"),
                "stderr": "",
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        except Exception as exc:
            _record_host_refusal(self, exc, round_index=self.feedback_round, call_index=call_index)
            result = {
                "returncode": 125,
                "stdout": "",
                "stderr": agent_visible_refusal("source-convolution preparation refused", exc),
                "elapsed_s": round(time.monotonic() - started, 3),
            }

        return result, feedback_document

    def _changed_region(self, request, action_name, rendered, call_index, timeout_s, started):
        feedback_document = None
        try:
            if self.global_experiment is None or self.global_semantic_provider is None:
                raise StageGateError("host changed-region semantic extraction is unavailable")
            document = self.global_experiment.probes.qualify_changed_region(
                self.candidate, provider=self.global_semantic_provider, timeout_s=timeout_s
            )
            result = {
                "returncode": 0,
                "stdout": _canonical_json(document).decode("utf-8"),
                "stderr": "",
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        except Exception as exc:
            _record_host_refusal(self, exc, round_index=self.feedback_round, call_index=call_index)
            result = {
                "returncode": 125,
                "stdout": "",
                "stderr": agent_visible_refusal("changed-region semantic qualification refused", exc),
                "elapsed_s": round(time.monotonic() - started, 3),
            }

        return result, feedback_document

    def _feedback(self, request, action_name, rendered, call_index, timeout_s, started):
        feedback_document = None
        try:
            if self.global_experiment is not None:
                raise StageGateError(
                    "a corpus microbenchmark sweep is not a global iteration; use full-model "
                    "analysis and an admitted separate mechanism probe"
                )
        except Exception as exc:  # noqa: BLE001 - receipt the refusal; never expose raw evaluator data
            # THE AGENT MAY NOT SEE WHY, BUT THE HOST MUST RECORD IT. What the agent gets stays
            # the exception's type and nothing else, because the refusal reason names gate
            # internals. What was written nowhere was the reason itself: the receipt keeps only
            # a DIGEST of that one-line string, so a refusal after a full sweep left the type
            # and no message, no traceback and no cell. Measured 2026-09-04: an eighty-minute
            # measurement in which all 38 members ran was thrown away, and recovering the cause
            # meant brute-forcing the exception name against the receipt's stderr digest.
            _record_host_refusal(self, exc, round_index=self.feedback_round, call_index=call_index)
            result = {
                "returncode": 125,
                "stdout": "",
                "stderr": agent_visible_refusal("development GSIM feedback refused by the host-owned evaluator", exc),
                "elapsed_s": round(time.monotonic() - started, 3),
            }

        return result, feedback_document
