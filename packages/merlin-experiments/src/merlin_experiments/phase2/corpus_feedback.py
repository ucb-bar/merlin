"""Corpus-feedback workflow: certified GSIM deltas and final-byte evidence."""

from __future__ import annotations

import copy
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.common.digest import sha256_bytes as _sha256
from merlin.perf.agent_guidance import inspect_compiler_package

from .broker import BrokerAction
from .broker_evidence import _is_sha256, workflow_binding
from .broker_policy import (
    ANALYSIS_ACTION,
    CORPUS_FEEDBACK_V1,
    DEVELOPMENT_FEEDBACK_ACTION,
    E2E_ANALYSIS_ACTION,
    INVENTORY_ACTION,
    OCCUPANCY_PROFILE_ACTION,
    WorkflowPolicy,
    _build_action_registry,
    _record_host_refusal,
    agent_visible_refusal,
)
from .contracts import StageGateError
from .contracts import canonical_json as _canonical_json
from .contracts import sha256_file as _sha256_file


def validate_redacted_feedback(document: Mapping[str, Any]) -> dict[str, Any]:
    """Exact non-answer schema returned to the authoring agent."""
    required = {
        "schema_version",
        "kind",
        "round",
        "invocation",
        "tuning_corpus_sha256",
        "candidate_sha256",
        "certificate_sha256",
        "engine",
        "cells",
        "summary",
        "stopping",
    }
    if not isinstance(document, Mapping) or set(document) != required:
        raise StageGateError("development GSIM feedback violates its redacted top-level schema")
    if (
        document.get("schema_version") != 1
        or document.get("kind") != "host_owned_tuning_gsim_feedback"
        or document.get("engine") != "gsim"
    ):
        raise StageGateError("development feedback is not the host-owned GSIM schema")
    for attribute in ("round", "invocation"):
        value = document.get(attribute)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise StageGateError(f"development feedback {attribute} is invalid")
    for attribute in ("tuning_corpus_sha256", "candidate_sha256", "certificate_sha256"):
        if not _is_sha256(document.get(attribute)):
            raise StageGateError(f"development feedback {attribute} is not a SHA-256")
    cells = document.get("cells")
    if not isinstance(cells, list) or not cells:
        raise StageGateError("development feedback contains zero tuning cells")
    cell_fields = {
        "family",
        "capsule",
        "baseline_correct",
        "candidate_correct",
        "baseline_gsim_cycles",
        "candidate_gsim_cycles",
        "candidate_minus_baseline_cycles",
        "baseline_over_candidate",
        "comparable",
        "declared_macs",
        "declared_work_basis",
        "ideal_cycles_at_peak",
        "baseline_utilization",
        "candidate_utilization",
        "baseline_share_of_achievable",
        "candidate_share_of_achievable",
        # The ceiling used for THIS member and the geometry-matching rule that selected
        # it.  The summary's global maximum is context only; these fields are the
        # denominator used by the ratios and stopping rule.
        "achievable_macs_per_cycle",
        "achievable_basis",
        # Whether this member is FINISHED, better, or still owes cycles. Everything
        # above is a number the reader has to interpret; without this the cell records
        # a measurement and states no position on it, which is how a member at 3% of
        # the achievable rate and one at 100% came to read identically.
        "verdict",
        "verdict_reason",
        # HOW MUCH is left, beside whether any is. `factor_to_achievable` is the multiple
        # this member is off the rate something on this machine demonstrably reached, so a
        # reader can rank where to spend effort instead of treating every open member alike.
        "factor_to_achievable",
        "ideal_cycles_at_achievable",
        "cycles_saved",
        "gap_closed",
        # A cell the sweep did not pay for says so, rather than being omitted. Omitting
        # it would let a short sweep read as a complete one.
        "measured",
        "skip_reason",
    }
    identities: set[tuple[str, str]] = set()
    for index, row in enumerate(cells):
        if not isinstance(row, Mapping) or set(row) != cell_fields:
            raise StageGateError(f"development feedback cell {index} violates the redacted schema")
        family, capsule = row.get("family"), row.get("capsule")
        if (
            not isinstance(family, str)
            or not family
            or not isinstance(capsule, str)
            or not capsule
            or (family, capsule) in identities
        ):
            raise StageGateError(f"development feedback cell {index} has an invalid identity")
        identities.add((family, capsule))
        # AN UNMEASURED CELL HAS NULLS WHERE A MEASURED ONE HAS NUMBERS, and demanding numbers from
        # it is not strictness -- it discards the whole sweep. Measured 2026-09-04: a sweep that
        # walks members cheapest-first and stops once a candidate is already losing emits a cell for
        # each member it did not pay for; every one carried null correctness and null cycles, this
        # check rejected the first of them, and an eighty-minute measurement in which all 38 members
        # ran was thrown away with nothing recorded but an exception type. An absent number must
        # read as absent. What is still demanded is that the cell say so: `measured` False with a
        # reason, `comparable` False, and no derived delta -- so a short sweep can never be read as
        # a complete one, which is the failure this null was introduced to prevent.
        if not isinstance(row.get("measured"), bool):
            raise StageGateError(f"development feedback cell {index} does not say whether it ran")
        measured = bool(row["measured"])
        if not measured:
            if row.get("comparable") is not False or not str(row.get("skip_reason") or ""):
                raise StageGateError(
                    f"development feedback cell {index} was not measured but claims a comparison or gives no reason"
                )
            for field in (
                "baseline_correct",
                "candidate_correct",
                "baseline_gsim_cycles",
                "candidate_gsim_cycles",
                "candidate_minus_baseline_cycles",
                "baseline_over_candidate",
            ):
                if row.get(field) is not None:
                    raise StageGateError(f"development feedback cell {index} was not measured but carries {field}")
            continue
        matched_rate = row.get("achievable_macs_per_cycle")
        if matched_rate is not None and (
            isinstance(matched_rate, bool) or not isinstance(matched_rate, (int, float)) or matched_rate <= 0
        ):
            raise StageGateError(f"development feedback cell {index} has an invalid member-matched achievable rate")
        if not isinstance(row.get("achievable_basis"), str) or not row["achievable_basis"]:
            raise StageGateError(f"development feedback cell {index} omits its member-matched achievable basis")
        # Utilization is derived or it is null. A ratio outside (0, 1] would mean the program beat a
        # ceiling its own RTL says is unreachable, which is a broken derivation, not a fast program.
        macs = row.get("declared_macs")
        if macs is not None and (isinstance(macs, bool) or not isinstance(macs, int) or macs <= 0):
            raise StageGateError(f"development feedback cell {index} has invalid declared work")
        if not isinstance(row.get("declared_work_basis"), str):
            raise StageGateError(f"development feedback cell {index} omits its work basis")
        for field in (
            "ideal_cycles_at_peak",
            "baseline_utilization",
            "candidate_utilization",
            "baseline_share_of_achievable",
            "candidate_share_of_achievable",
        ):
            value = row.get(field)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise StageGateError(f"development feedback cell {index} has an invalid {field}")
            # TWO RATIOS, TWO DIFFERENT CEILINGS, AND ONLY ONE OF THEM IS A BOUND.
            #
            # `utilization` is taken against the STRUCTURAL peak, which the target's own RTL derives
            # from its array geometry. Nothing can exceed it, so a value above 1 is a broken
            # derivation and is still refused.
            #
            # `share_of_achievable` is taken against the best rate any MEASURED program has been
            # observed to reach. That is an empirical best-so-far, not a bound, and a better program
            # is exactly the thing that beats it. Refusing it here discarded an entire 38-member
            # sweep: measured 2026-09-04, the ceiling harvested from the functional run was 80.01
            # MACs/cycle while four members of the PERF corpus already ran above it -- PC01_k128 at
            # 99.79 (share 1.247), PR01 at 94.06, PR03 at 88.33, PR02 at 87.96 -- because the perf
            # corpus carries larger, more efficient shapes than the corpus the ceiling came from.
            # An eighty-minute measurement in which every member ran and passed was thrown away for
            # reporting the good news that the ceiling was too low.
            if field.endswith("_utilization") and value > 1:
                raise StageGateError(
                    f"development feedback cell {index} reports {field} above the derived "
                    f"structural peak, which no program can exceed"
                )
        if any(
            not isinstance(row.get(field), bool) for field in ("baseline_correct", "candidate_correct", "comparable")
        ):
            raise StageGateError(f"development feedback cell {index} has invalid correctness")
        if row["comparable"] != (row["baseline_correct"] and row["candidate_correct"]):
            raise StageGateError(f"development feedback cell {index} has inconsistent comparability")
        for field in ("baseline_gsim_cycles", "candidate_gsim_cycles"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise StageGateError(f"development feedback cell {index} has invalid GSIM cycles")
        delta, ratio = row.get("candidate_minus_baseline_cycles"), row.get("baseline_over_candidate")
        if row["comparable"]:
            expected_delta = row["candidate_gsim_cycles"] - row["baseline_gsim_cycles"]
            expected_ratio = row["baseline_gsim_cycles"] / row["candidate_gsim_cycles"]
            if (
                isinstance(delta, bool)
                or not isinstance(delta, int)
                or delta != expected_delta
                or isinstance(ratio, bool)
                or not isinstance(ratio, (int, float))
                or not math.isfinite(float(ratio))
                or float(ratio) != expected_ratio
            ):
                raise StageGateError(f"development feedback cell {index} has inconsistent deltas")
        elif delta is not None or ratio is not None:
            raise StageGateError(f"development feedback cell {index} exposes a failed comparison")
    summary = document.get("summary")
    if (
        not isinstance(summary, Mapping)
        or set(summary)
        != {
            "members",
            "comparable",
            "all_correct",
            "peak_macs_per_cycle",
            "peak_basis",
            "achievable_macs_per_cycle",
            "achievable_basis",
            # WHERE THE OBJECTIVE'S CYCLES ARE. Derivable from the cells all
            # along and never derived: a campaign converged at 0.2% because 92.4%
            # of its cycles sat in members the agent never aimed at. Required, so
            # a future summary cannot quietly stop saying it.
            "recoverable",
        }
        or summary.get("members") != len(cells)
        or summary.get("comparable") != sum(bool(row["comparable"]) for row in cells)
        # "all correct" is a claim about what was MEASURED. Counting a member the sweep never
        # paid for as a failure reports a correctness problem that nothing observed.
        or summary.get("all_correct") != (summary.get("comparable") == sum(1 for row in cells if row.get("measured")))
    ):
        raise StageGateError("development feedback summary is inconsistent")
    # Exact schemas above already exclude goldens, outputs, paths, shapes, and
    # Verilator.  This serialized audit makes that boundary easy to regression-test.
    encoded = _canonical_json(document).decode("utf-8").lower()
    for forbidden in ('"golden', '"output', '"shape', '"verilator', '"elf', '"path'):
        if forbidden in encoded:
            raise StageGateError(f"development feedback leaks forbidden field {forbidden}")
    return copy.deepcopy(dict(document))


def verify_broker_receipts(
    path: Path, actions: Sequence[BrokerAction], audit: Mapping[str, Any], *, candidate_sha256: str | None = None
) -> dict[str, Any]:
    """Join transcript broker invocations to host-owned append-only completion receipts."""
    if path.is_symlink() or not path.is_file():
        raise StageGateError("host-owned broker receipt stream is absent or linked")
    by_name = {action.name: action for action in actions}
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise StageGateError(f"broker receipt {line_number} is malformed") from exc
        # A refused invocation is a real, host-recorded event and belongs in the join. It can never
        # satisfy a required action: `successful` below counts returncode 0 only, and a refusal is
        # required to carry a non-zero one.
        if (
            not isinstance(row, dict)
            or row.get("receipt_schema_version") != 1
            or row.get("state") not in ("complete", "rejected")
            or (row.get("state") == "rejected" and row.get("returncode") == 0)
            or row.get("action") not in by_name
            or not isinstance(row.get("index"), int)
            or row["index"] != len(rows)
            or not _is_sha256(row.get("argv_sha256"))
            or not _is_sha256(row.get("bindings_command_sha256"))
            or not _is_sha256(row.get("stdout_sha256"))
            or not _is_sha256(row.get("stderr_sha256"))
        ):
            raise StageGateError(f"broker receipt {line_number} violates the action schema")
        rows.append(row)
    try:
        binding = workflow_binding(rows, CORPUS_FEEDBACK_V1)
    except ValueError as exc:
        raise StageGateError(str(exc)) from exc
    invocations = audit.get("broker_invocations")
    if not isinstance(invocations, list):
        raise StageGateError("transcript audit omitted exact broker invocations")
    observed = [(row["action"], row["bindings_command_sha256"]) for row in rows]
    claimed = [(row.get("action"), row.get("bindings_sha256")) for row in invocations if isinstance(row, Mapping)]
    if observed != claimed:
        raise StageGateError("host broker receipts do not exactly match transcript invocations")
    successful = {str(row["action"]) for row in rows if row.get("returncode") == 0}
    required = {action.name for action in actions if action.required}
    missing = sorted(required - successful)
    if missing:
        raise StageGateError(f"required broker actions lack successful receipts: {missing}")
    feedback_rows = [
        row for row in rows if row.get("action") == DEVELOPMENT_FEEDBACK_ACTION and row.get("returncode") == 0
    ]
    if not feedback_rows:
        raise StageGateError("mandatory tuning GSIM feedback was not successfully invoked")
    feedback_receipts: list[dict[str, Any]] = []
    for row in feedback_rows:
        receipt_path, receipt_sha = row.get("feedback_receipt_path"), row.get("feedback_receipt_sha256")
        if not isinstance(receipt_path, str) or not _is_sha256(receipt_sha):
            raise StageGateError("tuning GSIM feedback lacks a content-addressed host receipt")
        receipt = Path(receipt_path)
        expected_root = path.parent.resolve()
        if (
            receipt.is_symlink()
            or not receipt.is_file()
            or expected_root not in receipt.resolve().parents
            or _sha256(receipt.read_bytes()) != receipt_sha
        ):
            raise StageGateError("tuning GSIM feedback host receipt is absent, linked, or changed")
        document = validate_redacted_feedback(json.loads(receipt.read_text(encoding="utf-8")))
        if _sha256(_canonical_json(document)) != receipt_sha:
            raise StageGateError("tuning GSIM feedback receipt is not canonical")
        feedback_receipts.append({"path": str(receipt), "sha256": receipt_sha})
    if candidate_sha256 is not None:
        if not _is_sha256(candidate_sha256):
            raise StageGateError("final round candidate digest is not a SHA-256")
        matching = [
            row
            for row in feedback_receipts
            if validate_redacted_feedback(json.loads(Path(row["path"]).read_text(encoding="utf-8"))).get(
                "candidate_sha256"
            )
            == candidate_sha256
        ]
        if not matching:
            raise StageGateError("mandatory tuning GSIM feedback did not evaluate the final round candidate bytes")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "count": len(rows),
        "successful_actions": sorted(successful),
        "required_actions": sorted(required),
        "feedback_successes": len(feedback_rows),
        "workflow_policy": binding,
        "feedback_receipts": feedback_receipts,
        "candidate_sha256": candidate_sha256,
        "final_candidate_feedback_verified": candidate_sha256 is not None,
        "all_required_succeeded": True,
    }


class CorpusFeedbackPolicy(WorkflowPolicy):
    @property
    def workflow_id(self) -> str:
        return CORPUS_FEEDBACK_V1

    @property
    def unavailable(self):
        return {}

    def build_registry(self):
        return _build_action_registry(self.candidate, self.target_experiment)

    verify_receipts = staticmethod(verify_broker_receipts)

    def execute(self, request, action_name, rendered, call_index, timeout_s, started):
        handlers = {
            E2E_ANALYSIS_ACTION: self._analyze,
            INVENTORY_ACTION: self._inventory,
            OCCUPANCY_PROFILE_ACTION: self._profile,
            ANALYSIS_ACTION: self._command_buffers,
            DEVELOPMENT_FEEDBACK_ACTION: self._feedback,
        }
        handler = handlers.get(action_name)
        if handler is None:
            return None
        return handler(request, action_name, rendered, call_index, timeout_s, started)

    def _analyze_document(self, request, rendered, call_index, timeout_s):
        if self.functional_base is None or self.e2e_sentinel is None:
            raise StageGateError("whole-model baseline or sentinel is unavailable")
        else:
            evaluator = self.feedback_evaluator
            if self.services.whole_model_analysis is None:
                raise StageGateError("whole-model analysis service is unavailable")
            document = self.services.whole_model_analysis(
                self.functional_base,
                self.candidate,
                self.e2e_sentinel,
                timeout_s=timeout_s,
                peak_macs_per_cycle=getattr(evaluator, "peak_macs_per_cycle", None),
                achievable_macs_per_cycle=getattr(evaluator, "achievable_macs_per_cycle", None),
                target=str(getattr(self.target_experiment, "target", "") or ""),
            )
        return document

    def _profile_document(self, request, rendered, call_index, timeout_s):
        if self.feedback_evaluator is None or self.feedback_round is None:
            raise StageGateError("reduced global profile evaluator is unavailable")
        else:
            document = self.feedback_evaluator.profile(
                self.candidate, round_index=self.feedback_round, call_index=call_index, timeout_s=timeout_s
            )
        return document

    def _inventory_document(self, request, rendered, call_index, timeout_s):
        return inspect_compiler_package(self.candidate).to_dict()

    def _feedback(self, request, action_name, rendered, call_index, timeout_s, started):
        feedback_document = None
        try:
            if self.feedback_evaluator is None or self.feedback_round is None:
                raise StageGateError("development GSIM feedback certificate is unavailable")
            feedback_document = validate_redacted_feedback(
                self.feedback_evaluator.evaluate(
                    self.candidate, round_index=self.feedback_round, call_index=call_index, timeout_s=timeout_s
                )
            )
            stdout = _canonical_json(feedback_document).decode("utf-8")
            # THE SEARCH'S OWN VERDICT, kept so the round loop can end on evidence. It was
            # computed and recorded and read by nothing, so a converged search looked exactly
            # like one that had merely run out of rounds.
            stopping = feedback_document.get("stopping")
            if isinstance(stopping, Mapping) and stopping.get("status") == "stop":
                fired = [
                    str(v.get("name"))
                    for v in (stopping.get("verdicts") or [])
                    if isinstance(v, Mapping) and v.get("fired")
                ]
                self.stop_verdict = {
                    "conditions": fired,
                    "share_of_attainable": stopping.get("share_of_attainable"),
                    "queries": stopping.get("queries"),
                }
            result = {
                "returncode": 0,
                "stdout": stdout,
                "stderr": "",
                "elapsed_s": round(time.monotonic() - started, 3),
            }
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
