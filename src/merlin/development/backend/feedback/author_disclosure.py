"""Author publication policy; private grading documents are never modified here.

Only an explicit ``disclosure.rtl_facts: false`` selects this projection. Old
profiles keep their original objects and serialization. The restricted view is
an allowlisted correctness protocol, not a blacklist of known hardware facts:
unknown diagnostic/advice containers and opaque harness logs stay private.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import tempfile
from pathlib import Path

_DEFAULT = object()
_WITHHELD = "Diagnostic detail withheld by the selected author disclosure policy."
_SCALARS = frozenset(
    """
schema version protocol selfcheck_protocol selfcheck_request_id request_id
submission requested_submission official graded_newer_bytes
submission_sha256 requested_submission_sha256 graded_source_sha256
current_source_sha256 source_sha256 source_hash_algorithm execution_digest
requested_at_unix_ns grade_started_at_unix_ns graded_at_unix_ns
graded_at source_checked_at snapshot_current current_source_eligible
current_source_eligibility_scope count_scope currency token reach_sha256
capsule name test_case suite sim requested_sim model program split mode
status pass passed all_pass capsule_all_pass complete gradeable certified
state n_selected n_completed n_execution_passed certified_complete previous_failures_are_current_verdict
fail failed unavailable skipped gated running unknown numerics final
plane category tier component first_bad_group returncode
mandatory_tiers_complete original_cohort_complete structural_evidence_complete
numeric_status numeric_consistency_passed numerical_execution_passed
structural_qualification barrier_tier barrier_declared barrier_status
native_executed qualification screened_only mismatch_count output index your_value
observed count total n_passed n_capsules n_required_development n_selected_development
max_abs_error mean_abs_error max_relative_error mean_relative_error cosine_similarity
atol rtol max_error relative_error absolute_error
n_pending_development current_passed current_total executed carried tiers_abandoned
rows from_newest_grade from_current_source older_version_rows older_version_passed
current_reach_sha256 reconciled_reach_sha256 qa_gate
trace_status trace_requirement_status failure_plane failure_category failure_tier
required_tiers tiers_graded tiers_not_run result_rows gated_rows suite_size
selected_count requested_count completed pending queued in_flight healthy expired
elapsed_s duration_s timeout_s queued_behind_this claimed_at_unix_ns phase stage
n_remaining n_numerics_decided n_finished n_expected n_running n_total wall_s
completed_at_unix_ns started_at_unix_ns response_available source_changed
kind cycles t recorded_at grade_token l2 l3 sha256 namespace
recorded_grades current_rows total_rows newer_rows older_rows timestamp
member_index public_member member_count numerical_status selected_rtl_status source_stability
wall_seconds authority
""".split()
)
_CONTAINERS = frozenset(
    {
        "numeric",
        "first_mismatch",
        "expected_refusal",
        "failure",
        "graded",
        "structural_qualification",
        "latest_completed_grade",
        "completed_rows",
        "current_source",
        "model_gate",
        "source_identity",
        "tier_reuse",
        "candidate_native_verification",
        "result_index",
        "counts",
        "numerics_counts",
        "summary",
        "progress",
        "qualification_counts",
        "request",
        "timing",
        "report",
    }
)
_ROWS = frozenset({"per_capsule", "cases", "results", "by_grade", "failures", "members"})
_NAMED_ROWS = frozenset({"programs", "models"})
_TIER_MAPS = frozenset({"tiers", "tier_cycles", "tier_engines"})
_CANDIDATE = frozenset({"declined", "candidate_diagnostics", "candidate_compile_error"})
_DIAGNOSTICS = frozenset({"error", "reason", "failure_detail", "detail", "tier_reasons"})
_CANDIDATE_ARTIFACTS = frozenset({"command_buffer.json", "lowered.llvm.mlir"})


def restricted(profile=_DEFAULT):
    """Use the trusted invocation profile, never an author-writable record."""
    if profile is _DEFAULT:
        from merlin.development.backend.author_profiles import current

        profile = current()
    if profile is None:
        return False
    disclosure = getattr(profile, "disclosure", None)
    if disclosure is None:
        disclosure = profile.record().get("disclosure")
    return isinstance(disclosure, dict) and disclosure.get("rtl_facts") is False


def _plain(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value if not isinstance(item, dict)]
    return None


def _performance(value):
    if not isinstance(value, dict):
        return None
    cycles = value.get("estimated_whole_soc_cycles")
    if type(cycles) not in (int, float) or not math.isfinite(cycles) or cycles < 0:
        return None
    return {
        "status": "PROVISIONAL",
        "estimated_whole_soc_cycles": cycles,
        "evidence_kind": "provisional_spike_total",
        "qualification_issued": False,
    }


def _project(document, workspace=None):
    if isinstance(document, list):
        return [_project(item, workspace) for item in document if isinstance(item, dict)]
    if not isinstance(document, dict):
        return _plain(document)
    candidate_failure = (
        document.get("plane", document.get("failure_plane")) == "candidate_compilation"
        and document.get("category", document.get("failure_category")) == "CANDIDATE_EMISSION_FAILED"
    )
    out = {}
    for key, value in document.items():
        if key == "performance":
            projected = _performance(value)
            if projected is not None:
                out[key] = projected
        elif key in _CANDIDATE:
            # These are explicitly candidate-authored data, not harness console
            # text or decoder diagnostics. Do not rewrite the author's own code.
            out[key] = copy.deepcopy(value)
        elif key == "your_artifacts" and isinstance(value, dict):
            artifacts = {}
            for name, path in value.items():
                if name not in _CANDIDATE_ARTIFACTS or not isinstance(path, str):
                    continue
                candidate = Path(path)
                if candidate.is_absolute():
                    if workspace is None or not candidate.is_relative_to(Path(workspace)):
                        continue
                    candidate = candidate.relative_to(workspace)
                if (
                    candidate.parts
                    and candidate.parts[0] == "selfcheck_out"
                    and candidate.name == name
                    and ".." not in candidate.parts
                    and "\\" not in path
                    and not any(character.isspace() for character in path)
                ):
                    artifacts[name] = candidate.as_posix()
            out[key] = artifacts
        elif key in _TIER_MAPS and isinstance(value, dict):
            out[key] = {
                tier: _project(item, workspace) if isinstance(item, dict) else _plain(item)
                for tier, item in value.items()
                if tier in {"L0", "L1", "L2", "L3"}
            }
        elif key in _NAMED_ROWS and isinstance(value, dict):
            out[key] = {name: _project(item, workspace) for name, item in value.items() if isinstance(item, dict)}
        elif key in _ROWS and isinstance(value, list):
            out[key] = _project(value, workspace)
        elif key in _CONTAINERS and isinstance(value, dict):
            out[key] = _project(value, workspace)
        elif candidate_failure and key in {"detail", "failure_detail"} and isinstance(value, str):
            out[key] = value
        elif key in _DIAGNOSTICS and value is not None:
            # Free-form host errors can quote capacities, address encodings or
            # model parameters. Their typed plane/category and numeric outcome
            # remain available; candidate diagnostics have their own owner above.
            out[key] = _WITHHELD
        elif key in _SCALARS:
            # Protocol metadata is an identifier, never an arbitrary scope/path
            # sentence that could quote an input program or extracted fact.
            if isinstance(value, str) and key not in {"your_value", "observed"}:
                if len(value) > 256 or any(
                    character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:+@-"
                    for character in value
                ):
                    continue
                if key.endswith("sha256") and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
                    continue
            out[key] = _plain(value)
    return out


def project(document, *, workspace=None, profile=_DEFAULT):
    """Return an independent author view; leave the private input untouched."""
    if not restricted(profile):
        return document
    if workspace is not None:
        from merlin.development.backend.tools.result_details import hydrate

        document = hydrate(document, workspace=workspace)
    return _project(document, workspace)


def json_text(text, *, workspace=None):
    """Project JSON immediately before public response/progress publication."""
    if not restricted():
        return text
    try:
        document = json.loads(text)
    except (ValueError, TypeError):
        document = {"error": _WITHHELD, "all_pass": False}
    return json.dumps(project(document, workspace=workspace), sort_keys=True)


def copy_candidate_artifact(source, destination):
    """Copy only the two raw candidate outputs, with exact-byte verification.

    The caller must supply its original candidate-generated artifact path. This
    does not establish producer provenance for arbitrary files or decoded traces.
    """
    source, destination = Path(source), Path(destination)
    if source.name not in _CANDIDATE_ARTIFACTS or source.is_symlink() or not source.is_file():
        return False
    data = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    if source.read_bytes() != data or destination.read_bytes() != data:
        destination.unlink(missing_ok=True)
        return False
    return True


def private_work_root():
    """Require the operator's existing private evidence selection, without fallback."""
    from merlin.development.backend.feedback.raw_selfcheck_evidence import _selected_root, create_runs_root

    if _selected_root(os.environ) is None:
        raise ValueError("author feedback requires operator-private evidence staging")
    return create_runs_root()


def broker_publication(root, workspace, channel, log):
    """Select private broker logs and evidence environment for restricted profiles."""
    public = Path(workspace) / channel / log
    if not restricted():
        return None, public
    from merlin.development.backend.feedback.raw_selfcheck_evidence import broker_environment, operator_log_dir

    environment = broker_environment(root, workspace)
    directory = operator_log_dir(environ=environment)
    if directory is None:
        raise ValueError("restricted feedback requires operator-private broker logs")
    return environment, directory / (channel.lstrip(".") + "-" + log)


def _publish_response(path, document):
    text = json.dumps(project(document), indent=2, sort_keys=True)
    if path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=destination.parent, prefix=".disclosure-", delete=False, encoding="utf-8"
        ) as stream:
            stream.write(text)
            temporary = Path(stream.name)
        os.replace(temporary, destination)
    print(text)


def private_cli(arguments, callback):
    """Run an unchanged CLI check privately, then publish its curated JSON only.

    Invocation context and compiler inputs remain unchanged. Every raw artifact,
    temporary file, diagnostic stream and intermediate response stays under the
    existing operator-private evidence root. This entry is for one-request CLI
    children; the broker itself never changes its process environment.
    """
    if not restricted():
        return callback(arguments)
    output = getattr(arguments, "out", None)
    try:
        private = private_work_root()
    except (OSError, ValueError):
        _publish_response(
            output, {"status": "UNKNOWN", "all_pass": False, "error": "operator-private feedback staging unavailable"}
        )
        return 2
    selected = copy.copy(arguments)
    selected.out = str(private / "response.json")
    selected.artifact_root = private / "artifacts"
    old_tmpdir, old_tempdir = os.environ.get("TMPDIR"), tempfile.tempdir
    try:
        os.environ["TMPDIR"] = str(private)
        tempfile.tempdir = str(private)
        with (
            (private / "stdout.log").open("w") as stdout,
            (private / "stderr.log").open("w") as stderr,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            try:
                code = callback(selected)
                document = json.loads(Path(selected.out).read_text())
            except (Exception, SystemExit):
                import traceback

                traceback.print_exc()
                # A partial/malformed response cannot claim any result. Detailed
                # diagnostics remain private, and no source excerpt is echoed.
                code, document = (
                    2,
                    {"status": "UNKNOWN", "all_pass": False, "error": "private diagnostic response unavailable"},
                )
    finally:
        tempfile.tempdir = old_tempdir
        if old_tmpdir is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = old_tmpdir
    _publish_response(output, document)
    return code


def unavailable(output):
    """Refuse a fact-derived optional probe before supplying private probe inputs."""
    _publish_response(
        output,
        {"status": "UNKNOWN", "all_pass": False, "error": "probe unavailable under the selected disclosure policy"},
    )
    return 2
