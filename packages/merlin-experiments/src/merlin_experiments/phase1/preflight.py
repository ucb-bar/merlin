"""Stop an ordinary reviewed session after readiness, before authoring or grading.

This continuation reopens the existing owners. Its result records only that the
ordinary native startup checks completed; it cannot certify a compiler, runtime,
numerical policy, fresh client isolation or any phase.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from merlin.targetgen.sandbox import bwrap
from merlin_experiments.corpus.release import verify_snapshot_for_phase1

from . import run_inputs


def complete(prepared) -> int:
    """Consume only the actual fresh reviewed normal preparation."""
    request = prepared.request
    if not request.options.preflight_only or prepared.resuming or prepared.reviewed_roots is None:
        raise ValueError("preflight-only needs a fresh reviewed ordinary preparation")
    from .session import validate_preflight_options

    validate_preflight_options(request.options)
    identity = verify_prepared_inputs(prepared)
    record = {
        "schema": "phase1_readiness_preflight.v1",
        "mode": "preflight_only",
        "status": "startup_checks_completed",
        "formal_complete": False,
        "provider_started": False,
        **({"runtime_selection": prepared.selected_codex_runtime.record()} if request.options.codex_binary else {}),
        "bundle_manifest_sha256": identity,
        "snapshot_content_sha256": prepared.environment["bundle_input_snapshot"]["content_sha256"],
        "scope": "ordinary native startup only; no author/client isolation, compiler, runtime or phase qualification",
    }
    path = prepared.run_dir / "preflight_result.json"
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "w") as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
    return 0


def verify_prepared_inputs(prepared) -> str:
    """Replay the same complete fresh reviewed input owners for both continuations."""
    request = prepared.request
    if prepared.resuming or prepared.reviewed_roots is None:
        raise ValueError("readiness needs a fresh reviewed ordinary preparation")
    prepared.verify_inputs()
    manifest = prepared.run_dir / "input_bundle_manifest.yaml"
    identity = run_inputs.bundle_manifest_identity(manifest, prepared.bundle)
    if identity != prepared.environment["bundle_manifest_sha256"]:
        raise ValueError("preflight-only admitted manifest changed")
    bwrap.verify_snapshot_binding(
        prepared.workspace,
        prepared.bundle,
        prepared.environment["bundle_input_snapshot"],
        repo=request.context.repo,
    )
    verify_snapshot_for_phase1(
        Path(request.options.corpus_seal),
        request.context.descriptor,
        prepared.workspace,
        prepared.bundle,
        repo=request.context.repo,
    )
    return identity
