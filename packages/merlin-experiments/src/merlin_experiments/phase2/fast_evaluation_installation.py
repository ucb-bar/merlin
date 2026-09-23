"""Pinned host-only fast-evaluation installation and worker argument contract."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
from pathlib import Path

from merlin.common.digest import sha256_bytes

from . import broker_evidence as BE
from . import contracts as P2_CONTRACTS

_HOST_QUALITY_OBSERVER_CONTRACT = "host_reference_only_no_target_or_model_simulator_v1"


def _exact_file(path: Path, digest: str, *, label: str) -> Path:
    """Resolve and verify one host-owned input without following a link."""
    resolved = Path(path).resolve()
    if (
        not BE._is_sha256(digest)
        or Path(path).is_symlink()
        or not resolved.is_file()
        or P2_CONTRACTS.sha256_file(resolved) != digest
    ):
        raise ValueError(f"{label} is absent, linked, or differs from its exact SHA-256 pin")
    return resolved


def worker_arguments(
    calibration: Path | None,
    calibration_sha256: str | None,
    quality_observer: Path | None,
    quality_observer_sha256: str | None,
    quality_observer_symbol: str | None,
    classification_member_sha256: str | None,
    corpora: list[list[str]],
    maximum_model_seconds: float,
) -> tuple[str, ...]:
    """Forward fast-evaluator host inputs exactly; the source worker re-verifies every pin."""
    values: list[str] = ["--fast-evaluation-maximum-model-seconds", str(maximum_model_seconds)]
    if calibration is not None:
        values.extend(
            (
                "--fast-evaluation-calibration",
                str(calibration.resolve()),
                "--fast-evaluation-calibration-sha256",
                str(calibration_sha256),
            )
        )
    if quality_observer is not None:
        values.extend(
            (
                "--fast-evaluation-quality-observer",
                str(quality_observer.resolve()),
                "--fast-evaluation-quality-observer-sha256",
                str(quality_observer_sha256),
                "--fast-evaluation-quality-observer-symbol",
                str(quality_observer_symbol),
            )
        )
    if classification_member_sha256 is not None:
        values.extend(("--fast-evaluation-classification-member-sha256", classification_member_sha256))
    for member_sha256, path, corpus_sha256 in corpora:
        values.extend(("--fast-evaluation-held-out-corpus", member_sha256, str(Path(path).resolve()), corpus_sha256))
    return tuple(values)


def validate_cli(args, parser: argparse.ArgumentParser) -> bool:
    """Validate the all-or-none accuracy-bounded installation contract.

    A run with no supplied fast-evaluation evidence is valid and stays exact-only. A partial
    accuracy configuration is rejected rather than silently losing approximation authority.
    """
    paired = (
        (args.fast_evaluation_calibration, args.fast_evaluation_calibration_sha256, "calibration"),
        (args.fast_evaluation_quality_observer, args.fast_evaluation_quality_observer_sha256, "quality observer"),
    )
    for path, digest, label in paired:
        if bool(path) != bool(digest):
            parser.error(f"fast-evaluation {label} requires both path and exact SHA-256")
    if (
        isinstance(args.fast_evaluation_maximum_model_seconds, bool)
        or not 0 < args.fast_evaluation_maximum_model_seconds <= 60
    ):
        parser.error("fast-evaluation model budget must be in (0, 60] seconds")
    requested = any(
        (
            args.fast_evaluation_calibration,
            args.fast_evaluation_quality_observer,
            args.fast_evaluation_quality_observer_symbol,
            args.fast_evaluation_classification_member_sha256,
            args.fast_evaluation_held_out_corpus,
        )
    )
    if not requested:
        return False
    if (
        args.fast_evaluation_calibration is None
        or args.fast_evaluation_quality_observer is None
        or not args.fast_evaluation_quality_observer_symbol
        or not BE._is_sha256(args.fast_evaluation_classification_member_sha256)
        or len(args.fast_evaluation_held_out_corpus) != 4
    ):
        parser.error(
            "accuracy-bounded fast evaluation requires a pinned calibration, pinned host quality "
            "observer and symbol, classification member identity, and exactly four held-out corpora"
        )
    members = [row[0] for row in args.fast_evaluation_held_out_corpus]
    if (
        len(set(members)) != 4
        or any(not BE._is_sha256(member) for member in members)
        or any(not BE._is_sha256(row[2]) for row in args.fast_evaluation_held_out_corpus)
    ):
        parser.error("fast-evaluation corpus bindings require four distinct member and corpus SHA-256s")
    try:
        _exact_file(
            args.fast_evaluation_calibration,
            args.fast_evaluation_calibration_sha256,
            label="fast-evaluation calibration JSON",
        )
        _exact_file(
            args.fast_evaluation_quality_observer,
            args.fast_evaluation_quality_observer_sha256,
            label="fast-evaluation host quality observer",
        )
        for _member, path, digest in args.fast_evaluation_held_out_corpus:
            _exact_file(Path(path), digest, label="fast-evaluation held-out corpus")
    except ValueError as exc:
        parser.error(str(exc))
    return True


def _load_host_quality_observer(path: Path, digest: str, symbol: str, corpus_records: dict[str, dict[str, str]]):
    """Load one pinned host-only adapter and bind each invocation to its exact corpus bytes."""
    observer_path = _exact_file(path, digest, label="fast-evaluation host quality observer")
    payload = observer_path.read_bytes()
    if sha256_bytes(payload) != digest:
        raise ValueError("host quality observer changed before installation")
    module_name = "_merlin_phase2_quality_observer_" + digest
    spec = importlib.util.spec_from_file_location(module_name, observer_path)
    if spec is None or spec.loader is None:
        raise ValueError("fast-evaluation quality observer cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # Execute the bytes just verified, never timestamp/size-selected cached bytecode.
    try:
        exec(compile(payload, str(observer_path), "exec"), module.__dict__)
    except BaseException:
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        raise
    if getattr(module, "MERLIN_HOST_QUALITY_OBSERVER_CONTRACT", None) != _HOST_QUALITY_OBSERVER_CONTRACT:
        raise ValueError("quality observer does not declare the host-only execution contract")
    implementation = getattr(module, symbol, None)
    if not callable(implementation):
        raise ValueError("quality observer symbol is absent or not callable")
    frozen_corpus_records = copy.deepcopy(corpus_records)

    def observe(**kwargs):
        if P2_CONTRACTS.sha256_file(observer_path) != digest:
            raise ValueError("host quality observer changed after installation")
        sentinel = kwargs.get("sentinel")
        member_sha256 = getattr(sentinel, "capsule_sha256", None)
        record = frozen_corpus_records.get(member_sha256)
        if record is None:
            raise ValueError("quality observer received an unbound portfolio member")
        corpus_path = _exact_file(Path(record["path"]), record["sha256"], label="fast-evaluation held-out corpus")
        return implementation(**kwargs, corpus_path=corpus_path)

    return observe


def prepare(
    *,
    sentinels,
    target_sha256: str,
    stage_root: Path,
    configured: bool,
    calibration: Path | None,
    calibration_sha256: str | None,
    quality_observer: Path | None,
    quality_observer_sha256: str | None,
    quality_observer_symbol: str | None,
    classification_member_sha256: str | None,
    corpora: list[list[str]],
    maximum_model_seconds: float,
):
    """Build and receipt the production host evaluator, or retain an exact-only loop."""
    from merlin.perf.phase2_analytical_provider import build_fast_evaluator_installation
    from merlin.perf.phase2_portfolio import unavailable_fast_evaluation

    members = tuple(sentinel.capsule_sha256 for sentinel in sentinels)
    if len(members) != 4 or len(set(members)) != 4 or any(not BE._is_sha256(member) for member in members):
        if configured:
            raise ValueError("accuracy-bounded fast evaluation requires exactly four content-addressed sentinels")
        installation = None
        fallback = unavailable_fast_evaluation(
            reason="four distinct content-addressed portfolio sentinels are not installed"
        )
        receipt = {
            "schema": "phase2_fast_evaluator_installation_receipt_v1",
            "status": "exact_only_fallback",
            "portfolio_member_sha256s": list(members),
            "quality_schema": None,
            "provider_binding": None,
            "fallback": fallback,
            "input_bindings": None,
            "experiment_kwargs_installed": [],
            "execution": "host_analytical_only_no_complete_model_or_layer_simulation",
        }
    else:
        corpus_records: dict[str, dict[str, str]] = {}
        observer = None
        if configured:
            for member_sha256, raw_path, digest in corpora:
                path = _exact_file(Path(raw_path), digest, label="fast-evaluation held-out corpus")
                corpus_records[member_sha256] = {"path": str(path), "sha256": digest, "hash_scope": "exact_file_bytes"}
            if set(corpus_records) != set(members):
                raise ValueError("held-out corpus pins must exactly cover the four sentinels")
            if classification_member_sha256 not in members:
                raise ValueError("classification member must be one of the four sentinels")
            observer = _load_host_quality_observer(
                quality_observer, quality_observer_sha256, quality_observer_symbol, corpus_records
            )
        installation = build_fast_evaluator_installation(
            members,
            classification_member_sha256=(classification_member_sha256 or members[0]),
            corpus_sha256_by_member=(
                {member: row["sha256"] for member, row in corpus_records.items()} if configured else None
            ),
            calibration=calibration if configured else None,
            calibration_sha256=calibration_sha256 if configured else None,
            quality_observer=observer,
            quality_observer_sha256=quality_observer_sha256 if configured else None,
            maximum_model_seconds=maximum_model_seconds,
        )
        if (
            installation.provider_binding is not None
            and installation.provider_binding.get("target_sha256") != target_sha256
        ):
            raise ValueError("fast-evaluation calibration targets different descriptor bytes")
        receipt = {
            "schema": "phase2_fast_evaluator_installation_receipt_v1",
            "status": ("installed" if installation.provider is not None else "exact_only_fallback"),
            "portfolio_member_sha256s": list(members),
            "quality_schema": installation.quality_schema.to_dict(),
            "provider_binding": copy.deepcopy(installation.provider_binding),
            "fallback": copy.deepcopy(installation.fallback),
            "input_bindings": (
                {
                    "calibration": {
                        "path": str(Path(calibration).resolve()),
                        "sha256": calibration_sha256,
                        "hash_scope": "exact_file_bytes",
                    },
                    "quality_observer": {
                        "path": str(Path(quality_observer).resolve()),
                        "sha256": quality_observer_sha256,
                        "symbol": quality_observer_symbol,
                        "contract": _HOST_QUALITY_OBSERVER_CONTRACT,
                        "hash_scope": "exact_file_bytes",
                    },
                    "held_out_corpora": corpus_records,
                    "classification_member_sha256": classification_member_sha256,
                }
                if configured
                else None
            ),
            "experiment_kwargs_installed": sorted(installation.experiment_kwargs()),
            "execution": "host_analytical_only_no_complete_model_or_layer_simulation",
        }
    receipt_path = stage_root / "fast_evaluation_installation.json"
    P2_CONTRACTS.write_json(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)
    receipt["receipt_sha256"] = P2_CONTRACTS.sha256_file(receipt_path)
    return installation, receipt
