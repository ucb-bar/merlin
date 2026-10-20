"""Explicit direct-pointer timing intake after original complete source checks.

This adds no source, execution, geometry or activity qualification. The original
queued receipt supplies the checked source proof; current source and emitted ABI
are independently rebound before exact public TRAIN support is considered.
"""

from merlin.perf.candidate_source import MAX_TEXT
from merlin.runtime import _native_files as F
from merlin.targetgen import checked_source_model_diagnostic as SELECTOR

from . import _archive_source_domain as DOMAIN
from . import archive_native_screen as NATIVE
from . import ordinary_checked_source as CHECKED
from .contracts import StageGateError, document_sha256

PROTOCOL = "merlin.public_native_direct_pointer_domain.v1"


def source_modules():
    """Load the extra core checker only for the explicitly selected protocol."""
    from merlin.llvmlower import direct_source_contraction

    return (direct_source_contraction, SELECTOR, CHECKED)


def selection_record(policy):
    """Require the original live parent selector; the child carries its record."""
    record = CHECKED.record(policy._checked_source_diagnostic, policy._compiler_pass_selection)
    if record.get("native_execution_protocol") != SELECTOR.POINTER_PROTOCOL:
        raise StageGateError("direct timing domain requires the original explicit pointer-source selection")
    return record


def current(policy, member, workspace, value, catalog):
    """No saved status or caller-supplied proof can substitute for the queued receipt."""
    from xdsl.utils.exceptions import ParseError, VerifyException

    from merlin.llvmlower import direct_source_contraction as BIND

    unknown = {"status": "UNKNOWN", "family": None}
    selected = selection_record(policy)
    checked = policy._checked_source_diagnostic, policy._compiler_pass_selection
    original, receipt_pin = NATIVE.original_numerical_observation(workspace, value, checked_source=checked)
    try:
        if (member.descriptor.get("operation") or {}).get("op") != "model":
            raise ValueError("direct timing intake requires a typed public model source")
        if member.descriptor.get("performance", {}).get("capacity"):
            raise ValueError("capacity extrapolation has no paired public qualification")
        qualification = original["original_grade"].get("complete_source_qualification") or {}
        proof = qualification.get("source_proof") or {}
        realization = qualification.get("realization") or {}
        from merlin_experiments.phase0 import connected_integer_qualification as SOURCE

        if (qualification.get("schema") != "merlin.phase0.connected_integer_qualification.v1"
                or qualification.get("status") != "qualified" or qualification.get("method") != SOURCE.SELECTION
                or realization.get("status") != "verified_source_execution_correspondence"
                or realization.get("source_sha256") != proof.get("interface_sha256")
                or realization.get("source_rule") != qualification.get("source_rule")):
            raise ValueError("original queued grade lacks its complete source/effect qualification")
        path, source_pin, raw, digest, cb_path, cb_pin, cb, wanted = DOMAIN.emitted_source(
            member, value["compiled"], workspace)
        bound = BIND.bind_direct_source_contraction(path, cb, proof)
        profile = DOMAIN.typed_profile(raw.decode())
        if tuple(bound[key] for key in ("m", "k", "n")) != tuple(profile[key] for key in ("m", "k", "n")):
            raise ValueError("current direct source and positional ABI shapes differ")
        supported = DOMAIN.train_support(catalog, profile)
        if not supported:
            raise ValueError("exact joint source semantics/geometry absent from public TRAIN evidence")
        if (F.pin(path, max_bytes=MAX_TEXT) != source_pin or F.pin(cb_path, max_bytes=MAX_TEXT) != cb_pin
                or F.pin(workspace / "ordinary_observation.json", max_bytes=64 << 20) != receipt_pin
                or selection_record(policy) != selected):
            raise ValueError("direct timing source, compiled ABI, receipt or live selection changed")
        result = {"status": "source_domain_supported", "family": profile["family"], "profile": profile,
                  "source_sha256": digest, "command_buffer_sha256": wanted,
                  "training_source_groups": supported, "source_domain_protocol": PROTOCOL,
                  "numerical_receipt_sha256": receipt_pin["sha256"],
                  "authority": "source timing-family/domain selection only; no target or timing qualification"}
        result["sha256"] = document_sha256(result)
        return result
    except (ValueError, KeyError, TypeError, AttributeError, NotImplementedError) as error:
        return {**unknown, "reason": "current direct-pointer source domain refused: " + str(error)[:180]}
    except (OSError, ParseError, VerifyException) as error:
        return {**unknown, "reason": "current direct-pointer source unavailable: " + type(error).__name__}
