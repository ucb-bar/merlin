"""Explicit direct-pointer timing intake after original complete source checks.

This adds no source, execution, geometry or activity qualification. The original
queued receipt supplies the checked source proof; current source and emitted ABI
are independently rebound before exact public TRAIN support is considered.
"""

import hashlib
import json

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
    from merlin.common import invocation_record
    from merlin.llvmlower import direct_source_contraction
    from merlin.targetgen import native_pointer_source_binding
    from merlin_experiments.phase1.feedback import candidate_observations

    return (direct_source_contraction, SELECTOR, CHECKED, invocation_record,
            native_pointer_source_binding, candidate_observations)


def emitted_source(member, compiled, workspace, original):
    """Reopen the original direct producer roles, not a generic plan verdict.

    The private queued grade already binds complete source/effect numerics.
    This reuses its original consumption observer for source/LLVM/object/ELF
    custody only; it supplies no source, execution or timing qualification.
    """
    from merlin.targetgen.native_pointer_source_binding import PROTOCOL as NATIVE_PROTOCOL
    from merlin_experiments.phase1.feedback import candidate_observations as PRODUCTS

    grade = original["original_grade"]
    native, emission = grade["candidate_native_execution"], grade["candidate_emission"]
    if (native.get("schema") != "merlin_candidate_native_pointer_source_v1"
            or native.get("source_execution_protocol") != NATIVE_PROTOCOL
            or native.get("status") != "numeric_match_diagnostic"):
        raise ValueError("direct timing intake lacks its original standalone native producer")
    products, producers = PRODUCTS.reopen_native_products(
        workspace, emission=emission, elf_pin=native.get("elf"), source=native.get("source"),
        build_artifacts=native.get("build_artifacts"), candidate=native.get("candidate"),
    )
    pins = {name: F.pin(path, max_bytes=64 << 20) for name, path in products.items()}
    identities = {name: pin["sha256"] for name, pin in pins.items()}
    expected = {
        "kind": "original_native_model_products", "source_sha256": identities["source"],
        "command_buffer_sha256": identities["command_buffer"],
        "producer_sha256s": [F.digest(path) for path in producers],
    }
    artifacts = (compiled.get("candidate_diagnostics") or {}).get("artifacts") or {}
    if (compiled.get("status") != "observed" or compiled.get("artifact_join") != expected
            or compiled.get("elf_sha256") != identities["elf"]
            or compiled.get("llvm_mlir_sha256") != identities["emitted"]
            or any((artifacts.get(name) or {}).get("sha256") != digest for name, digest in identities.items())):
        raise ValueError("direct timing intake differs from its original observed producer roles")
    if member.descriptor.get("interface_mlir") != "capsule.interface.mlir":
        raise ValueError("direct timing source has no exact selected interface role")
    path = member.source_dir / "capsule.interface.mlir"
    source_pin = F.pin(path, max_bytes=MAX_TEXT)
    raw = path.read_bytes()
    if (hashlib.sha256(raw).hexdigest() != source_pin["sha256"]
            or source_pin["sha256"] != identities["source"]
            or F.pin(path, max_bytes=MAX_TEXT) != source_pin):
        raise ValueError("direct timing source differs from the original consumed source")
    cb_path = products["command_buffer"]
    cb_pin = F.pin(cb_path, max_bytes=MAX_TEXT)
    cb_raw = cb_path.read_bytes()
    if hashlib.sha256(cb_raw).hexdigest() != cb_pin["sha256"]:
        raise ValueError("direct timing command buffer changed during reopening")
    return path, source_pin, raw, source_pin["sha256"], cb_path, cb_pin, json.loads(cb_raw), cb_pin["sha256"]


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
        native = "candidate_native_execution" in original["original_grade"]
        # Historical generic emission receipts keep their original reader. A
        # current native receipt cannot fall back when its actual join refuses.
        inputs = (emitted_source(member, value["compiled"], workspace, original) if native
                  else DOMAIN.emitted_source(member, value["compiled"], workspace))
        path, source_pin, raw, digest, cb_path, cb_pin, cb, wanted = inputs
        bound = BIND.bind_direct_source_contraction(path, cb, proof)
        profile = DOMAIN.typed_profile(raw.decode())
        if tuple(bound[key] for key in ("m", "k", "n")) != tuple(profile[key] for key in ("m", "k", "n")):
            raise ValueError("current direct source and positional ABI shapes differ")
        supported = DOMAIN.train_support(catalog, profile)
        if not supported:
            raise ValueError("exact joint source semantics/geometry absent from public TRAIN evidence")
        final_inputs = (emitted_source(member, value["compiled"], workspace, original) if native
                        else DOMAIN.emitted_source(member, value["compiled"], workspace))
        if (final_inputs != inputs
                or F.pin(path, max_bytes=MAX_TEXT) != source_pin or F.pin(cb_path, max_bytes=MAX_TEXT) != cb_pin
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
