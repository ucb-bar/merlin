"""Shared capture source identity and preapplied selection helpers."""


def unavailable_original_snapshot(reason="selected model2MLIR does not expose frontend capture tracing"):
    """Keep unavailable source identity explicit through capture failure paths."""
    return {"status": "unavailable", "stage": "original", "reason": reason}


def external_original_snapshot(exported, selection, config, original_snapshot, *, trace_supported):
    """Attest only the explicitly selected external export as the numeric source."""
    if not trace_supported or exported is None or not all(selection):
        return original_snapshot
    from m2m.capture.external_quantization import validate_manifest
    from m2m.capture.trace import snapshot_exported_program

    # The selected external export is the numeric source, not an earlier
    # floating-point model. Authenticate it before stamping this exact graph;
    # conversion must consume the same export without recapture or casts.
    validate_manifest(getattr(exported, "_m2m_external_quantization_manifest", None), config)
    return snapshot_exported_program(exported, stage="original")


def bundle_metadata(conversion_result):
    """Derive buffer roles from the same prepared export used by conversion."""
    if __package__:
        from ._pt2_capture_worker import _read_only_buffer_arguments
    else:
        from _pt2_capture_worker import _read_only_buffer_arguments

    exported = conversion_result.exported_program
    return {"read_only_buffer_arguments": _read_only_buffer_arguments(exported) if exported is not None else []}


def add_arguments(parser):
    parser.add_argument(
        "--already-quantized",
        action="store_true",
        help="capture the loader's own materialized numeric graph without applying a TorchAO recipe or scheme",
    )
    parser.add_argument("--quant-adapter", default="", help="installed external quantization adapter")
    parser.add_argument("--contract", default="", help="selected external contract JSON")
    parser.add_argument("--policy", default="", help="selected external policy JSON")
    parser.add_argument("--expected-contract-sha256", default="", help="expected contract SHA-256 from the source receipt")
    parser.add_argument("--expected-policy-sha256", default="", help="expected policy SHA-256 from the source receipt")


def select_external(args, parser):
    selection = tuple(getattr(args, name, "") for name in ("quant_adapter", "contract", "policy"))
    digests = tuple(getattr(args, name, "") for name in ("expected_contract_sha256", "expected_policy_sha256"))
    if any(selection) and (not all(selection) or not args.already_quantized):
        parser.error("--quant-adapter, --contract, and --policy require each other and --already-quantized")
    if any(digests) and not all(selection):
        parser.error("expected source digests require an external quantization selection")
    for digest in digests:
        if digest and (len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)):
            parser.error("expected source digests must be lowercase SHA-256 values")
    return selection


def prepared_module(model, *, already_quantized):
    # External adapters return an ExportedProgram with node-bound receipts. Calling
    # eval() on its unlifted module is unsupported, and re-export would stale those
    # bindings. The PT2 file worker does not accept loader-owned adapter selection.
    exported = model if hasattr(model, "graph_signature") and hasattr(model, "graph_module") else None
    if exported is not None:
        if not already_quantized:
            raise ValueError("an exported loader requires --already-quantized to preserve capture semantics")
        return exported, exported.module()
    model.eval()
    return None, model


def normalization_options(loader, selection):
    if not all(selection):
        return {}
    normalize = getattr(loader, "normalize_capture", None)
    if not callable(normalize):
        raise ValueError("external already-quantized loader must expose normalize_capture(module, manifest)")
    return {"normalize": normalize}


def quantization(args, selection, *, recipe, quant_for):
    """Select a capture configuration without reapplying a loader's numerical transforms."""
    q = None if (recipe is not None or args.already_quantized) else quant_for(args.dtype, args.scheme or None)
    if all(selection):
        from m2m.capture.external_quantization import ExternalQuantizationConfig

        q = ExternalQuantizationConfig(
            *selection,
            getattr(args, "expected_contract_sha256", "") or None,
            getattr(args, "expected_policy_sha256", "") or None,
        )
    quant_stats = (
        {"applied": False, "why": "the loader declares its numeric graph already materialized"}
        if args.already_quantized
        else None
    )
    return q, quant_stats
