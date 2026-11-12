"""Validation of extractor and consumed-source byte commitments."""

from pathlib import Path

from merlin.common.digest import is_sha256


def cache_pins_match(doc: dict, *, contract_path, file_digest) -> bool:
    """Check declared extractor and consumed-source commitments before cache reuse.

    This is not hardware qualification. Artifacts without these optional pins
    remain legacy, unverified evidence. A declared pin whose source cannot be
    resolved or no longer matches is not a reusable cache entry.
    """
    from merlin.common.paths import module_source_path

    inputs = doc.get("inputs") or {}
    if not isinstance(inputs, dict):
        return False

    checked = {}

    def matches(path, digest):
        if not is_sha256(digest) or not isinstance(path, (str, Path)) or not path:
            return False
        try:
            if path not in checked:
                checked[path] = file_digest(path)
            return checked[path] == digest
        except OSError:
            return False

    extractor = inputs.get("extractor_sha256")
    if extractor is not None:
        generator = doc.get("generator") or {}
        # Only these core extractors declare this commitment. Do not
        # import arbitrary generator names from an evidence document.
        owners = {"merlin.development.targetgen.rtl.circt_introspect", "merlin.development.targetgen.rtl.introspect"}
        if not isinstance(generator, dict) or generator.get("name") not in owners:
            return False
        if not matches(module_source_path(generator["name"]), extractor):
            return False
    reader = inputs.get("extraction_reader_sha256")
    if reader is not None and not matches(
        module_source_path("merlin.development.targetgen.rtl.extraction_contract"), reader
    ):
        return False
    extraction_contract = inputs.get("extraction_contract_sha256")
    if extraction_contract is not None:
        target = inputs.get("target")
        if not isinstance(target, str) or not target or not matches(contract_path(target), extraction_contract):
            return False
    bundle = inputs.get("source_bundle_path")
    if bundle is not None:
        if not matches(bundle, inputs.get("source_bundle_sha256")):
            return False
        from merlin.development.targetgen.rtl.source_selection import load_selection, production_consistency

        try:
            selected = load_selection(bundle, target=inputs.get("target"))
            if production_consistency(selected)["status"] != "verified":
                return False
        except (OSError, ValueError, KeyError, TypeError):
            return False
    genericization = (doc.get("source_consistency") or {}).get("genericization")
    if genericization is not None:
        if not isinstance(genericization, dict):
            return False
        for key in ("input", "output", "tool"):
            member = genericization.get(key)
            if not isinstance(member, dict) or not matches(member.get("path"), member.get("sha256")):
                return False
    body = doc.get("facts") or {}
    if not isinstance(body, dict):
        return False
    sentinels = (None, "unresolved", "missing", "n/a")
    # Legacy documents lacking paths remain diagnostic evidence. New documents
    # bind all resolved paths, including the full census FIRRTL and hierarchy;
    # changing either invalidates the extraction even when features are absent.
    for owner in (inputs, body.get("source") or {}):
        if not isinstance(owner, dict):
            return False
        for path_key, digest_key in (
            ("hw_path", "hw_sha256"),
            ("core_hw_path", "core_hw_sha256"),
            ("generic_hw_path", "generic_hw_sha256"),
            ("fir_path", "fir_sha256"),
            ("hierarchy_path", "hierarchy_sha256"),
            ("isa_path", "isa_sha256"),
        ):
            if owner.get(path_key):
                digest = owner.get(digest_key)
                if digest == "missing" and Path(owner[path_key]).is_file():
                    return False
                if digest not in sentinels and not matches(owner[path_key], digest):
                    return False
        for key in ("firrtl_inputs", "reader_sources"):
            reads = owner.get(key) or []
            if not isinstance(reads, list):
                return False
            for receipt in reads:
                if not isinstance(receipt, dict):
                    return False
                if receipt.get("sha256") == "missing" and receipt.get("path") and Path(receipt["path"]).is_file():
                    return False
                if receipt.get("sha256") not in sentinels and not matches(receipt.get("path"), receipt["sha256"]):
                    return False
    interfaces = body.get("interfaces") or []
    fir_sources = []
    for item in interfaces:
        if not isinstance(item, dict):
            continue
        digest = item.get("source_sha256")
        if digest in (None, "unresolved", "missing", "n/a"):
            continue
        if not matches(item.get("source"), digest):
            return False
        if item.get("name") == "elaborated_rtl_features":
            fir_sources.append(item.get("source"))
    fir_digest = inputs.get("fir_sha256")
    if fir_digest not in (None, "unresolved", "missing", "n/a"):
        sources = [inputs["fir_path"]] if inputs.get("fir_path") else fir_sources
        if not sources or not all(matches(path, fir_digest) for path in sources):
            return False
    return True
