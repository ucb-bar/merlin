"""Robust, non-regex decoders for the kernel-mining pipeline.

These replace the fragile regex-over-C / regex-over-objdump-text extraction. The asm decoder
(``asm_mc``) is the authoritative substrate every framework lowers to; the source decoder
(``clang_ast``, added later) is a typed cross-check.
"""


def decode_for_endpoint(raws, target: str, endpoint):
    """Decode one disassembled stream through whichever decoder THIS endpoint's encoding needs.

    ONE dispatcher, because there were two. ``asm_audit`` grew a full version of this and the mining
    pipeline grew a shorter one, and they drifted in four ways that all produced the same symptom --
    an expert kernel lifting to nothing, reported as "decoded nothing of this endpoint", which reads
    as an empty corpus rather than as a decoder that was never given what it needed:

    * ``roles_of`` was not passed, so every instruction decoded with NO role and the caller's
      "nothing carries a role" guard returned None for a stream the audit read fine;
    * the ``stream_width`` override was missing, so a target whose object words are narrower than its
      internal instruction width declined every architectural word;
    * the intrinsics ``custom_table`` was missing, so custom-space words kept the coarse SPACE name
      instead of resolving to the operation the target's own header names;
    * ``cede_funct7_in`` was missing, so one engine's instructions were counted as another's gap.

    Returns ``[]`` when the endpoint's encoding source has no object decoder (a text-ISA endpoint is
    assembled, not disassembled) -- distinguishable from "decoded and found nothing" by being empty
    rather than role-less.
    """
    from merlin.kernels import endpoints as _ep

    block = ((_ep._spec().get("endpoints") or {}).get(getattr(endpoint, "name", "")) or {})
    enc_block = block.get("encoding") or {}
    kind = str(enc_block.get("source") or "")

    if kind == "rtl_facts":
        from merlin.kernels.decode import rocc as _rocc
        return _rocc.decode_stream(raws, _rocc.funct_table_for(target), endpoint.roles_of)

    if kind == "isa_encoding":
        from merlin.kernels.decode import derived_isa as _isa
        from merlin.kernels.decode import insn_header as _ih
        enc = dict(_isa.encoding_for(target))
        width = enc_block.get("stream_width")
        if width:
            enc["inst_width"] = int(width)
        custom, _problems = _ih.table_for(target, endpoint)
        cede = tuple(
            str(other.get("opcode_space") or "")
            for e in _ep.endpoints_for(target)
            if e.name != getattr(endpoint, "name", "")
            for other in [((_ep._spec()["endpoints"].get(e.name) or {}).get("encoding") or {})]
            if str(other.get("discriminator") or "") == "funct7" and other.get("opcode_space"))
        return _isa.decode_stream(raws, enc, enc_block.get("spaces") or (),
                                  endpoint.roles_of, custom, cede)

    if kind == "mnemonic_grammar":
        # The vocabulary is DECLARED from the ISA grammar rather than derived from a decode table.
        # Omitting this branch is how the reference lane backend broke: its expert object lifted to
        # None and the mining run reported "divergences=0" -- agreement with an expert it never read.
        from merlin.kernels.decode import grammar as _gram
        return _gram.decode_stream(raws, endpoint)

    if kind == "funct_header":
        # A RoCC-shaped endpoint whose funct table comes from the target's own C ISA header rather
        # than from RTL. Same decoder; only the provenance of the codes differs.
        from pathlib import Path as _Path

        from merlin.common import provenance as _prov
        from merlin.kernels import asm_audit as _aa
        from merlin.kernels.decode import rocc as _rocc
        from merlin.targetgen.rtl.circt_introspect import _functs_from_headers
        try:
            root = _Path(_prov.verify(str(enc_block.get("pin"))).observed.path)
            by_code = _functs_from_headers([root / str(enc_block.get("path"))])
        except (KeyError, OSError, ValueError):
            return []          # header unresolved: decode nothing rather than guess a table
        opcode = (_aa._derived_opcodes(target) or {}).get(str(enc_block.get("opcode_space") or ""))
        if opcode is None:
            return []          # refuse to guess an opcode value that is not in the derived table
        table = {"custom_opcode": opcode, "legal_funct": sorted(by_code),
                 "names": {str(k): v for k, v in by_code.items()}}
        decoded = _rocc.decode_stream(raws, table, endpoint.roles_of)
        if str(enc_block.get("discriminator") or "") == "funct7":
            # Shares its opcode space with the target's SIMT surface, told apart by field: a command
            # carries its operation in funct7, an intrinsic has funct7 == 0. Claim only the
            # unambiguous half rather than resolving the tie by preference.
            for d in decoded:
                if getattr(d, "from_endpoint", False) and not d.fields.get("funct"):
                    object.__setattr__(d, "from_endpoint", False)
                    object.__setattr__(d, "roles", ())
        return decoded

    if kind == "matrix_units":
        from merlin.kernels import asm_audit as _aa
        from merlin.kernels.decode import opu as _opu
        encodings, _why = _aa._matrix_encodings(target, block)
        if not encodings:
            return []
        decoded = _opu.decode_stream(raws, encodings, endpoint.roles_of)
        for d in decoded:   # this decoder spells it `from_extension`; consumers read `from_endpoint`
            object.__setattr__(d, "from_endpoint", d.from_extension)
        return decoded

    return []


def disasm_settings(target: str, endpoint) -> dict:
    """The triple/mattr the ENDPOINT declares, for disassembling its objects.

    A probe that does not pin these reports the tool's ignorance as the corpus's nature: the same
    omission once put radiance's unknown-word rate at 76% where pinning ``--mattr`` put it at 15%.
    """
    from merlin.kernels import endpoints as _ep

    enc = (((_ep._spec().get("endpoints") or {}).get(getattr(endpoint, "name", "")) or {})
           .get("encoding") or {})
    return {"triple": str(enc.get("disasm_triple") or "riscv64"),
            "mattr": enc.get("disasm_mattr")}
