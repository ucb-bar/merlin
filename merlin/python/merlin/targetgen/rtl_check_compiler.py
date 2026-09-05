"""Compile RTL-derived facts + a capsule's declared shape into **FileCheck** assertions.

This is the "compile the RTL to the checks" step: instead of hand-coded Python predicates, the RTL facts
(:mod:`merlin.targetgen.rtl.circt_introspect`) and the capsule's *declared* problem statement are turned
into FileCheck directives that LLVM's ``FileCheck`` runs against the agent's emitted artifacts. The
arithmetic (tile count = ⌈M/DIM⌉·⌈N/DIM⌉, legal-opcode set, required ops) is evaluated HERE, at
check-generation time, so concrete RTL-grounded literals are baked into the CHECK lines; FileCheck does
the matching and its diagnostic (the offending IR line) is the feedback.

The check family is ENDPOINT-derived (never a target-name test): a RoCC command-ISA target
(``endpoint_kind == inline_asm_insn``) is checked over its decoded RoCC trace; a self-hosted-ISA target
(``external_backend``) over its emitted kernel instruction stream.

* ``trace`` — over a canonical text rendering of the decoded RoCC trace: ``MVOUT_COUNT`` exact tile
  coverage, ``ILLEGAL_FUNCT_COUNT 0`` (every funct ∈ the RTL-derived legal set), ``COMPUTE_PRESENT`` for
  compute ops, and the ABI opcode/funct3 from the RTL facts.
* ``kernel`` — over a rendered decode of a self-hosted target's emitted instruction stream: opcode
  LEGALITY (``ILLEGAL_OPCODE_COUNT 0``) plus class coverage / mesh-tiling count / field-sanity / order,
  all keyed by the capsule's DERIVED ``expected.instruction_classes`` (not target data).

Both families assert over the target's actual EMITTED commands/instructions — the decoded ground truth —
so no check depends on the agent's dialect op mnemonics (those are invented per generated OOT dialect and
carry no derivation source; a text FileCheck over them would be both target-overfit and false-failing).

NON-OVERFIT: every literal comes from RTL facts + declared shape + ISA rules — never a golden trace or a
per-capsule expected count. For shapes where an exact tile count cannot be derived generally (multi-matmul
resident reuse, conv) the check degrades to a lower bound or is omitted, honestly.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import rtl_checks as RC  # reuse _declared_output_shape / _mesh / _declared_op

RENDER_SCHEMA = "rtl-trace-render/v0"
_COMPUTE_OPS = {"matmul", "resident_reuse", "conv2d", "conv", "matmul_resident"}


def _facts_abi(facts: dict) -> tuple[str, str] | None:
    """The RoCC ABI (custom opcode, funct3) from the DERIVED decode facts, or None when the target's
    facts carry no RoCC command interface — never a hardcoded gemmini default (a non-RoCC target then
    simply omits the ABI assertion instead of being checked against 0x7b)."""
    for i in (facts.get("interfaces") or []):
        if i.get("name") == "funct_decode_table" and i.get("custom_opcode") is not None:
            return (f"0x{i['custom_opcode']:x}", f"0x{i['funct3']:x}")
    return None


def _endpoint_kind_for(target: str, facts_rec: dict) -> str | None:
    """The target's DERIVED codegen endpoint_kind, resolved the SAME way the capability layer derives it,
    so the FileCheck family routing matches the grader. Order: the residual+facts deriver
    (``manifest_for`` — works for every target that ships a residual, including the ones with no committed
    ``target_contract.yaml``), then the committed contract (``load_capability_manifest``), then straight
    from this run's facts record (``_endpoint_from_facts`` over the funct7 width / self-hosted-ISA signal).
    None only when nothing grounds it (caller then treats it as non-RoCC, honestly)."""
    from .capability_manifests import _endpoint_from_facts, _facts_body, manifest_for
    try:
        ek = manifest_for(target).get("endpoint_kind")
        if ek:
            return ek
    except Exception:  # noqa: BLE001 — no residual/derivation for this target
        pass
    try:
        from .target_experiment import load_capability_manifest
        return load_capability_manifest(target).endpoint_kind
    except Exception:  # noqa: BLE001 — no committed contract either
        return _endpoint_from_facts(_facts_body(facts_rec))


def _is_rocc_target(target: str, facts_rec: dict) -> bool:
    """Is this a RoCC command-ISA target (the TRACE FileCheck applies) vs a self-hosted-ISA
    (external_backend, the KERNEL FileCheck) / other target? DERIVED, never a target-name test: routes on
    the DERIVED ``endpoint_kind`` (``inline_asm_insn`` == RoCC). We do NOT key on the presence of a
    ``funct_decode_table`` — the mlc icmp-fanout extractor synthesises one for ANY decoder (a self-hosted
    ISA gets a table too), so that would false-positive; the endpoint deriver instead reads the funct7
    WIDTH (<= 0x7f -> RoCC) / the self-hosted-ISA signal. When nothing grounds an endpoint, it is not
    RoCC (the KERNEL family / honest non-routing)."""
    return _endpoint_kind_for(target, facts_rec) == "inline_asm_insn"


def _facts_to_rc(facts_rec: dict) -> dict:
    """Translate a circt_introspect facts record into the flat dict rtl_checks helpers expect
    (mesh + scratchpad capacity)."""
    facts = facts_rec.get("facts", facts_rec)
    mesh = next((a for a in facts.get("arrays", []) if a["name"] == "mesh"), {})
    sp = next((m for m in facts.get("memories", []) if m["name"] == "scratchpad"), {})
    out = {}
    if mesh:
        out["mesh"] = [mesh["rows"], mesh["cols"]]
    if sp.get("bytes"):
        out["scratchpad_bytes"] = sp["bytes"]
    layouts = next((i.get("bundles", {}) for i in facts.get("interfaces", [])
                    if i.get("name") == "register_bundle_layouts"), {})
    mvout_layout = layouts.get("ConfigMvoutRs1") if isinstance(layouts, dict) else None
    if isinstance(mvout_layout, dict):
        out["config_mvout_fields"] = sorted((mvout_layout.get("fields") or {}).keys())
    build_features = next((i for i in facts.get("interfaces", [])
                           if i.get("name") == "elaborated_rtl_features"), {})
    max_pool = (build_features.get("features") or {}).get("max_pool")
    if isinstance(max_pool, bool) and build_features.get("status") == "derived":
        out["max_pool_supported"] = max_pool
    return out


def compile_trace_checks(facts_rec: dict, capsule: dict, prefix: str = "TRACE") -> str | None:
    """FileCheck lines over the rendered RoCC trace (see rtl_check_runner.render_trace)."""
    op = RC._declared_op(capsule)
    if op is None:
        return None
    rc_facts = _facts_to_rc(facts_rec)
    mesh = rc_facts.get("mesh")                             # DERIVED mesh only (None -> not grounded)
    abi = _facts_abi(facts_rec.get("facts", facts_rec))
    mesh_str = f"{mesh[0]}x{mesh[1]}" if mesh else "unknown"
    L = [f"// RTL-derived trace checks (op={op}, mesh={mesh_str}) — generated, do not edit"]
    if abi is not None:                                      # ABI only when the RTL facts carry it (derived)
        L.append(f"// {prefix}-DAG: ABI custom={abi[0]} funct3={abi[1]}")
    # {{$}} anchors end-of-line so "..._COUNT 1" does NOT substring-match "..._COUNT 16".
    # ILLEGAL_FUNCT_COUNT 0 only when the RTL facts ground a legal-funct set — else legality is not
    # derivable and the check is OMITTED (the runner renders '-'), fail-closed rather than asserting
    # against a baked default set.
    dt = _decode_table(facts_rec) or {}
    if dt.get("legal_funct"):
        L.append(f"// {prefix}-DAG: ILLEGAL_FUNCT_COUNT 0{{{{$}}}}")   # every emitted funct ∈ RTL legal set
    shape = RC._declared_output_shape(capsule)
    if op in ("matmul", "matmul_resident") and shape is not None:
        # exact tile count needs the DERIVED mesh DIM — fail-closed: emit COMPUTE_PRESENT always, but the
        # exact MVOUT_COUNT only when the mesh is grounded (else a mesh-less RTL would be checked against a
        # silent DIM=16 default and false-fail/false-pass).
        if mesh:
            expected = RC.expected_mvout_count(capsule, rc_facts)
            if expected is not None:
                tiles, _geometry = expected
                L.append(f"// {prefix}-DAG: MVOUT_COUNT {tiles}{{{{$}}}}")  # exact RTL store geometry
        L.append(f"// {prefix}-DAG: COMPUTE_PRESENT yes")
    elif op == "resident_reuse":
        # multi-matmul lower bound is not FileCheck-exact; leave the count to rtl_checks.screen().
        L.append(f"// {prefix}-DAG: COMPUTE_PRESENT yes")
    elif op in ("conv2d", "conv"):
        L.append(f"// {prefix}-DAG: COMPUTE_PRESENT yes")      # conv tile geometry not derived generally
    elif op == "movement":
        L.append(f"// {prefix}-DAG: MVIN_PRESENT yes")
    return "\n".join(L) + "\n"


def _decode_table(facts_rec: dict) -> dict | None:
    """The mlc-derived RoCC decode interface (funct_decode_table), if the facts carry one. Its presence
    is the DERIVED signal that this target speaks the RoCC command ISA the dialect/trace checks assume —
    a SIMT/program-MMIO target has none, and those checks are dropped rather than emitted meaninglessly."""
    facts = facts_rec.get("facts", facts_rec)
    return next((i for i in (facts.get("interfaces") or []) if i.get("name") == "funct_decode_table"), None)


def _provenance(facts_rec: dict, capsule: dict, target: str) -> dict[str, Any]:
    """Per-check-family audit: the derivation source and whether it is genuinely DERIVED (vs a hand
    grouping / a fallback). This is how we answer "did we hand-pick this?" — every emitted check names
    its source, and a family with no resolvable source is reported unavailable, never guessed."""
    from .rtl import mlc_bridge
    dt = _decode_table(facts_rec) or {}
    facts = facts_rec.get("facts", facts_rec)
    has_mesh = any(a.get("name") == "mesh" for a in facts.get("arrays", []))
    roles = mlc_bridge.semantic_roles(target)
    return {
        # legality + ABI + DIM come straight from the mlc decoder/geometry facts — derived when present.
        "isa_legality": {"source": dt.get("method", "funct_decode_table"),
                         "derived": bool(dt.get("legal_funct")), "evidence": dt.get("evidence")},
        "abi_encoding": {"source": "funct_decode_table.custom_opcode/funct3",
                         "derived": dt.get("custom_opcode") is not None},
        "tile_coverage": {"source": "discovered mesh DIM + declared output shape", "derived": has_mesh},
        # the opcode->ROLE grouping is the one still-ungrounded axis: derived ONLY once the mlc effect
        # probe has populated a roles cache; until then the dialect/trace checks use the hand funct
        # classes (rocc_decode), which we flag honestly rather than present as rigorous.
        "semantic_roles": {"source": roles["source"] or "rocc_decode(hand funct classes)",
                           "derived": roles["derived"], "reason": roles["reason"],
                           "n_roles": len(roles["roles"])},
    }


def compile_kernel_checks(capsule: dict, prefix: str = "KERNEL",
                          facts_rec: dict | None = None, target: str | None = None) -> str | None:
    """FileCheck lines over a rendered decode of the emitted self-hosted-ISA kernel (external_backend,
    e.g. atlas ``kernel.S`` → its `.word`/`.insn` instruction stream). The RTL-grounded insight — the kind
    you would otherwise pay a Verilog run for and that spike/npu_model's functional output never gives —
    is ISA LEGALITY: every emitted instruction's opcode must be one the target's decoder actually accepts
    (the legal-opcode set discovered from the RTL / ISA definition). This catches a fabricated or
    mis-encoded ISA (opcodes the hardware would reject) statically, before the cosim.

    The check carries NO target literals: it asserts ``ILLEGAL_OPCODE_COUNT 0`` over the rendered decode;
    the legal set + the decode itself are computed at run time in :func:`rtl_check_runner.render_kernel_decode`
    from the DERIVED taxonomy. Returns None if the capsule declares no operation."""
    op = RC._declared_op(capsule)
    if op is None:
        return None
    required = list(((capsule.get("expected") or {}).get("instruction_classes") or []))
    # legality (ILLEGAL_OPCODE_COUNT 0) is assertable only when the runner can DETERMINE legality — a
    # taxonomy (per-op decode signatures) OR a discovered legal-opcode set resolves. With neither, the
    # render emits '-' and asserting "0" would be a vacuous pass, so we OMIT it (fail-closed).
    tax = {}
    if target:
        try:
            from . import isa_taxonomy as IT
            tax = IT.taxonomy_for_target(target)
        except Exception:  # noqa: BLE001 — taxonomy unavailable -> skip these two, keep coverage/order
            tax = {}
    legality_determinable = bool(tax) or bool((_decode_table(facts_rec or {}) or {}).get("legal_funct"))
    L = [f"// RTL-derived kernel checks (op={op}) — legality + coverage + tiling + order + field-sanity",
         f"// {prefix}-DAG: EMPTY_KERNEL no"]                   # the kernel must actually emit instructions
    if legality_determinable:
        L.insert(1, f"// {prefix}-DAG: ILLEGAL_OPCODE_COUNT 0{{{{$}}}}")   # every emitted opcode ∈ legal set
    # (1) CLASS COVERAGE: every instruction class the capsule requires (DERIVED per-target in
    # expected.instruction_classes) must actually be EMITTED — the render classifies each word via the
    # ISA-def decode signatures, so a matmul that emitted VADD instead of the MXU matmul fails here
    # (legality alone passes it). Literals are the capsule's own derived class names, not target data.
    for cls in required:
        L.append(f"// {prefix}-DAG: CLASS_PRESENT {cls}{{{{$}}}}")

    if tax:
        from . import isa_taxonomy as IT
        roles = IT.role_classes(tax)
        compute, memory = roles.get("compute"), roles.get("memory")
        # tiling: the compute (matmul) class must appear exactly ceil(M/DIM)*ceil(N/DIM) times — the tile
        # count the discovered mesh geometry + the declared output shape imply. Skipped unless both resolve.
        shape = RC._declared_output_shape(capsule)
        mesh = _facts_to_rc(facts_rec or {}).get("mesh")   # DERIVED mesh only — fail-closed, no DIM=16 default
        # An EXACT tile count is only sound for a single-matmul op. A resident_reuse capsule issues several
        # matmuls against a resident weight, so ceil(M/DIM)*ceil(N/DIM) understates the true compute count
        # and false-rejects a correct multi-matmul kernel (parity with compile_trace_checks, which already
        # degrades resident_reuse to COMPUTE_PRESENT). Emit only class-PRESENCE for it; leave the count to
        # rtl_checks.screen()'s lower-bound check.
        if compute and compute in required and shape and mesh and op != "resident_reuse":
            tiles = math.ceil(shape[0] / mesh[0]) * math.ceil(shape[1] / mesh[1])
            L.append(f"// {prefix}-DAG: CLASS_COUNT {compute} {tiles}{{{{$}}}}")
        # field-sanity: a memory (load/store) instruction with an all-zero operand payload addresses DRAM 0
        # — the "TensorBaseOffset encodes address 0" bug. Require zero such instructions.
        if memory and memory in required:
            L.append(f"// {prefix}-DAG: CLASS_ZEROOPS {memory} 0{{{{$}}}}")

    # (3) ORDER: the required classes must first appear in their DERIVED canonical order (AW6 emits the
    # sequence load -> weight-push -> matmul -> pop). An ordered CHECK (own prefix) over the per-INSTR
    # class= lines enforces first-occurrence order without constraining the interleaving.
    order = "\n".join(f"// KORDER: class={cls}" for cls in required)
    return "\n".join(L) + "\n" + (order + "\n" if order else "")


def compile_checks(facts_rec: dict, capsule: dict, target: str) -> dict[str, Any]:
    """Compile the check files for a capsule + a per-family PROVENANCE audit, ENDPOINT-aware and fully
    derived. A RoCC command-ISA target (endpoint ``inline_asm_insn``, e.g. gemmini) gets the trace
    FileCheck over its decoded RoCC stream; a self-hosted-ISA target (``external_backend``, e.g. atlas)
    gets the kernel opcode-legality FileCheck over its emitted `.word` stream. The RoCC vs self-hosted
    decision is the DERIVED ``endpoint_kind`` (never funct_decode_table presence — the mlc icmp-fanout
    extractor synthesises a table for a self-hosted decoder too, so that would mis-route). Both families
    check the target's actual emitted commands/instructions; neither depends on the agent's (per-run,
    un-derivable) dialect op mnemonics. We emit every check we can ground for the target's endpoint and
    drop the rest, never guessing."""
    is_rocc = _is_rocc_target(target, facts_rec)
    return {
        "schema": "rtl_checks_filecheck/v0",
        "capsule": capsule.get("name"),
        "target": target,
        "trace": compile_trace_checks(facts_rec, capsule) if is_rocc else None,
        "kernel": None if is_rocc else compile_kernel_checks(capsule, facts_rec=facts_rec, target=target),
        "provenance": _provenance(facts_rec, capsule, target),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import yaml
    ap = argparse.ArgumentParser(description="Compile RTL facts + capsule -> FileCheck assertion files.")
    ap.add_argument("capsule", help="path to capsule.yaml")
    from .rtl.facts import load_facts
    ap.add_argument("--target", required=True,
                    help="target whose RTL facts + check family to compile")
    ap.add_argument("--facts", default=None, help="facts.json (default: regenerate the target from RTL)")
    a = ap.parse_args(argv)
    facts = json.loads(Path(a.facts).read_text()) if a.facts else load_facts(a.target)
    capsule = yaml.safe_load(Path(a.capsule).read_text())
    cc = compile_checks(facts, capsule, a.target)
    print(f"# capsule={cc['capsule']}\n# --- trace ---\n{cc['trace']}\n# --- kernel ---\n{cc['kernel']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
