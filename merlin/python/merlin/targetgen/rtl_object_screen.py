"""Static RTL/ISA screen of a target's emitted MACHINE CODE — the check family for a compiler that emits
an MLIR lowering rather than hand-authored instruction words.

Why this exists
---------------
:mod:`rtl_check_runner` had exactly two check families, and both key on an artifact the agent hand-wrote:
a decoded RoCC command trace (``generated/instruction_trace.json``) or a decoded self-hosted kernel
assembly (``generated/kernel.S``). A target whose codegen endpoint is an LLVM-dialect MLIR module writes
NEITHER — its machine code only exists after a real toolchain compiles the lowering — so ``screen_run``
returned ``None`` for every capsule and the whole advisory came out empty. Measured: 18 consecutive rounds
across three repeats reported ``rtl_checks: []`` with no error beside it, and the arm was read as working.

This family closes that hole by screening the words the target's own emit path actually produced, recorded
at :data:`WORDS_ARTIFACT` — the same convention the other two families use (core fixes the artifact NAME;
each target's emit path is what produces it, because only the producer knows which object is final).

What it can honestly assert
---------------------------
Everything below is derived from the target's own facts, and every check is DROPPED — visibly, with a
reason — when its derivation is unavailable. Nothing here is target-specific; each check applies to the
CLASS of targets whose facts evidence the relevant structure.

* ``DECODE_ILLEGAL`` — a word whose opcode field is outside the derived ``opcode_table``. Note this is
  **attestation, not discovery** for any target whose emit path already lints with the same derived
  disassembler; the render says so, so a reader never mistakes 0 for a finding.
* ``AMBIGUOUS_DECODE`` / ``UNDEFINED_ADDRESS_SPACE`` — from :mod:`isa_lint`, over the same words.
* ``SIMT_CONTROL_COUNT`` + ``SIMT_IDENTITY_READS`` — how many emitted words are the target's own
  warp-control ops (from the RTL-derived ``runtime_abi.sfu_ops``, intersected with the base ISA's custom
  window) and how many read its own lane/warp identity registers (the ``special_csrs`` whose provenance
  cites the target's RTL rather than the base-ISA spec). Zero of BOTH means the kernel is written without
  reference to the machine's parallelism and runs on one of the streams ``facts.simt`` counts. Applies to
  every SIMT/GPU-class target; dropped entirely for a target whose facts carry no SIMT geometry.
* ``SPACE_COUNT <space>`` — the address-space selector value each memory access carries, from the derived
  ``address_spaces`` + ``address_space_field``. A kernel that only ever selects the default space never
  touches the target's other memories. Dropped for a single-flat-address-space target.
* ``OPCODE_COUNT`` / ``OPCODE_PRESENT`` — the decoded opcode histogram, so a capsule's declared
  ``expected.instruction_classes`` can be asserted the way the ``kernel.S`` family already asserts them.

Deliberately NOT asserted, and why (each of these would be a confidently-wrong advisory):

* "a contraction capsule must contain FP compute" — on a target that dispatches its contraction to a
  co-processor through a command buffer, the instruction stream legitimately carries none, so the check
  would fire on correct kernels. The FP histogram is REPORTED; no assertion rides it.
* "the kernel should have used fused multiply-add" — an optimisation claim, not a legality one, and
  nothing in the facts grounds a threshold.
* a static instruction-count lower bound from the declared extent — the loop trip count is not statically
  available, so any bound would be a guess.

No regex, no target-name literal, no golden.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import isa_disasm as D
from . import isa_lint as LINT
from . import isa_transcode as TX
from .isa_model import IsaModel

#: Convention artifact under a capsule run's ``generated/``: the machine-code word stream the target's
#: emit path produced, recorded by that emit path. Parallel to ``kernel.S`` (self-hosted assembly) and
#: ``instruction_trace.json`` (decoded RoCC commands) — one artifact name per check family, fixed by core.
WORDS_ARTIFACT = "emitted_words.json"

WORDS_SCHEMA = "emitted-words/v0"
RENDER_SCHEMA = "rtl-object-render/v0"


# --------------------------------------------------------------------------- the artifact
def write_words(generated_dir: str | Path, words: list[int], *, inst_width: int, source: str,
                symbol: str | None = None, lint_enforced: bool = False) -> Path:
    """Record an emitted word stream for the static screen. Called by a TARGET's emit path, which is the
    only party that knows which of the objects it produced is the final machine code.

    ``source`` names that object (for a human reading the artifact); ``symbol`` is the kernel entry symbol
    when known. ``lint_enforced`` must be True when the emit path itself already rejects an undecodable
    word — the screen then reports its legality result as attested rather than discovered, so a clean
    result is not read as a finding it never could have been."""
    p = Path(generated_dir) / WORDS_ARTIFACT
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"schema": WORDS_SCHEMA, "inst_width": int(inst_width), "source": str(source),
                             "symbol": symbol, "lint_enforced": bool(lint_enforced),
                             "n": len(words), "words": [int(w) for w in words]}, indent=1))
    return p


def load_words(generated_dir: str | Path) -> dict | None:
    """The recorded word stream, or None when this run's emit path recorded none (the caller then reports
    the family as not applicable — never as clean)."""
    p = Path(generated_dir) / WORDS_ARTIFACT
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text())
    except ValueError:
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("words"), list):
        return None
    return doc


# --------------------------------------------------------------------------- derivations
def simt_control_signatures(model: IsaModel) -> dict[tuple[int, int], str]:
    """``{(opcode, funct3): op name}`` for the target's own warp/thread-control ops.

    Both halves are derived. ``runtime_abi.sfu_ops`` is the RTL-sourced dispatch table (mlc's runtime_abi
    pass reads it out of the core's own sources), and it carries BOTH the control ops and the CSR-access
    ops, so the control ones are selected by their opcode falling in the base ISA's custom-extension
    window — a property of the base ISA, compared as data against this target's derived opcode table.
    Empty when the target derives no ``sfu_ops`` (the check is then dropped, not defaulted)."""
    groups = TX.base_isa_opcode_groups(model)
    window = groups.get("custom_extension")
    if not window:
        return {}
    out: dict[tuple[int, int], str] = {}
    for name, ent in ((model.runtime_abi or {}).get("sfu_ops") or {}).items():
        try:
            opcode, funct3 = int(ent["opcode"]), int(ent["funct3"])
        except (KeyError, TypeError, ValueError):
            continue
        if opcode in window:
            out[(opcode, funct3)] = str(name)
    return out


def identity_csr_reads(words: list[int], model: IsaModel) -> tuple[dict[str, int] | None, str]:
    """``({csr name: times read}, "")`` for the target's OWN (non-architectural) status registers — the ones
    that tell a kernel which lane/warp/core it is and how many there are — or ``(None, reason)``.

    Every part is DERIVED. ``runtime_abi.provenance`` records where each special CSR came from, and the
    target's own registers cite its RTL sources (``rtl:…``) while the architectural ones cite the base-ISA
    spec — so "identity CSR" needs no list. The CSR-access instructions are the ``sfu_ops`` entries sitting
    in the base ISA's system-opcode window. The register number is read from the field the target's own
    TRANSCODER writes an immediate into, which is the only field that can be right: this stream was
    produced by that transcoder, and asking any other field is asking where the value is not.

    That last point is not hypothetical. Reading the layout's own ``csrimm`` field instead looks obviously
    correct and is silently dead: on a measured target that field is 8 bits wide while every one of its 15
    own CSR numbers exceeds 4000, so the count comes back 0 for every kernel and reads as a finding
    ("nothing reads its warp id") when in fact nothing could ever have matched. Hence the explicit width
    check below, and hence a REASON on every refusal rather than a bare ``None``."""
    abi = model.runtime_abi or {}
    prov, csrs = abi.get("provenance") or {}, abi.get("special_csrs") or {}
    op_span, f3_span = _field(model, "opcode"), _field(model, "f3")
    own = {int(num): name for name, num in csrs.items()
           if str(prov.get(f"csr.{name}", "")).startswith("rtl:")}
    if not own:
        return None, ("runtime_abi records no CSR as coming from this target's own RTL sources, so its "
                      "identity registers cannot be told apart from the architectural ones")
    sys_window = (TX.base_isa_opcode_groups(model)).get("system")
    csr_ops = {(int(e["opcode"]), int(e["funct3"]))
               for e in (abi.get("sfu_ops") or {}).values()
               if isinstance(e, dict) and "opcode" in e and "funct3" in e
               and sys_window and int(e["opcode"]) in sys_window}
    if not csr_ops:
        return None, "no derived sfu_op sits in the base ISA's system-opcode window (no CSR-access form)"
    if not (op_span and f3_span):
        return None, "the derived field layout has no opcode+funct3 pair to identify a CSR access"
    try:
        tc = TX.FixedFormatTranscoder(model)
        span, bits = _field(model, tc.imm_field), int(tc.imm_bits)
    except Exception as e:  # noqa: BLE001 — no transcoder for this model
        return None, f"the target's immediate field is not derivable ({type(e).__name__}: {e})"
    if not span:
        return None, f"the derived layout has no {tc.imm_field!r} field to read a CSR number from"
    if max(own) >= (1 << bits):
        return None, (f"the target's immediate field {tc.imm_field!r} is {bits} bits, too narrow for its "
                      f"own CSR numbers (largest {max(own)}) — a count from it would be 0 by construction")
    hits: dict[str, int] = {}
    for w in words:
        if (_sel(w, op_span), _sel(w, f3_span)) in csr_ops:
            name = own.get(_sel(w, span))
            if name:
                hits[name] = hits.get(name, 0) + 1
    return hits, ""


def simt_instruction_streams(facts_rec: dict) -> int | None:
    """How many INDEPENDENT instruction streams the machine has, from its SIMT geometry facts
    (``warps_per_core`` x ``cores``). None when the facts carry no SIMT block — a non-SIMT target, whose
    warp-control check is dropped rather than answered with 1."""
    facts = facts_rec.get("facts", facts_rec)
    simt = facts.get("simt")
    if not isinstance(simt, dict):
        return None
    try:
        warps, cores = int(simt["warps_per_core"]), int(simt["cores"])
    except (KeyError, TypeError, ValueError):
        return None
    return warps * cores if warps > 0 and cores > 0 else None


def _field(model: IsaModel, name: str):
    return model.field_layout.get(name)


def _sel(word: int, span: tuple[int, int]) -> int:
    hi, lo = span
    return (word >> lo) & ((1 << (hi - lo + 1)) - 1)


# --------------------------------------------------------------------------- the screen
def screen(words: list[int], model: IsaModel, facts_rec: dict, capsule: dict | None = None, *,
           lint_enforced: bool = False) -> dict[str, Any]:
    """Screen an emitted word stream. Returns
    ``{verdict, checks: [...], metrics: {...}, grounded: {...}, dropped: {...}}``.

    ``checks`` entries carry the same shape the numeric screen produces (``id``/``status``/``severity``/
    ``message``/``expected``/``got``/``fix_hint``) so an existing consumer redacts and renders them
    unchanged. ``grounded``/``dropped`` are the honesty ledger: every check that COULD not run is named
    with the derivation it needed, so an empty ``checks`` list is never mistaken for a clean bill."""
    checks: list[dict] = []
    metrics: dict[str, Any] = {}
    grounded: dict[str, str] = {}
    dropped: dict[str, str] = {}

    if not model.is_fixed_format():
        dropped["all"] = ("the target's ISA model is not fixed-format (no field layout + opcode table), so "
                          "an emitted word stream cannot be decoded from the model alone")
        return {"verdict": "unknown", "checks": [], "metrics": {"n_words": len(words)},
                "grounded": {}, "dropped": dropped}

    recs = D.disassemble(model, words)
    n = len(recs)
    metrics["n_words"] = n
    hist: dict[str, int] = {}
    for r in recs:
        m = r.get("mnemonic")
        if m and not r.get("illegal"):
            hist[m] = hist.get(m, 0) + 1
    metrics["opcode_histogram"] = dict(sorted(hist.items()))

    # --- (1) decode legality -----------------------------------------------------------------------
    n_illegal = sum(1 for r in recs if r.get("illegal"))
    metrics["n_illegal"] = n_illegal
    metrics["legality_attested_at_emit"] = bool(lint_enforced)
    grounded["decode_legality"] = ("the target's derived opcode_table"
                                   + (" (also enforced by its emit path, so a clean result here is an "
                                      "attestation, not a discovery)" if lint_enforced else ""))
    checks.append(_check(
        "decode_legality", n_illegal == 0, "error",
        f"{n_illegal} of {n} emitted word(s) carry an opcode this target's decoder does not define",
        expected=0, got=n_illegal,
        fix_hint="assemble with the derived encoder (isa_tools asm) instead of hand-packing words"))

    lint_findings = LINT.lint(model, words) if n else []
    for rule in ("ambiguous_decode", "undefined_address_space"):
        hits = [f for f in lint_findings if f.get("rule") == rule]
        metrics[f"n_{rule}"] = len(hits)
        if hits:
            checks.append(_check(rule, False, hits[0].get("severity", "warning"),
                                 f"{len(hits)} instruction(s): {hits[0].get('detail')}",
                                 expected=0, got=len(hits)))

    # --- (2) SIMT control: does the kernel use more than one of the machine's instruction streams? ---
    sigs = simt_control_signatures(model)
    streams = simt_instruction_streams(facts_rec)
    op_span, f3_span = _field(model, "opcode"), _field(model, "f3")
    if not sigs:
        dropped["simt_control"] = ("the target derives no runtime_abi.sfu_ops in the base ISA's "
                                   "custom-extension window, so its warp-control ops are unknown")
    elif streams is None:
        dropped["simt_control"] = "the target's RTL facts carry no SIMT geometry (warps_per_core x cores)"
    elif not (op_span and f3_span):
        dropped["simt_control"] = ("the derived field layout has no opcode+funct3 pair to match a control "
                                   "op against")
    else:
        found: dict[str, int] = {}
        for w in words:
            name = sigs.get((_sel(w, op_span), _sel(w, f3_span)))
            if name:
                found[name] = found.get(name, 0) + 1
        n_ctl = sum(found.values())
        metrics["simt_control_ops"] = dict(sorted(found.items()))
        metrics["simt_control_count"] = n_ctl
        metrics["simt_instruction_streams"] = streams
        # The companion evidence: a kernel can be parallel by READING which lane/warp it is rather than by
        # spawning. Folded into this one finding instead of raised as a second, because two near-identical
        # warnings crowd out the ones that differ — but reported separately in the metrics, and named in the
        # message, because "you never read warp_id" is what makes the finding actionable.
        ident, why_no_ident = identity_csr_reads(words, model)
        if ident is None:
            dropped["simt_identity"] = why_no_ident
        else:
            metrics["identity_csr_reads"] = dict(sorted(ident.items()))
            metrics["identity_csr_read_count"] = sum(ident.values())
        n_ident = sum((ident or {}).values())
        # Name the TARGET'S OWN registers in the hint, not every special CSR: the architectural ones
        # (mstatus, marchid, …) tell a kernel nothing about which stream it is, and listing them sends the
        # reader to the wrong registers. Same provenance test identity_csr_reads uses.
        _abi = model.runtime_abi or {}
        _prov = _abi.get("provenance") or {}
        own_csrs = sorted(n for n in (_abi.get("special_csrs") or {})
                          if str(_prov.get(f"csr.{n}", "")).startswith("rtl:"))
        grounded["simt_control"] = (f"runtime_abi.sfu_ops ({len(sigs)} control ops) + facts.simt "
                                    f"({streams} independent instruction streams)"
                                    + ("" if ident is None else " + runtime_abi own-CSR reads"))
        # ADVISORY, and a warning rather than a reject: a single-stream kernel is correct, just serial.
        # It PASSES on either kind of evidence — control ops or identity reads — because either one means
        # the kernel is written against the machine's parallelism; zero of both means it is not.
        uses = n_ctl > 0 or n_ident > 0 or streams <= 1
        why = f"emits no warp/thread-control instruction ({', '.join(sorted(set(sigs.values())))})"
        if ident is not None:
            why += " and reads none of the target's own lane/warp identity registers"
        checks.append(_check(
            "simt_control", uses, "warning",
            f"the kernel {why}, so it executes on 1 of this machine's {streams} independent instruction "
            f"streams; the other {streams - 1} stay parked",
            expected=">0", got=n_ctl + n_ident,
            fix_hint=("partition the work across warps: read the derived identity CSRs ("
                      + ", ".join(own_csrs[:6]) + (", …" if len(own_csrs) > 6 else "")
                      + ") to find this stream's slice, and emit the target's own spawn/mask/barrier ops")))

    # --- (3) address spaces: which of the target's memories does the kernel actually reach? ----------
    spaces, asf = dict(model.address_spaces or {}), model.address_space_field
    groups = TX.base_isa_opcode_groups(model)
    mem_window = groups.get("memory")
    as_span = _field(model, asf) if asf else None
    if len(spaces) < 2 or not as_span:
        dropped["address_space"] = ("the target declares a single flat address space (or no selector "
                                    "field), so there is no space choice to screen")
    elif not mem_window:
        dropped["address_space"] = "no derived opcode value falls in the base ISA's memory-access window"
    else:
        by_space: dict[str, int] = {n_: 0 for n_ in spaces}
        n_mem = 0
        by_val = {int(v): k for k, v in spaces.items()}
        for w in words:
            if _sel(w, op_span) not in mem_window:
                continue
            n_mem += 1
            nm = by_val.get(_sel(w, as_span))
            if nm:
                by_space[nm] = by_space[nm] + 1
        metrics["memory_accesses"] = n_mem
        metrics["accesses_by_space"] = dict(sorted(by_space.items()))
        grounded["address_space"] = (f"derived address_spaces {sorted(spaces)} on field {asf!r} + the base "
                                     f"ISA's memory-access opcode window")
        # The non-default spaces are the ones a kernel has to ASK for; selector 0 is what it gets by
        # default. Naming them by their derived names keeps this free of target vocabulary.
        non_default = sorted(k for k, v in spaces.items() if int(v) != 0)
        used = sum(by_space.get(k, 0) for k in non_default)
        if non_default and n_mem:
            checks.append(_check(
                "address_space_use", used > 0, "warning",
                f"all {n_mem} memory access(es) select the default address space; the kernel never reaches "
                f"{', '.join(non_default)}",
                expected=">0", got=used,
                fix_hint=f"stage reused operands in {non_default[0]} by setting the derived {asf!r} "
                         f"selector on the access"))

    # --- (4) the kernel must write its result -------------------------------------------------------
    # The one CORRECTNESS claim this family can make without an oracle and without an optimisation
    # argument: a capsule declares an output, and a kernel that emits no memory WRITE cannot have produced
    # one. It catches a lowering whose store was dead-code-eliminated, or an entry symbol whose body never
    # got emitted — both of which otherwise cost a full oracle run to discover.
    # Checked against the corpus before being given `error` severity: across 35 capsule runs that PASSED
    # the oracle, the minimum write count was 1 and no passing run had zero, so this cannot fire on a
    # correct kernel here. It is dropped, not guessed, for a target with no derivable write opcode.
    store_window = groups.get("memory_store")
    if not store_window:
        dropped["result_write"] = "no derived opcode value falls in the base ISA's memory-write window"
    elif not (capsule or {}):
        dropped["result_write"] = "no capsule record, so nothing declares that this kernel owes an output"
    else:
        n_store = sum(1 for w in words if _sel(w, op_span) in store_window)
        metrics["memory_writes"] = n_store
        grounded["result_write"] = "the base ISA's memory-write opcode window over the emitted stream"
        checks.append(_check(
            "result_write", n_store > 0, "error",
            "the kernel emits no memory write, so it cannot have produced the output the capsule declares "
            "— the emitted body is empty, or its store was eliminated",
            expected=">0", got=n_store,
            fix_hint="check that the lowering's result store survives to the emitted object (an unused "
                     "result is legal to delete, so the output pointer must actually be written through)"))

    # --- (5) declared class coverage (only when the capsule declares one) ---------------------------
    required = list(((capsule or {}).get("expected") or {}).get("instruction_classes") or [])
    if required:
        missing = [c for c in required if c not in hist]
        grounded["class_coverage"] = "the capsule's own declared expected.instruction_classes"
        checks.append(_check("class_coverage", not missing, "error",
                             f"the capsule requires instruction class(es) {required} and the emitted "
                             f"stream contains none of {missing}",
                             expected=required, got=sorted(hist)))
    else:
        dropped["class_coverage"] = "this capsule declares no expected.instruction_classes"

    worst = "ok"
    for c in checks:
        if c["status"] == "fail":
            worst = "reject" if c["severity"] == "error" else ("warn" if worst == "ok" else worst)
    return {"verdict": worst, "checks": checks, "metrics": metrics,
            "grounded": grounded, "dropped": dropped}


def _check(cid: str, passed: bool, severity: str, message: str, *, expected=None, got=None,
           fix_hint: str | None = None) -> dict:
    return {"id": cid, "status": "pass" if passed else "fail", "severity": severity,
            "message": None if passed else message, "expected": expected, "got": got,
            "fix_hint": None if passed else fix_hint}


# --------------------------------------------------------------------------- FileCheck rendering
def render(rep: dict) -> str:
    """Canonical text the OBJECT FileCheck lines are matched against — same discipline as
    :func:`rtl_check_runner.render_trace`: one fact per line, and a value that could not be determined
    renders ``-`` rather than a passing 0."""
    m = rep.get("metrics") or {}
    n = m.get("n_words", 0)
    L = [f"# {RENDER_SCHEMA}",
         f"EMPTY_OBJECT {'yes' if not n else 'no'}",
         f"INSTR_COUNT {n}",
         f"DECODE_ILLEGAL {m.get('n_illegal', '-')}",
         f"LEGALITY_ATTESTED_AT_EMIT {'yes' if m.get('legality_attested_at_emit') else 'no'}",
         f"AMBIGUOUS_DECODE {m.get('n_ambiguous_decode', '-')}",
         f"UNDEFINED_ADDRESS_SPACE {m.get('n_undefined_address_space', '-')}",
         f"SIMT_CONTROL_COUNT {m.get('simt_control_count', '-')}",
         f"SIMT_IDENTITY_READS {m.get('identity_csr_read_count', '-')}",
         f"SIMT_STREAMS {m.get('simt_instruction_streams', '-')}",
         f"MEMORY_ACCESSES {m.get('memory_accesses', '-')}",
         f"MEMORY_WRITES {m.get('memory_writes', '-')}"]
    for space, cnt in sorted((m.get("accesses_by_space") or {}).items()):
        L.append(f"SPACE_COUNT {space} {cnt}")
    for name, cnt in sorted((m.get("simt_control_ops") or {}).items()):
        L.append(f"SIMT_CONTROL_OP {name} {cnt}")
    for name, cnt in sorted((m.get("identity_csr_reads") or {}).items()):
        L.append(f"SIMT_IDENTITY_CSR {name} {cnt}")
    for op, cnt in sorted((m.get("opcode_histogram") or {}).items()):
        L.append(f"OPCODE_PRESENT {op}")
        L.append(f"OPCODE_COUNT {op} {cnt}")
    for cid, why in sorted((rep.get("dropped") or {}).items()):
        L.append(f"DROPPED {cid} {why}")
    return "\n".join(L) + "\n"


def compile_object_checks(capsule: dict, rep: dict, prefix: str = "OBJECT") -> str | None:
    """FileCheck assertions over :func:`render`. Only facts the screen actually GROUNDED are asserted — an
    ungrounded check is left out entirely rather than asserted against the ``-`` the render emits, which
    would either pass vacuously or fail for the wrong reason.

    Legality is asserted whenever it was grounded; the SIMT/address-space findings are advisory and are
    NOT asserted here (they ride the screen's own warning verdict), because a serial single-space kernel is
    correct-but-slow and this file decides whether an oracle run is worth spending."""
    grounded = rep.get("grounded") or {}
    L = [f"// RTL-derived emitted-object checks (capsule={capsule.get('name')}) — decode legality",
         f"// {prefix}-DAG: EMPTY_OBJECT no"]
    if "decode_legality" in grounded:
        L.append(f"// {prefix}-DAG: DECODE_ILLEGAL 0{{{{$}}}}")
    if "result_write" in grounded:
        L.append(f"// {prefix}-NOT: MEMORY_WRITES 0{{{{$}}}}")
    for cls in list(((capsule.get("expected") or {}).get("instruction_classes") or [])):
        L.append(f"// {prefix}-DAG: OPCODE_PRESENT {cls}{{{{$}}}}")
    return "\n".join(L) + "\n" if len(L) > 1 else None
