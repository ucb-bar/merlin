"""(B) Static silicon-precondition linter.

Each rule encodes a real "ran in the functional oracle, hung/faulted on real silicon" bug as a check
against a DERIVED fact (never a per-target literal); where the needed fact cannot be derived it fails
closed to an ``UNKNOWN`` finding rather than passing silently.

Rules implemented here (all HW-agnostic):

* ``funct-legality`` — every accelerator instruction's ``funct`` lies in the RTL-derived legal set;
  an out-of-range funct is an encoding the silicon decoder rejects.
* ``untranscodable-op`` — for a self-hosted (fixed-format) target, the kernel's rv32 ``.text`` is
  actually encodable by the target's derived transcoder. This catches the fork-free-SIMT ``fence`` wall:
  a base-ISA ``fence`` (MISC_MEM, opcode ``0x0f``) is untranscodable, so a kernel containing one cannot
  be built/run and its final-store visibility cannot be guaranteed.
* ``host-assist`` — a program destined for a hostless substrate must not depend on the HTIF debug
  channel (``.htif`` section / ``tohost``): an HTIF image on bare silicon spins forever inside its FIRST
  print, before any work. (The caller supplies the two derived booleans — image audit + substrate — so
  this stays target-name-free.)
"""
from __future__ import annotations

from .facts import SiliconFacts
from .report import Finding, Severity


def funct_legality(trace: dict, facts: SiliconFacts) -> list[Finding]:
    """Every emitted accelerator ``funct`` must be in the RTL-derived legal set."""
    legal = facts.legal_funct
    ins = trace.get("instructions", [])
    if legal is None:
        if any(isinstance(i.get("funct"), int) for i in ins):
            return [Finding(
                "funct-legality", Severity.UNKNOWN,
                "legal funct set not derivable — cannot verify the emitted opcodes are decodable",
                derived_from=facts.provenance)]
        return []
    legal_set = set(legal)
    bad = [i for i in ins if isinstance(i.get("funct"), int) and i["funct"] not in legal_set]
    if not bad:
        return []
    functs = sorted({i["funct"] for i in bad})
    return [Finding(
        "funct-legality", Severity.FAULT,
        f"{len(bad)} instruction(s) use funct {functs} outside the RTL legal set "
        f"[{min(legal_set)}..{max(legal_set)}] — the silicon decoder would reject them",
        where=f"#{bad[0].get('index')}", derived_from=facts.provenance,
        evidence={"illegal_functs": functs, "indices": [i.get("index") for i in bad][:8]})]


def untranscodable_op(rv32_text: bytes, target: str) -> list[Finding]:
    """For a fixed-format self-hosted target, confirm the kernel ``.text`` is encodable by the target's
    derived transcoder. A ``TranscodeError`` (e.g. a base-ISA ``fence`` the transcoder cannot encode) is a
    hard precondition failure — the kernel cannot be built/run and its stores' visibility is unguaranteed.

    Not applicable to a RoCC inline-asm target (its transcoder rejects the *model*, not the program) —
    reported as an informational skip, never a fault."""
    try:
        from merlin.targetgen.isa_model import isa_model_for_target
        from merlin.targetgen.isa_transcode import FixedFormatTranscoder, TranscodeError
    except Exception:  # noqa: BLE001 — transcode toolchain unavailable
        return [Finding("untranscodable-op", Severity.UNKNOWN,
                        "ISA transcoder unavailable — cannot verify instruction encodability",
                        derived_from="isa_transcode import")]
    try:
        model = isa_model_for_target(target)
    except Exception as e:  # noqa: BLE001 — no derivable ISA model → fail closed
        return [Finding("untranscodable-op", Severity.UNKNOWN,
                        f"ISA model for the target not derivable ({e}) — cannot verify encodability",
                        derived_from="isa_model_for_target")]
    try:
        transcoder = FixedFormatTranscoder(model)
    except TranscodeError:
        # Not a fixed-format substrate (e.g. a RoCC inline-asm target) — this rule does not apply.
        return [Finding("untranscodable-op", Severity.INFO,
                        "target is not a fixed-format transcode substrate — rule not applicable",
                        derived_from="isa_transcode.FixedFormatTranscoder")]
    try:
        transcoder.transcode_text(rv32_text)
    except TranscodeError as e:
        return [Finding(
            "untranscodable-op", Severity.FAULT,
            f"the kernel .text contains an instruction the target cannot encode: {e}",
            derived_from="isa_transcode (target-derived opcode table)",
            fix_hint="remove the untranscodable op (e.g. a base-ISA fence on a fork-free SIMT path) or "
                     "route through a delivery path that supports it")]
    return []


def host_assist(*, hostless: bool | None, has_htif: bool | None,
                has_tohost: bool | None = None) -> list[Finding]:
    """A program bound for a hostless substrate must not depend on the HTIF debug channel.

    ``hostless`` = the delivery substrate has no fesvr host (derived by the caller from the board/substrate);
    ``has_htif`` / ``has_tohost`` = the image audit (a ``.htif`` section / a ``tohost`` symbol). All three
    are booleans the caller derives, so this rule carries no target name and no ELF parsing of its own."""
    if hostless is None or (has_htif is None and has_tohost is None):
        return [Finding("host-assist", Severity.UNKNOWN,
                        "substrate host-assist and/or image HTIF audit not supplied — cannot check the "
                        "HTIF-first-print-hang precondition",
                        derived_from="caller (substrate + elf_audit)")]
    depends_on_htif = bool(has_htif) or bool(has_tohost)
    if hostless and depends_on_htif:
        which = ".htif section" if has_htif else "tohost symbol"
        return [Finding(
            "host-assist", Severity.FAULT,
            f"image depends on the HTIF host channel ({which}) but the delivery substrate has no host — "
            f"it will spin forever inside its FIRST print on bare silicon",
            derived_from="substrate host-assist boundary + image audit",
            fix_hint="link the board's own UART console (console_init before the first character), "
                     "not HTIF")]
    return []
