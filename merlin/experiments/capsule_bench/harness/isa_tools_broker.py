"""Driver-side ISA-DEV-TOOLS broker — gives the assisted arms (arm-3 / arm-4) a derived assembler,
disassembler, and static linter for the target's self-hosted ISA, WITHOUT the target model entering the
sandbox. It is ORACLE-FREE and reads NO golden: it only encodes the syntax the agent chose (the derived
assembler) and inspects the agent's OWN emitted words (assembled with the same stock ``llvm-mc`` the oracle
uses, then disassembled/linted against the target's derived instruction model). The derived model is public
ISA structure — the same the agent already has via its ISA grounding — so nothing here is a cheat.

Mirrors :mod:`selfcheck_broker`: watch ``<ws>/.isa_channel``, answer each request with a JSON result. The
model derivation needs the target model venv, which is masked inside the sandbox — hence a driver-side
broker, exactly like the self-check.

Channel (under ``<ws>/.isa_channel/``):
  req_<id>.json   agent -> broker : {cmd: asm|disasm|lint, text|kernel_s, ...}
  resp_<id>.json  broker -> agent : the tool result JSON
  done_<id>       broker -> agent : completion marker
  STOP            driver -> broker: sentinel to exit
"""
from __future__ import annotations
import argparse
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_HERE))                                   # _common (target descriptor)
sys.path.insert(0, str(_REPO / "merlin" / "python"))            # merlin (out-of-box, oracle-free use)

_MODEL = None                                                    # derived once, cached in-process


def _model():
    """The target's derived IsaModel (assembler/disassembler/linter substrate), from the run's descriptor.
    Cached; an empty model (no shipped ISA definition) makes every tool return an honest 'unavailable'."""
    global _MODEL
    if _MODEL is None:
        import _common as _C
        from merlin.targetgen.target_experiment import load_target_experiment
        from merlin.targetgen.isa_model import isa_model_for, isa_model_for_target
        te = load_target_experiment(_C.EXP / "target_experiment.yaml")
        # prefer the mlc-derived fixed-format encoding fact (a wide-word SIMT core's whole ISA, from its RTL
        # decoder) so those tools work too; fall back to the shipped-ISA-definition probe for a
        # variable-format self-hosted ISA.
        m = isa_model_for_target(getattr(te, "target", "")) if getattr(te, "target", "") else None
        _MODEL = m if (m is not None and (m.is_fixed_format() or not m.is_empty())) else isa_model_for(te)
    return _MODEL


def _assemble(kernel_s_text: str) -> list[int]:
    """Assemble the agent's kernel.S to IMEM words with the SAME stock llvm-mc the oracle uses — so the
    disassembler/linter inspect exactly what will run."""
    from merlin.targetgen.program_oracle import _assemble_kernel_words
    with tempfile.TemporaryDirectory() as td:
        ks = Path(td) / "kernel.S"
        ks.write_text(kernel_s_text or "")
        # a fixed-format wide-word ISA (a SIMT core) is grouped at its own instruction width, so the
        # disassembler/linter see whole instructions rather than half-words.
        return _assemble_kernel_words(ks, Path(td), inst_width=getattr(_model(), "inst_width", 32))


_ENDPOINT = None  # (endpoint_kind, target), derived once


def _endpoint_and_target() -> tuple[str | None, str | None]:
    global _ENDPOINT
    if _ENDPOINT is None:
        import _common as _C
        from merlin.targetgen.target_experiment import load_target_experiment
        from merlin.targetgen import capsule_runner as CR
        te = load_target_experiment(_C.EXP / "target_experiment.yaml")
        _ENDPOINT = (CR._endpoint_of(te.target)[0], te.target)
    return _ENDPOINT


# The endpoint kind whose canonical artifact is ``llvm.inline_asm`` MLIR rather than a ``.word``
# kernel — those requests need rocc_asm/rocc_decode, not the IsaModel tools.
ROCC_ENDPOINT = "inline_asm_insn"


def is_rocc_endpoint(endpoint: str | None) -> bool:
    return endpoint == ROCC_ENDPOINT


@dataclass(frozen=True)
class BrokerCtx:
    """Everything :func:`_handle` needs from the run, passed IN rather than reached for.

    The endpoint, the derived ISA model, and the assembler are all properties of the run's
    descriptor, and resolving them from module state made the request routing untestable without
    reaching into that state — a caller could not say "route this as a fixed-format target" and had
    to arrange the ambient environment to say it instead. ``model``/``assemble`` stay CALLABLES so
    that deriving the model (which needs the target model venv) is still deferred until a request
    actually reaches the IsaModel arm; a RoCC target never pays for it."""
    endpoint: str | None
    target: str | None
    model: "Callable[[], Any]" = _model
    assemble: "Callable[[str], list[int]]" = _assemble


_CTX: "BrokerCtx | None" = None


def broker_ctx() -> BrokerCtx:
    """The ambient run's context, resolved once from its descriptor."""
    global _CTX
    if _CTX is None:
        endpoint, target = _endpoint_and_target()
        _CTX = BrokerCtx(endpoint=endpoint, target=target)
    return _CTX


def _rocc_handle(req: dict, target: str) -> dict:
    """RoCC / ``inline_asm_insn`` target (e.g. gemmini): the derived ISA facts live in
    ``rocc_decode.isa_constants`` (not the atlas IsaModel), and the canonical artifact is
    ``llvm.inline_asm`` MLIR, not a ``.word`` kernel — so route to rocc_asm/rocc_decode."""
    from merlin.targetgen import rocc_asm, rocc_decode
    cmd = req.get("cmd")
    if cmd == "asm":
        try:
            mlir = rocc_asm.assemble_text(target, req.get("text", ""))
        except rocc_asm.AsmError as e:
            return {"error": str(e)}
        return {"mlir": mlir, "n": mlir.count("llvm.inline_asm")}
    if cmd in ("disasm", "lint"):
        text = req.get("mlir") or req.get("kernel_s") or req.get("text", "")
        trace = rocc_decode.decode_text(text, source="isa_tools", target=target)
        classes = [i["class"] for i in trace["instructions"]]
        if cmd == "disasm":
            return {"instructions": trace["instructions"], "classes": classes, "n": len(classes)}
        n_unknown = classes.count("UNKNOWN")
        findings = []
        if n_unknown:
            findings.append(f"{n_unknown} instruction(s) decode to UNKNOWN — most likely inline-literal "
                            f"operands or a non-canonical .insn form. Emit each instruction via `asm` so "
                            f"operands are SSA values (llvm.mlir.constant), which assemble AND decode.")
        if not classes:
            findings.append("no instructions decoded — the artifact has no llvm.inline_asm `.insn` ops "
                            "(did you emit textual LLVM-IR or high-level ops instead of MLIR inline_asm?).")
        return {"findings": findings, "class_histogram": trace["summary"]["class_histogram"],
                "n_unknown": n_unknown, "n": len(classes)}
    if cmd == "debug":
        return _rocc_debug(req, target)
    return {"error": f"unknown cmd {cmd!r} (use asm|disasm|lint|debug)"}


def _rocc_debug(req: dict, target: str) -> dict:
    """LITE DEBUGGER for a RoCC / command-buffer target (e.g. gemmini): answer the agent's OWN command
    buffer on the RTL-derived mlc arc model and return the REDACTED per-op hardware state (cycles +
    scratchpad/accumulator/DRAM-refill counts per command + the RTL fingerprint). The counterpart of the
    external_backend kernel.S debugger. Golden-free: it runs the agent's cb over the capsule's CANONICAL
    inputs; the OUTPUT values and the pass/fail verdict are withheld by ``program_oracle`` (answer key)."""
    from merlin.targetgen.contract.materialize import public_capsules_for
    from merlin.targetgen.target_experiment import load_target_experiment
    from merlin.targetgen import program_oracle as PO
    import _common as _C
    cname = (req.get("capsule") or "").strip()
    caps_root = public_capsules_for(load_target_experiment(_C.EXP / "target_experiment.yaml"))
    cap_dir = caps_root / cname
    if not cname or not (cap_dir / "capsule.yaml").exists():
        avail = sorted(p.parent.name for p in caps_root.glob("*/capsule.yaml"))
        return {"error": f"debug: unknown capsule {cname!r}; pick one of {avail}"}
    raw = req.get("command_buffer")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception as e:  # noqa: BLE001
            return {"error": f"debug: command_buffer is not valid JSON: {str(e)[-200:]}"}
    if not isinstance(raw, dict) or not raw.get("commands"):
        return {"error": "debug: pass your emitted command_buffer.json (it must contain 'commands'); "
                         "for a RoCC target the debugger runs your COMMAND BUFFER on the arc model"}
    try:
        out = PO.run_command_buffer_debug(target, cb=raw, capsule_dir=cap_dir)
    except PO.OracleUnavailable as e:
        return {"error": f"debug oracle unavailable (mlc arc model absent): {e}"}
    except Exception as e:  # noqa: BLE001 — a run fault is the agent's cb error, reported as-is
        return {"error": f"debug run failed: {type(e).__name__}: {str(e)[-300:]}"}
    out["capsule"] = cname
    return out


def _handle(req: dict, ctx: BrokerCtx | None = None) -> dict:
    ctx = ctx or broker_ctx()
    if is_rocc_endpoint(ctx.endpoint):
        return _rocc_handle(req, ctx.target)
    from merlin.targetgen import isa_asm, isa_disasm, isa_lint
    model = ctx.model()
    # A fixed-format model (a wide-word SIMT core's whole ISA, from its RTL decoder) carries its ops in
    # ``field_layout``/``opcode_table`` and leaves ``by_mnemonic`` empty, so ``is_empty()`` is True even
    # though the disassembler/linter (which branch on ``is_fixed_format()``) can fully use it. Only refuse
    # when the model is BOTH signature-empty and not fixed-format — i.e. the target truly ships no ISA.
    if model.is_empty() and not model.is_fixed_format():
        return {"error": "no derived ISA model for this target (it ships no ISA definition)"}
    cmd = req.get("cmd")

    if cmd == "asm":
        try:
            words = isa_asm.assemble_text(model, req.get("text", ""))
        except isa_asm.AssembleError as e:
            return {"error": str(e)}
        return {"words": [f"0x{w:08x}" for w in words], "word_lines": isa_asm.to_word_lines(words),
                "n": len(words)}

    if cmd in ("disasm", "lint"):
        try:
            words = ctx.assemble(req.get("kernel_s", ""))
        except Exception as e:  # noqa: BLE001 — assembly failure is the agent's kernel error, reported as-is
            return {"error": f"kernel.S did not assemble: {str(e)[-300:]}",
                    "hint": "stock llvm-mc assembles ONLY raw `.word`/`.insn` directives — it cannot "
                            "assemble this target's custom mnemonics (VMATMUL-style). Emit each instruction "
                            "as a `.word 0x..`; use `asm` (a `CLASS field=value` listing) to get the exact "
                            "words from the target's own encoder."}
        recs = isa_disasm.disassemble(model, words)
        if cmd == "disasm":
            return {"records": recs, "n": len(recs)}
        findings = isa_lint.lint(model, words, op=req.get("op", "matmul"),
                                 output_dtype=req.get("output_dtype"),
                                 epilogue=tuple(req.get("epilogue") or ()),
                                 movement=bool(req.get("movement", False)))
        cov = isa_disasm.coverage(model, recs, op=req.get("op", "matmul"),
                                  output_dtype=req.get("output_dtype"),
                                  epilogue=tuple(req.get("epilogue") or ()),
                                  movement=bool(req.get("movement", False)))
        return {"findings": findings, "formatted": isa_lint.format_findings(findings), "coverage": cov}

    if cmd == "debug":
        return _handle_debug(req)

    return {"error": f"unknown cmd {cmd!r} (use asm|disasm|lint|debug)"}


def _debug_ctx():
    """(target, model_ext, public-capsule dir) for the debugger, from the run's descriptor. Cached-free
    (called rarely); raises with an actionable message if the target is not a self-hosted-ISA backend."""
    import _common as _C
    from merlin.targetgen.target_experiment import load_target_experiment
    from merlin.targetgen.contract.materialize import public_capsules_for
    from merlin.targetgen import capsule_runner as CR
    te = load_target_experiment(_C.EXP / "target_experiment.yaml")
    endpoint_kind, model_ext = CR._endpoint_of(te.target)
    if endpoint_kind != "external_backend":
        raise ValueError("debug is only for a self-hosted-ISA (external_backend) target — this target "
                         "runs a command-buffer/host-stream backend, so use disasm/lint + self_check")
    return te.target, model_ext, public_capsules_for(te)


def _handle_debug(req: dict) -> dict:
    """Run the agent's kernel.S on the functional model to instruction ``run_to`` and return committed
    scalar state + REDACTED DRAM windows (the output region is refused; see program_oracle). Oracle-touching
    but golden-free: the model runs the AGENT'S kernel, so DRAM holds only the given inputs + what the
    kernel itself wrote."""
    import tempfile
    from merlin.targetgen import program_oracle as PO
    try:
        target, model_ext, caps_root = _debug_ctx()
    except Exception as e:  # noqa: BLE001 — configuration/eligibility error, reported to the agent as-is
        return {"error": f"debug unavailable: {e}"}
    cname = (req.get("capsule") or "").strip()
    cap_dir = caps_root / cname
    if not cname or not (cap_dir / "capsule.yaml").exists():
        avail = sorted(p.parent.name for p in caps_root.glob("*/capsule.yaml"))
        return {"error": f"debug: unknown capsule {cname!r}; pick one of {avail}"}
    try:
        cb = PO.build_debug_cb(target, cap_dir)
    except Exception as e:  # noqa: BLE001
        return {"error": f"debug: could not build the command buffer for {cname}: {str(e)[-300:]}"}
    with tempfile.TemporaryDirectory() as td:
        ks = Path(td) / "kernel.S"
        ks.write_text(req.get("kernel_s", "") or "")
        try:
            out = PO.run_program_debug(target, model_ext=model_ext, cb=cb, kernel_s=ks,
                                       dump_regions=req.get("regions") or [],
                                       run_to=req.get("run_to"),
                                       state_summary=bool(req.get("state_summary", False)),
                                       workdir=Path(td), timeout=int(req.get("timeout", 300)))
        except PO.OracleUnavailable as e:
            return {"error": f"debug oracle unavailable (model venv / functional runner absent): {e}"}
        except Exception as e:  # noqa: BLE001 — a run fault is the agent's kernel error, reported as-is
            return {"error": f"debug run failed: {type(e).__name__}: {str(e)[-300:]}"}
    out["capsule"] = cname
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", required=True)
    ap.add_argument("--poll", type=float, default=0.4)
    a = ap.parse_args(argv)
    ch = Path(a.ws) / ".isa_channel"
    ch.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    while True:
        if (ch / "STOP").exists():
            break
        for req_f in sorted(ch.glob("req_*.json")):
            if req_f.name in seen:
                continue
            seen.add(req_f.name)
            rid = req_f.stem[len("req_"):]
            resp = ch / f"resp_{rid}.json"
            try:
                out = _handle(json.loads(req_f.read_text()))
            except Exception as e:  # noqa: BLE001 — never crash the broker on one bad request
                out = {"error": f"isa-tools broker: {type(e).__name__}: {str(e)[:200]}"}
            resp.write_text(json.dumps(out, indent=2))
            (ch / f"done_{rid}").write_text("ok")
        time.sleep(a.poll)


if __name__ == "__main__":
    raise SystemExit(main())
