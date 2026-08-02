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
from pathlib import Path

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
        from merlin.targetgen.isa_model import isa_model_for
        _MODEL = isa_model_for(load_target_experiment(_C.EXP / "target_experiment.yaml"))
    return _MODEL


def _assemble(kernel_s_text: str) -> list[int]:
    """Assemble the agent's kernel.S to IMEM words with the SAME stock llvm-mc the oracle uses — so the
    disassembler/linter inspect exactly what will run."""
    from merlin.targetgen.program_oracle import _assemble_kernel_words
    with tempfile.TemporaryDirectory() as td:
        ks = Path(td) / "kernel.S"
        ks.write_text(kernel_s_text or "")
        return _assemble_kernel_words(ks, Path(td))


def _handle(req: dict) -> dict:
    from merlin.targetgen import isa_asm, isa_disasm, isa_lint
    model = _model()
    if model.is_empty():
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
            words = _assemble(req.get("kernel_s", ""))
        except Exception as e:  # noqa: BLE001 — assembly failure is the agent's kernel error, reported as-is
            return {"error": f"kernel.S did not assemble: {str(e)[-300:]}"}
        recs = isa_disasm.disassemble(model, words)
        if cmd == "disasm":
            return {"records": recs, "n": len(recs)}
        findings = isa_lint.lint(model, words)
        cov = isa_disasm.coverage(model, recs, op=req.get("op", "matmul"),
                                  output_dtype=req.get("output_dtype"),
                                  epilogue=tuple(req.get("epilogue") or ()),
                                  movement=bool(req.get("movement", False)))
        return {"findings": findings, "formatted": isa_lint.format_findings(findings), "coverage": cov}

    return {"error": f"unknown cmd {cmd!r} (use asm|disasm|lint)"}


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
