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
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin_experiments.phase1.context import InvocationContext


def _model(context: InvocationContext):
    """The target's derived IsaModel (assembler/disassembler/linter substrate), from the run's descriptor.
    An empty model (no shipped ISA definition) makes every tool return an honest 'unavailable'."""
    from merlin.targetgen.isa_model import isa_model_for, isa_model_for_target
    from merlin.targetgen.target_experiment import load_target_experiment

    te = load_target_experiment(context.descriptor)
    # Prefer the derived fixed-format encoding; otherwise use the shipped ISA definition.
    m = isa_model_for_target(getattr(te, "target", "")) if getattr(te, "target", "") else None
    return m if (m is not None and (m.is_fixed_format() or not m.is_empty())) else isa_model_for(te)


def _read_schedule_contract(path: Path | None) -> dict:
    """Read a target's small declarative scheduling contract.

    Do not cache this document in the long-lived broker.  Compiler bring-up commonly tightens the
    target contract while an authoring session is alive; retaining the first read makes the linter
    silently apply stale latency facts until the whole run is restarted.  The ISA model remains cached
    because deriving it is expensive, while this is one local YAML read per lint request.
    """
    if path is None or not path.is_file():
        return {}
    import yaml

    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return doc if isinstance(doc, dict) else {}


def _schedule_contract(context: InvocationContext) -> dict:
    """Optional explicit-latency facts shipped inside this target's frozen hardware bring-up set.

    The checker is target-agnostic; a target opts in by providing ``schedule_contract.yaml`` beside its
    RTL/ISA/examples. Missing data means "no scheduling check", never guessed latency.
    """
    from merlin.targetgen.sandbox.bwrap import resolve_grant
    from merlin.targetgen.target_experiment import load_target_experiment

    te = load_target_experiment(context.descriptor)
    root = resolve_grant(str(te.hwbringup_set), context.repo) if te.hwbringup_set else None
    return _read_schedule_contract(root / "schedule_contract.yaml" if root else None)


def _assemble(kernel_s_text: str, model: Callable[[], Any]) -> list[int]:
    """Assemble the agent's kernel.S to IMEM words with the SAME stock llvm-mc the oracle uses — so the
    disassembler/linter inspect exactly what will run."""
    from merlin.targetgen.program_oracle import _assemble_kernel_words

    with tempfile.TemporaryDirectory() as td:
        ks = Path(td) / "kernel.S"
        ks.write_text(kernel_s_text or "")
        # a fixed-format wide-word ISA (a SIMT core) is grouped at its own instruction width, so the
        # disassembler/linter see whole instructions rather than half-words.
        return _assemble_kernel_words(ks, Path(td), inst_width=getattr(model(), "inst_width", 32))


def _endpoint_and_target(context: InvocationContext) -> tuple[str | None, str | None]:
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen.target_experiment import load_target_experiment

    te = load_target_experiment(context.descriptor)
    return CR._endpoint_of(te.target)[0], te.target


# The endpoint kind whose canonical artifact is ``llvm.inline_asm`` MLIR rather than a ``.word``
# kernel — those requests need rocc_asm/rocc_decode, not the IsaModel tools.
ROCC_ENDPOINT = "inline_asm_insn"


def is_rocc_endpoint(endpoint: str | None) -> bool:
    return endpoint == ROCC_ENDPOINT


def _unconfigured(*args):
    raise RuntimeError("ISA broker operation requires an explicit invocation context")


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
    model: Callable[[], Any] = _unconfigured
    assemble: Callable[[str], list[int]] = _unconfigured
    schedule_contract: Callable[[], dict] = _unconfigured
    debug: Callable[[dict], dict] = _unconfigured


def broker_ctx(context: InvocationContext) -> BrokerCtx:
    """Bind one invocation, caching only its model and endpoint, never schedule facts."""
    endpoint, target = _endpoint_and_target(context)
    cached_model = None

    def model():
        nonlocal cached_model
        if cached_model is None:
            cached_model = _model(context)
        return cached_model

    return BrokerCtx(
        endpoint=endpoint,
        target=target,
        model=model,
        assemble=lambda text: _assemble(text, model),
        schedule_contract=lambda: _schedule_contract(context),
        debug=(lambda req: _rocc_debug(req, target, context))
        if is_rocc_endpoint(endpoint)
        else (lambda req: _handle_debug(req, context)),
    )


def _capability_utilization(text: str, target: str) -> dict:
    """Declared-versus-emitted instruction use for ``text``, or an explicit UNKNOWN.

    DELEGATED, not reimplemented. The measurement lives in
    :func:`merlin.perf.isa_utilization.capability_utilization_for_target` so that this phase and
    the whole-model optimization phase read the SAME declared table by the same rule. It was
    written here first and copied nowhere; when the optimization agent needed it too, the producer
    moved into the library rather than being written a second time.
    """
    from merlin.perf.isa_utilization import capability_utilization_for_target

    return capability_utilization_for_target(text, target=target)


def _rocc_handle(req: dict, target: str, debug: Callable[[dict], dict] = _unconfigured) -> dict:
    """RoCC / ``inline_asm_insn`` target (e.g. gemmini): the derived ISA facts live in
    ``rocc_decode.isa_constants`` (not the atlas IsaModel), and the canonical artifact is
    ``llvm.inline_asm`` MLIR, not a ``.word`` kernel — so route to rocc_asm/rocc_decode."""
    # The RoCC tools are a package, not legacy top-level ``targetgen`` modules.  Importing the old
    # names made every live Arm4 request fail only after launch, while the fixed-format broker tests
    # continued to pass because they never exercised this endpoint.
    from merlin.targetgen.rocc import asm as rocc_asm
    from merlin.targetgen.rocc import decode as rocc_decode

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
            findings.append(
                f"{n_unknown} instruction(s) decode to UNKNOWN — most likely inline-literal "
                f"operands or a non-canonical .insn form. Emit each instruction via `asm` so "
                f"operands are SSA values (llvm.mlir.constant), which assemble AND decode."
            )
        if not classes:
            findings.append(
                "no instructions decoded — the artifact has no llvm.inline_asm `.insn` ops "
                "(did you emit textual LLVM-IR or high-level ops instead of MLIR inline_asm?)."
            )
        utilization = _capability_utilization(text, target)
        unused = utilization.get("unused") or []
        if unused:
            findings.append(
                f"your program emits {utilization['used_count']} of the "
                f"{utilization['declared_count']} instructions this target's own RTL declares. "
                f"Never emitted: {', '.join(str(u['name']) for u in unused)}. An unused "
                f"instruction is an OPPORTUNITY, not a defect — but a whole device-side sequencer "
                f"left unemitted means its work is running as host scalar code, and nothing else "
                f"in this lint can tell you that."
            )
        if utilization.get("undeclared_emitted"):
            findings.append(
                f"{len(utilization['undeclared_emitted'])} emitted funct(s) are NOT in the "
                f"target's derived decode table: {utilization['undeclared_emitted']}. That is a "
                f"provenance problem, not an optimization one — the program is using an "
                f"instruction nobody has vouched for."
            )
        return {
            "findings": findings,
            "class_histogram": trace["summary"]["class_histogram"],
            "n_unknown": n_unknown,
            "n": len(classes),
            # WHAT THE MACHINE OFFERS AGAINST WHAT THE PROGRAM USES. The lint above judges only the
            # instructions that ARE there; a capability the compiler never reaches for is invisible
            # to every one of its checks and to the histogram beside it. Measured on a whole-model
            # emission: 8 of 25 declared functs were used, and the 17 unused included the entire
            # device-side convolution sequencer, so every convolution's patch generation ran as
            # host scalar code — 37% of all host dynamic operations — while the command buffer was
            # well formed and the gate passed.
            "capability_utilization": utilization,
        }
    if cmd == "debug":
        return debug(req)
    return {"error": f"unknown cmd {cmd!r} (use asm|disasm|lint|debug)"}


def _rocc_debug(req: dict, target: str, context: InvocationContext) -> dict:
    """LITE DEBUGGER for a RoCC / command-buffer target (e.g. gemmini): answer the agent's OWN command
    buffer on the RTL-derived mlc arc model and return the REDACTED per-op hardware state (cycles +
    scratchpad/accumulator/DRAM-refill counts per command + the RTL fingerprint). The counterpart of the
    external_backend kernel.S debugger. Golden-free: it runs the agent's cb over the capsule's CANONICAL
    inputs; the OUTPUT values and the pass/fail verdict are withheld by ``program_oracle`` (answer key)."""
    from merlin.targetgen import program_oracle as PO
    from merlin.targetgen.target_experiment import load_target_experiment
    from merlin_experiments.corpus.admission import public_capsules_for

    cname = (req.get("capsule") or "").strip()
    caps_root = public_capsules_for(load_target_experiment(context.descriptor))
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
        return {
            "error": "debug: pass your emitted command_buffer.json (it must contain 'commands'); "
            "for a RoCC target the debugger runs your COMMAND BUFFER on the arc model"
        }
    try:
        out = PO.run_command_buffer_debug(target, cb=raw, capsule_dir=cap_dir)
    except PO.OracleUnavailable as e:
        return {"error": f"debug oracle unavailable (mlc arc model absent): {e}"}
    except Exception as e:  # noqa: BLE001 — a run fault is the agent's cb error, reported as-is
        return {"error": f"debug run failed: {type(e).__name__}: {str(e)[-300:]}"}
    out["capsule"] = cname
    return out


def _handle(req: dict, ctx: BrokerCtx) -> dict:
    if is_rocc_endpoint(ctx.endpoint):
        return _rocc_handle(req, ctx.target, ctx.debug)
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
            words = isa_asm.assemble_text(model, req.get("text", ""), schedule_contract=ctx.schedule_contract())
        except isa_asm.AssembleError as e:
            return {"error": str(e)}
        # Rendered at the target's OWN instruction width. A 64-bit wide-word core assembled through a
        # 32-bit `.word` would hand the agent a TRUNCATED instruction that still looks like a result;
        # `to_data_lines` emits `.quad` above 32 bits and is byte-identical at 32.
        nibbles = max(8, (int(getattr(model, "inst_width", 32)) + 3) // 4)
        return {
            "words": [f"0x{w:0{nibbles}x}" for w in words],
            "word_lines": isa_asm.to_data_lines(words, getattr(model, "inst_width", 32)),
            "n": len(words),
        }

    if cmd in ("disasm", "lint"):
        try:
            words = ctx.assemble(req.get("kernel_s", ""))
        except Exception as e:  # noqa: BLE001 — assembly failure is the agent's kernel error, reported as-is
            return {
                "error": f"kernel.S did not assemble: {str(e)[-300:]}",
                "hint": "stock llvm-mc assembles ONLY raw `.word`/`.insn` directives — it cannot "
                "assemble this target's custom mnemonics (VMATMUL-style). Emit each instruction "
                "as a `.word 0x..`; use `asm` (a `CLASS field=value` listing) to get the exact "
                "words from the target's own encoder.",
            }
        recs = isa_disasm.disassemble(model, words)
        if cmd == "disasm":
            return {"records": recs, "n": len(recs)}
        schedule_contract = ctx.schedule_contract()
        cycle_budget = req.get("cycle_budget")
        cycle_budget = cycle_budget if isinstance(cycle_budget, int) else None
        findings = isa_lint.lint(
            model,
            words,
            op=req.get("op", "matmul"),
            output_dtype=req.get("output_dtype"),
            epilogue=tuple(req.get("epilogue") or ()),
            movement=bool(req.get("movement", False)),
            schedule_contract=schedule_contract,
            cycle_budget=cycle_budget,
        )
        schedule = isa_lint.analyze_schedule(
            model, words, schedule_contract=schedule_contract, cycle_budget=cycle_budget
        )
        cov = isa_disasm.coverage(
            model,
            recs,
            op=req.get("op", "matmul"),
            output_dtype=req.get("output_dtype"),
            epilogue=tuple(req.get("epilogue") or ()),
            movement=bool(req.get("movement", False)),
        )
        return {
            "findings": findings,
            "formatted": isa_lint.format_findings(findings),
            "coverage": cov,
            "schedule": {k: v for k, v in schedule.items() if k != "findings"},
        }

    if cmd == "debug":
        return ctx.debug(req)

    return {"error": f"unknown cmd {cmd!r} (use asm|disasm|lint|debug)"}


def _debug_ctx(context: InvocationContext):
    """(target, model_ext, public-capsule dir) for the debugger, from the run's descriptor. Cached-free
    (called rarely); raises with an actionable message if the target is not a self-hosted-ISA backend."""
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen.target_experiment import load_target_experiment
    from merlin_experiments.corpus.admission import public_capsules_for

    te = load_target_experiment(context.descriptor)
    endpoint_kind, model_ext = CR._endpoint_of(te.target)
    if endpoint_kind != "external_backend":
        raise ValueError(
            "debug is only for a self-hosted-ISA (external_backend) target — this target "
            "runs a command-buffer/host-stream backend, so use disasm/lint + self_check"
        )
    return te.target, model_ext, public_capsules_for(te)


def _handle_debug(req: dict, context: InvocationContext) -> dict:
    """Run the agent's kernel.S on the functional model to instruction ``run_to`` and return committed
    scalar state + REDACTED DRAM windows (the output region is refused; see program_oracle). Oracle-touching
    but golden-free: the model runs the AGENT'S kernel, so DRAM holds only the given inputs + what the
    kernel itself wrote."""
    import tempfile

    from merlin.targetgen import program_oracle as PO

    try:
        target, model_ext, caps_root = _debug_ctx(context)
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
            out = PO.run_program_debug(
                target,
                model_ext=model_ext,
                cb=cb,
                kernel_s=ks,
                dump_regions=req.get("regions") or [],
                run_to=req.get("run_to"),
                state_summary=bool(req.get("state_summary", False)),
                workdir=Path(td),
                timeout=int(req.get("timeout", 300)),
            )
        except PO.OracleUnavailable as e:
            return {"error": f"debug oracle unavailable (model venv / functional runner absent): {e}"}
        except Exception as e:  # noqa: BLE001 — a run fault is the agent's kernel error, reported as-is
            return {"error": f"debug run failed: {type(e).__name__}: {str(e)[-300:]}"}
    out["capsule"] = cname
    return out


def _completed_request_names(ch: Path) -> set[str]:
    """Return requests that already have an atomic response/completion pair.

    Channel state is durable across broker restarts.  Starting with an empty in-memory ``seen`` set
    replays every historical request—hundreds in a long authoring run—and can make the first new request
    time out behind obsolete disassemblies.  A response without ``done`` is deliberately not complete:
    clients only trust the pair, so the broker must repair that request after a crash.
    """
    completed: set[str] = set()
    for done in ch.glob("done_*"):
        rid = done.name[len("done_") :]
        req = ch / f"req_{rid}.json"
        resp = ch / f"resp_{rid}.json"
        if req.is_file() and resp.is_file():
            completed.add(req.name)
    return completed


def main(argv=None, *, context: InvocationContext | Callable[[], InvocationContext] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", required=True)
    ap.add_argument("--poll", type=float, default=0.4)
    ap.add_argument("--descriptor", type=Path, help="Explicit target experiment descriptor")
    ap.add_argument("--repo", type=Path, help="Explicit repository/work root for descriptor resources")
    a = ap.parse_args(argv)
    if context is None:
        if a.descriptor is None or a.repo is None:
            ap.error("installed ISA broker requires --descriptor and --repo")
        from merlin_experiments.phase1.context import load_context

        context = load_context(a.descriptor, repo=a.repo)
    elif a.descriptor is not None or a.repo is not None:
        ap.error("pass descriptor/repo arguments or an invocation context, not both")
    elif callable(context):
        context = context()  # Native compatibility defaults initialize only after parsing/help.
    ctx = None
    ch = Path(a.ws) / ".isa_channel"
    ch.mkdir(parents=True, exist_ok=True)
    seen = _completed_request_names(ch)
    # STOP alone does not bound this broker's life: the sentinel is written by the driver, so if the
    # driver dies first nobody ever writes it and the broker polls forever. Three sibling brokers were
    # found orphaned to init hours after their run ended, spawned for a round that never started. Exit
    # when the process that started us is gone.
    orig_ppid = os.getppid()
    while True:
        if (ch / "STOP").exists() or os.getppid() != orig_ppid:
            break
        for req_f in sorted(ch.glob("req_*.json")):
            if req_f.name in seen:
                continue
            seen.add(req_f.name)
            rid = req_f.stem[len("req_") :]
            resp = ch / f"resp_{rid}.json"
            try:
                request = json.loads(req_f.read_text())
                if ctx is None:
                    ctx = broker_ctx(context)
                out = _handle(request, ctx)
            except Exception as e:  # noqa: BLE001 — never crash the broker on one bad request
                out = {"error": f"isa-tools broker: {type(e).__name__}: {str(e)[:200]}"}
            resp.write_text(json.dumps(out, indent=2))
            (ch / f"done_{rid}").write_text("ok")
        time.sleep(a.poll)


if __name__ == "__main__":
    raise SystemExit(main())
