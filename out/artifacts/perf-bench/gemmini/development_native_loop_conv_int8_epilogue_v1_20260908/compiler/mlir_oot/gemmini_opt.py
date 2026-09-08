"""`gemmini-opt` — the four CLI entrypoints of this out-of-tree target backend.

    gemmini-opt --verify-diagnostics <in.mlir>
    gemmini-opt --convert-iface-to-gemmini <in.mlir>
    gemmini-opt --convert-iface-to-gemmini --emit-command-buffer=<out.json> <in.mlir>
    gemmini-opt --convert-iface-to-gemmini --emit-target-artifact <in.mlir>
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from xdsl.printer import Printer                                       # noqa: E402

from mlir_oot import cmdbuf                                            # noqa: E402
from mlir_oot.codegen import gemmini_module, llvm_emit                 # noqa: E402
from mlir_oot.frontend import linalg_reader, parse as _parse, reader   # noqa: E402
from mlir_oot.lowering.plan import Builder, LoweringDeclined           # noqa: E402
from mlir_oot.lowering import host_lane, model_lane                     # noqa: E402
from mlir_oot.lowering.schedule import schedule                        # noqa: E402


def _print(op) -> str:
    out = io.StringIO()
    # Generic LLVM syntax preserves discardable ownership attrs on inline asm
    # and constants and round-trips through the upstream LLVM dialect parser.
    Printer(stream=out, print_generic_format=any(
        node.name == "llvm.func" for node in op.walk())).print_op(op)
    return out.getvalue() + "\n"


class Pipeline:
    """One run of the backend over one interface module."""

    def __init__(self, text: str, *, enable_source_conv: bool = False, input_prologue: dict | None = None):
        self.text = text
        self.enable_source_conv = enable_source_conv
        self.input_prologue = input_prologue
        self.module = None
        self.plan = None
        self.instrs = None
        self.staging = None
        self.declined = None
        self.linalg = None
        self.artifact = None
        self.mixed_declined = None

    def _externalize_input_prologue(self) -> None:
        """Move one proven leading static input quantizer outside timed model entry.

        The frontend supplies semantic quantization metadata. We still require the
        mixed plan to prove that task zero is exactly that region, reads only the
        corresponding entry argument, and writes the expected i8 tensor plus scalar
        scale. Any disagreement fails closed instead of silently changing timing.
        """
        if self.input_prologue is None:
            return
        cb = self.plan.command_buffer
        params = cb["params"]
        tasks = params["global_program_plan"]["tasks"]
        host = params["host_lane_segments"]
        spec = dict(self.input_prologue)
        arg = f"arg{spec['input_arg_index']}"
        if (not tasks or not host or tasks[0]["kind"] != "host"
                or tasks[0]["reads"] != [arg]
                or host[0]["regions"] != [spec["region_id"]]
                or host[0]["reads"] != [arg]
                or host[0]["writes"] != tasks[0]["writes"]):
            raise LoweringDeclined(
                "static input quantizer is not the complete leading host task",
                op="input_prologue")
        tensors = cb["tensors"]
        qouts = [name for name in tasks[0]["writes"]
                 if tensors[name]["dtype"] == spec["output_dtype"]
                 and tensors[name]["shape"] == spec["shape"]]
        scales = [name for name in tasks[0]["writes"]
                  if tensors[name]["dtype"] == "f32"
                  and tensors[name]["shape"] == [1]]
        if len(qouts) != 1 or len(scales) != 1:
            raise LoweringDeclined(
                "static input quantizer outputs are not one i8 tensor plus one f32 scalar",
                op="input_prologue")
        omitted = [ins for ins in self.instrs if ins.attrs.get("global_task_index") == 0]
        if len(omitted) != 1 or omitted[0].kind != "host_segment":
            raise LoweringDeclined(
                "leading input quantizer is not one compiler host segment",
                op="input_prologue")
        self.instrs = [ins for ins in self.instrs if ins not in omitted]
        removed = len(omitted)
        old_end = tasks[0]["instruction_end"]
        tasks[0]["execution"] = "external_input_prologue"
        tasks[0]["runtime_instruction_omitted"] = True
        tasks[0]["instruction_end"] = tasks[0]["instruction_start"]
        for task in tasks[1:]:
            if task["instruction_start"] >= old_end:
                task["instruction_start"] -= removed
                task["instruction_end"] -= removed
        spec.update({"input_tensor": arg, "quantized_tensor": qouts[0],
                     "scale_tensor": scales[0], "timed": False,
                     "implementation": "bundle_input_prologue"})
        params["input_prologue"] = spec
        global_plan = params["global_program_plan"]
        global_plan["schedule_instruction_count"] = len(self.instrs)
        global_plan["epilogue_instruction_range"] = [n-removed for n in global_plan["epilogue_instruction_range"]]

    def parse(self):
        self.module = _parse.parse_module(self.text)
        return self.module

    def lower(self):
        if self.module is None:
            self.parse()
        if not _parse.is_merlin_iface(self.module):
            wl = linalg_reader.read(self.module)
            if self.enable_source_conv:
                from mlir_oot.lowering.source_conv_model_lane import place_source_convolutions
                place_source_convolutions(self.module, wl)
            self.linalg = wl
            if wl.mesh_regions or wl.host_regions:
                # Some region of this module IS work the mesh admits.  Lower it as a mixed-lane
                # program; leaving admitted work on the host would be a placement defect.
                try:
                    planner = model_lane
                    if self.enable_source_conv:
                        from mlir_oot.lowering import source_conv_model_lane
                        planner = source_conv_model_lane
                    plan = planner.build(self.module, wl)
                    plan.command_buffer["params"]["global_program_plan"]["source_sha256"] = (
                        hashlib.sha256(self.text.encode("utf-8")).hexdigest())
                    instrs, staging = schedule(plan)
                    # Prove the program EMITS before committing to it.  Every entrypoint has to
                    # answer the same way about the same capsule: a command buffer that says a
                    # program exists next to an artifact that could not be built is the one
                    # inconsistency the runner reads as a protocol failure rather than a decline.
                    self.plan, self.instrs, self.staging = plan, instrs, staging
                    self._externalize_input_prologue()
                    self.artifact = llvm_emit.emit(plan, self.instrs, staging)
                    return self.plan
                except LoweringDeclined as exc:
                    if not wl.mesh_regions:
                        # Pure-host graphs use the same owned source-native loop plan.
                        # Preserve the actual unsupported operation, not the old unroll budget.
                        raise
                    # Not a program this backend can build after all -- fall through to the
                    # host-lane form, which STATES why rather than emitting nothing.
                    self.mixed_declined = exc.reason
            self.plan = host_lane.build(wl)
            self.staging = {}
            try:
                self.instrs = host_lane.host_instrs(self.plan, self.module, wl)
                # Same rule as the mixed path: prove the artifact EMITS before the command buffer
                # claims a program exists.  A buffer with commands beside an empty artifact is the
                # one inconsistency the runner cannot read as a decline.  A buffer that ALREADY
                # declines claims nothing, so it costs nothing to prove -- and for a host-lane
                # region the artifact is megabytes, which is time the command-buffer entrypoint
                # should not spend twice.
                if (self.plan.command_buffer.get("commands")
                        or (self.plan.command_buffer.get("kernel_abi") or {}).get("kind") == "whole_program"):
                    self.artifact = llvm_emit.emit(self.plan, self.instrs, self.staging)
            except LoweringDeclined as exc:
                cb = self.plan.command_buffer
                cb["commands"] = []
                cb.setdefault("params", {})["host_lane_program_emitted"] = False
                cb["declined"] = {
                    "reason": exc.reason,
                    "op": exc.op or (wl.regions[0].op if wl.regions else "linalg_on_tensors"),
                    "shape": list(exc.shape) or (list(wl.results[0][0]) if wl.results else [])}
                self.instrs, self.artifact = [], None
            if self.mixed_declined:
                cb = self.plan.command_buffer
                cb.setdefault("params", {})["mesh_lowering_declined"] = self.mixed_declined
                if not cb.get("commands") and "declined" not in cb:
                    cb["declined"] = {
                        "reason": self.mixed_declined,
                        "op": wl.regions[0].op or "linalg_on_tensors",
                        "shape": list(wl.results[0][0]) if wl.results else []}
            return self.plan
        self.plan = Builder(reader.read(self.module)).build()
        self.instrs, self.staging = schedule(self.plan)
        return self.plan

    def run(self):
        try:
            self.lower()
        except LoweringDeclined as exc:
            target = "gemmini"
            if self.module is not None:
                attr = self.module.attributes.get("merlin_iface.target")
                target = getattr(attr, "data", target)
            self.declined = cmdbuf.declined(target, exc.reason, op=exc.op, shape=exc.shape)
        return self


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gemmini-opt", add_help=True)
    ap.add_argument("--source-convolution", action="store_true", help="enable source-derived integer convolution task scheduling")
    ap.add_argument("--integer-contract", choices=["designated_i32_staged_f32_v1"], help="explicitly adopt integer accumulation plus staged f32 epilogue; NOT original QDQ bit-equivalence")
    ap.add_argument("--externalize-static-input-quantizer", action="store_true", help="explicit deployment ABI/timing boundary change; requires designated integer contract")
    ap.add_argument("--verify-diagnostics", action="store_true")
    ap.add_argument("--convert-iface-to-gemmini", action="store_true")
    ap.add_argument("--emit-target-artifact", action="store_true")
    ap.add_argument("--emit-command-buffer", default=None, metavar="PATH")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("input", nargs="?", default="-")
    args = ap.parse_args(argv)

    text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text()
    original_text = text
    preparation = None
    input_prologue = None
    if args.externalize_static_input_quantizer and args.integer_contract is None:
        ap.error("external input prologue requires an explicit designated integer contract")
    if args.integer_contract is not None:
        from mlir_oot.frontend.integer_prepare import prepare_int8_text, static_input_prologue
        text, preparation = prepare_int8_text(original_text)
        if args.externalize_static_input_quantizer:
            input_prologue = static_input_prologue(original_text)
            if input_prologue is None:
                ap.error("no unique proven static input quantizer")
    pipe = Pipeline(text, enable_source_conv=args.source_convolution or args.integer_contract is not None,
                    input_prologue=input_prologue)

    # ---- parse + verify ------------------------------------------------------------------
    try:
        pipe.parse()
    except Exception as exc:                                          # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.verify_diagnostics and not (args.convert_iface_to_gemmini
                                        or args.emit_command_buffer
                                        or args.emit_target_artifact):
        return 0

    pipe.run()
    if pipe.plan is not None and args.integer_contract is not None:
        pipe.plan.command_buffer.setdefault("params", {})["designated_integer_contract"] = {
            "name": args.integer_contract, "original_source_sha256": hashlib.sha256(original_text.encode()).hexdigest(),
            "normalized_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "preparation": preparation, "original_qdq_bit_equivalence": "NOT_CLAIMED",
            "input_prologue_externalized": input_prologue is not None}

    # ---- command buffer ------------------------------------------------------------------
    if args.emit_command_buffer:
        cb = pipe.declined if pipe.declined is not None else pipe.plan.command_buffer
        problems = cmdbuf.write(cb, args.emit_command_buffer)
        for p in problems:
            print(f"command-buffer: {p}", file=sys.stderr)
        if problems:
            return 1
        if not args.emit_target_artifact and not args.convert_iface_to_gemmini:
            return 0

    if pipe.declined is not None:
        entry = pipe.declined["declined"]
        print(f"declined: {entry['reason']}", file=sys.stderr)
        if not (args.convert_iface_to_gemmini or args.emit_target_artifact):
            return 0 if args.emit_command_buffer else 1
        # A DECLINE still has to be answered in the entrypoint's own language.  An empty stdout
        # here is read as a crashed tool; a module that carries the reason and no command is the
        # same refusal, stated where the runner can read it.
        module = (llvm_emit.declined_artifact(entry["reason"]) if args.emit_target_artifact
                  else gemmini_module.declined_module(entry["reason"], entry.get("op", ""),
                                                      entry.get("shape")))
        text_out = _print(module)
        if args.output:
            Path(args.output).write_text(text_out)
        else:
            sys.stdout.write(text_out)
        return 0

    # ---- target artifact -----------------------------------------------------------------
    try:
        if args.emit_target_artifact:
            module = (pipe.artifact if pipe.artifact is not None
                      else llvm_emit.emit(pipe.plan, pipe.instrs, pipe.staging))
        elif args.convert_iface_to_gemmini:
            module = gemmini_module.build(pipe.plan, pipe.instrs, pipe.staging)
        else:
            return 0
    except LoweringDeclined as exc:
        # A stage that only CODEGEN can refuse (a buffer with no address, say) still has to
        # arrive as a stated decline -- and it has to arrive the SAME WAY at every entrypoint.
        # Printing nothing here used to leave the command buffer already written as a program
        # with real commands while stdout stayed empty and the exit code was 1: three
        # entrypoints claiming the capsule lowers and the fourth reading as a crashed tool.
        # The refusal is restated on the command buffer (the place the contract puts it) and
        # answered in this entrypoint's own language.
        print(f"declined: {exc.reason}", file=sys.stderr)
        entry = cmdbuf.declined("gemmini", exc.reason, op=exc.op, shape=exc.shape)
        if args.emit_command_buffer:
            cmdbuf.write(entry, args.emit_command_buffer)
        module = (llvm_emit.declined_artifact(exc.reason) if args.emit_target_artifact
                  else gemmini_module.declined_module(exc.reason, exc.op, exc.shape))
        text_out = _print(module)
        if args.output:
            Path(args.output).write_text(text_out)
        else:
            sys.stdout.write(text_out)
        return 0

    text_out = _print(module)
    if args.output:
        Path(args.output).write_text(text_out)
    elif not args.emit_command_buffer or args.emit_target_artifact or True:
        sys.stdout.write(text_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
