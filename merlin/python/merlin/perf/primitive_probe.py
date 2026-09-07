"""Probe-only extraction of setup, one compute body, and readback from emitted LLVM.

The compiler's complete short kernel remains the source of instructions. This diagnostic splits
its first initialized overwrite primitive from initialization/readback so a host wrapper can time
that mechanism alone. It neither changes the candidate compiler nor edits a benchmark evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .instruction_motif import initialized_compute_primitives


@dataclass(frozen=True)
class PrimitiveProbeProgram:
    module: Any
    setup_symbol: str
    body_symbol: str
    readback_symbol: str
    argument_count: int
    domain_digest: str
    source_instruction_indices: tuple[int, int]
    source_completion_index: int
    include_operand_movement: bool = False
    timed_instruction_indices: tuple[int, ...] = ()


def extract_primitive_program(module: Any, *, target: str,
                              symbol_prefix: str = "merlin_primitive",
                              include_operand_movement: bool = False,
                              include_trailing_operand_movement: bool = False) -> PrimitiveProbeProgram:
    """Split the first initialized compute pair from a single-output-tile short kernel.

    The prefix initializes on-chip operands, followed by an explicit completion barrier. The body
    contains only the original preload/overwrite-compute pair and a completion barrier. The tail
    configures/readbacks its output after measurement. Additional reduction compute instructions
    are omitted: the verifier must check the extracted *tile reduction*, not the original capsule's
    complete reduction. Multi-output readback and host work are refused rather than silently copied.
    ``include_operand_movement`` retains the complete load/configuration prefix in the body;
    setup then contains only entry configuration/address plumbing and completion. The host wrapper
    restores this exact entry state after warmup, before the measured prefix.
    """
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from merlin.targetgen.rocc import decode

    functions = [op for op in module.body.block.ops if op.name == "llvm.func"]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("primitive extraction requires one straight-line function")
    function = functions[0]
    block = function.body.block
    operations = list(block.ops)
    allowed = {"llvm.inline_asm", "llvm.mlir.constant", "llvm.ptrtoint", "llvm.add", "llvm.return"}
    if any(op.name not in allowed for op in operations):
        raise ValueError("primitive source has host work or unsupported address plumbing")
    trace = decode.decode_module(module, target=target)
    rows = trace["instructions"]
    if any(row["class"] == "UNKNOWN" for row in rows):
        raise ValueError("primitive source has unknown instructions")
    primitives = initialized_compute_primitives(trace, target=target)
    if not primitives or primitives[0]["missing"]:
        raise ValueError("first compute primitive has no complete initialization witness")
    primitive = primitives[0]
    start, end = primitive["instruction_indices"]
    asm = [op for op in operations if op.name == "llvm.inline_asm"]
    source_start = operations.index(asm[start])
    timed_start = start
    if include_operand_movement:
        timed_start = next((i for i in range(start) if "spad_addr" in rows[i].get("decoded", {})), start)
        if timed_start == start:
            raise ValueError("queued context has no operand movement in its timed body")
        source_start = operations.index(asm[timed_start])
    config_classes = set(decode.isa_constants(target)["CONFIG_SUBTYPE"].values())
    readouts = [i for i, row in enumerate(rows) if "acc_addr" in row.get("decoded", {})]
    if len(readouts) != 1:
        raise ValueError("primitive extraction requires exactly one output-tile readback")
    readout_index = readouts[0]
    if readout_index <= end:
        raise ValueError("readback precedes the selected primitive")
    timed_end = end
    if include_trailing_operand_movement:
        if not include_operand_movement:
            raise ValueError("trailing movement requires a fixed-work movement/compute body")
        trailing = [index for index in range(end + 1, readout_index)
                    if "spad_addr" in rows[index].get("decoded", {})]
        if trailing:
            timed_end = trailing[-1]
        if any("spad_addr" not in rows[index].get("decoded", {})
               and not (rows[index].get("class") in config_classes
                        and rows[index].get("decoded", {}).get("subtype") == "LD")
               for index in range(end + 1, timed_end + 1)):
            raise ValueError("fixed-work trailing window contains another compute or unsupported effect")
    completions = [index for index in range(readout_index + 1, len(rows))
                   if rows[index]["class"] == "FENCE"]
    if not completions:
        raise ValueError("short source has no decoded post-readback completion operation")
    completion_index = completions[-1]
    completion = asm[completion_index]
    if (completion.has_side_effects is None
            or "~{memory}" not in completion.constraints.data.split(",")):
        raise ValueError("source completion lacks side-effect and host-memory ordering semantics")
    # Only the final store configuration is needed. Keep its exact encoded payload rather than
    # inventing a new dtype, stride, scaling or address interpretation.
    store_configs = [i for i in range(end + 1, readout_index)
                     if rows[i]["class"] in config_classes
                     and rows[i].get("decoded", {}).get("subtype") == "ST"]
    if not store_configs:
        raise ValueError("output readback configuration is not explicit in the short kernel")
    readback_ops = [asm[store_configs[-1]], asm[readout_index]]

    def make(symbol: str, selected: Sequence[Any]):
        body = Block(arg_types=[value.type for value in block.args])
        mapping = dict(zip(block.args, body.args, strict=True))
        copied = set()

        def copy(op):
            if op in copied:
                return
            for operand in op.operands:
                if operand in mapping:
                    continue
                owner = operand.owner
                if owner not in operations or owner.name == "llvm.inline_asm":
                    raise ValueError("primitive has a dependency outside supported scalar definitions")
                copy(owner)
            cloned = op.clone(value_mapper=mapping)
            cloned.attributes.pop("merlin.global_task", None)
            body.add_op(cloned)
            mapping.update(zip(op.results, cloned.results, strict=True))
            copied.add(op)

        for op in selected:
            copy(op)
        # Clone the source's actual decoder-recognized completion, preserving its instruction,
        # operands, side effects and memory clobbers. Shared code must not invent a host ISA.
        copy(completion)
        body.add_op(llvm.ReturnOp())
        return llvm.FuncOp(symbol, llvm.LLVMFunctionType([value.type for value in block.args]),
                           linkage=llvm.LinkageAttr("external"), body=Region([body]))

    setup, body, readback = (f"{symbol_prefix}_{suffix}" for suffix in ("setup", "body", "readback"))
    result = ModuleOp([make(setup, operations[:source_start]),
                       make(body, asm[timed_start:timed_end + 1]), make(readback, readback_ops)])
    result.verify()
    return PrimitiveProbeProgram(result, setup, body, readback, len(block.args),
                                 primitive["domain_digest"], (start, end), completion_index,
                                 include_operand_movement, tuple(range(timed_start, timed_end + 1)))


def render_primitive_host_wrapper(program: PrimitiveProbeProgram, *,
                                   declarations: str, argument_expressions: Sequence[str],
                                   cycle_reader: str, verify_call: str,
                                   before_measurement: str = "", after_measurement: str = "") -> str:
    """Render a diagnostic C main: setup, warm body, measured body, readback and correctness.

    ``declarations`` supplies the host adapter's target timer include and initialized buffers.
    ``cycle_reader`` names its existing serialized cycle reader, not a target-specific instruction
    invented here. The body includes completion, so its measured cycles include all issued work.
    Initialization, output movement and correctness checks are outside the measured interval.
    These host-owned fragments are never accepted from an optimizing candidate.
    Optional counter configuration/reset goes after warmup and before the timer;
    snapshots/reads go after the timer and before output readback. No counter
    instrumentation is inserted into the timed compute interval.
    """
    if len(argument_expressions) != program.argument_count:
        raise ValueError("wrapper arguments do not match the extracted kernel ABI")
    if not declarations.strip() or not cycle_reader.strip() or not verify_call.strip():
        raise ValueError("host adapter must supply real data, serialized timer, and correctness check")
    args = ", ".join(argument_expressions)
    signature = ", ".join("void *" for _ in argument_expressions) or "void"
    # A queued-load body may change load configuration. Restore the exact setup state after
    # warmup, before counters/timing, so warm and measured bodies start in the same state.
    restore = f"  {program.setup_symbol}({args});\n" if program.include_operand_movement else ""
    return f"""#include <stdint.h>
#include <stdio.h>
{declarations}
extern void {program.setup_symbol}({signature});
extern void {program.body_symbol}({signature});
extern void {program.readback_symbol}({signature});
int main(void) {{
  {program.setup_symbol}({args});
  {program.body_symbol}({args});
{restore}{before_measurement}  const uint64_t begin = {cycle_reader}();
  {program.body_symbol}({args});
  const uint64_t end = {cycle_reader}();
{after_measurement}  {program.readback_symbol}({args});
  const int correct = ({verify_call});
  printf("MERLIN_PRIMITIVE correct=%d total_compute_cycles=%llu warmup_runs=1 measured_runs=1\\n",
         correct, (unsigned long long)(end - begin));
  return correct ? 0 : 1;
}}
"""
