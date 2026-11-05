"""Small passed-contract consumer of convolution effects through generated ISA.

Invoked by conv_epoch_recipe.py --adapter-contract FILE --output DIR. The toy
contract is functional only: it describes no physical target, backend or timing.
The reusable binding seam is conv_epoch_recipe.generated_effect_callbacks.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import fields as dataclass_fields
from fractions import Fraction
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def toy_contract():
    """Explicit premises for the sequential toy machine implemented below."""
    from merlin.compile.scheduling.convolution_exploration import ConvolutionRhsResidencyLaw
    from merlin.compile.scheduling.window_transfer import (
        ConvolutionComputeEpochLaw,
        ConvolutionInputRetentionLaw,
        ConvolutionLoadIssueLaw,
        ConvolutionReadoutEpochLaw,
    )

    def flags(cls):
        return {f.name: True for f in dataclass_fields(cls) if f.type in (bool, "bool")}

    return dict(schema="merlin.toy_convolution_adapter.v1", domain="sequential-toy-only",
        source=dict(shape=[1, 3, 5, 2, 5], kernel=[1, 1], stride=[1, 1], pad=[0, 0, 0, 0],
            readout=dict(numerator=1, denominator=8, rounding="nearest-even", clamp=[-128, 127])),
        machine=dict(primitive=2, operand_rows=128, accumulator_rows=64, input_base=4,
            row_pitch=5, slots=2, input_private_rows=20, stage_rows=2, rhs_base=64, pending=2),
        laws=dict(retention=dict(flags(ConvolutionInputRetentionLaw), max_live_tiles=3),
            issue=dict(flags(ConvolutionLoadIssueLaw), load_issue_budget=32),
            compute=dict(flags(ConvolutionComputeEpochLaw), max_compute_pairs=64, pending_compute_pairs=2),
            readout=dict(flags(ConvolutionReadoutEpochLaw), max_issued_readouts=32, pending_readouts=2),
            rhs=flags(ConvolutionRhsResidencyLaw),
            array=dict(qualification_identity="sequential-toy-fresh-retain"),
            transfer=dict(tile_columns=2, block_rows=2, max_columns=6, scaled_i8=True,
                exact_block_column_map=True, original_numerical_state=True)),
        # Alternatives demonstrate the interface; no timing rank or chosen default.
        alternatives=[dict(name="original", group=False, grouped_transport=False, wide_readout=False),
            dict(name="row", group=True, max_row_blocks=3, max_output_rows=1,
                grouped_transport=False, wide_readout=False),
            dict(name="union", group=True, max_row_blocks=6, max_output_rows=2,
                grouped_transport=True, wide_readout=True)])


class ScfControl:
    """Ordinary integer SCF control; payload effects stay in the supplied adapter."""

    def __init__(self):
        from xdsl.ir import Block
        self.block = Block()
        self.entry = self.block

    def add(self, op):
        self.block.add_op(op)
        return op.results[0] if op.results else None

    def value(self, value):
        from xdsl.dialects import arith
        from xdsl.dialects.builtin import i1, i64
        from xdsl.ir import SSAValue
        if type(value) in (int, bool):
            return self.add(arith.ConstantOp.from_int_and_width(int(value), 64))
        if isinstance(value, SSAValue) and value.type == i1:
            return self.add(arith.ExtUIOp(value, i64))
        if not isinstance(value, SSAValue) or value.type != i64:
            raise ValueError("toy instruction operands require exact i64 indices")
        return value

    def op(self, opcode, lhs, rhs, dtype="i64"):
        from xdsl.dialects import arith
        operators = dict(add=arith.AddiOp, sub=arith.SubiOp, mul=arith.MuliOp,
            udiv=arith.DivUIOp, urem=arith.RemUIOp, **{"and": arith.AndIOp})
        a, b = (lhs, rhs) if dtype == "i1" else (self.value(lhs), self.value(rhs))
        return self.add(operators[opcode](a, b))

    def cmp(self, predicate, lhs, rhs):
        from xdsl.dialects import arith
        return self.add(arith.CmpiOp(self.value(lhs), self.value(rhs), predicate))

    def select(self, condition, lhs, rhs):
        from xdsl.dialects import arith
        return self.add(arith.SelectOp(condition, self.value(lhs), self.value(rhs)))

    def region(self, block, callback):
        from xdsl.dialects import scf
        outer, self.block = self.block, block
        callback()
        self.block.add_op(scf.YieldOp())
        self.block = outer

    def branch(self, predicate, yes, no):
        from xdsl.dialects import scf
        from xdsl.ir import Block, Region
        if type(predicate) is bool:
            (yes if predicate else no)()
            return
        a, b = Block(), Block()
        self.region(a, yes)
        self.region(b, no)
        self.add(scf.IfOp(predicate, [], Region(a), Region(b)))

    def loop(self, start, stop, step, callback):
        from xdsl.dialects import scf
        from xdsl.dialects.builtin import i64
        from xdsl.ir import Block, Region
        bounds = [self.value(v) for v in (start, stop, step)]
        body = Block(arg_types=[i64])
        self.region(body, lambda: callback(body.args[0]))
        self.add(scf.ForOp(*bounds, [], Region(body)))


class ToyMachine:
    """Bounded FIFO execution of the *generated IR* effects, with real drains."""

    def __init__(self, source, inputs, image, weights, readout, pending):
        self.p, self.inputs, self.image, self.weights = source.plan, inputs, image, weights
        self.readout, self.pending_limit = readout, pending
        self.sp, self.acc, self.rhs_ids, self.output = {}, {}, {}, {}
        self.retained = None
        self.pending, self.counts = [], Counter()

    def issue(self, kind, a, *, completion=False):
        self.counts[("complete:" if completion else "effect:") + kind] += 1
        if completion:
            while self.pending:
                self.execute(*self.pending.pop(0))
        else:
            if len(self.pending) == self.pending_limit:
                self.execute(*self.pending.pop(0))
            self.pending.append((kind, a))

    def execute(self, kind, a):
        p = self.p
        if kind in ("stage_input_rows", "fill_input_rows"):
            for r in range(a["rows"]):
                for c in range(a["columns"]):
                    key = a["local_row"] + (c // p.channel_tile) * a["block_stride_rows"] + r, c % p.channel_tile
                    self.sp[key] = (a["value"] if kind == "fill_input_rows" else
                        self.image[a["first_element"] + r * a["row_stride_elements"] + c])
        elif kind == "load_weight":
            for r in range(a["rows"]):
                for c in range(a["columns"]):
                    self.sp[a["local_row"] + r, c] = self.weights[a["reduction_row"] + r][a["output_column"] + c]
            self.rhs_ids[a["local_row"]] = (a["reduction_row"], a["output_column"])
        elif kind == "compute":
            identity = self.rhs_ids[a["rhs_row"]], a["depth"], a["columns"]
            if a.get("fresh_weights", True):
                self.retained = identity
            elif self.retained != identity:
                raise ValueError("generated IR retained a different RHS")
            for r in range(a["rows"]):
                for c in range(a["columns"]):
                    key = a["accumulator_row"] + r, c
                    value = sum(self.sp[a["lhs_row"] + r * self.inputs.row_step, k]
                        * self.sp[a["rhs_row"] + k, c] for k in range(a["depth"]))
                    value += 0 if a["first_reduction"] else self.acc[key]
                    self.acc[key] = (value + 2**31) % 2**32 - 2**31
        elif kind == "store":
            for r in range(a["rows"]):
                for c in range(a["columns"]):
                    key = a["output_row"] + r, a["output_column"] + c
                    if key in self.output:
                        raise ValueError("generated IR wrote an original output twice")
                    self.output[key] = quantize(self.acc[a["accumulator_row"]
                        + (c // p.column_tile) * p.primitive + r, c % p.column_tile], self.readout)
        else:
            raise ValueError(f"toy interpreter has no semantic binding: {kind}")


def quantize(value, readout):
    if readout["rounding"] != "nearest-even" or readout["clamp"] != [-128, 127]:
        raise ValueError("toy interpreter requires its original exact signed-i8 readout")
    value = round(Fraction(value * readout["numerator"], readout["denominator"]))
    return min(127, max(-128, value))


def run(contract, output, epochs):
    """Consume explicit toy contracts; return application and exact IR execution evidence."""
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Region

    from merlin.compile.scheduling.convolution_exploration import (
        ConvolutionRhsResidencyLaw,
        ConvolutionScheduleChoice,
        plan_convolution_rhs_residency,
        plan_convolution_schedule,
    )
    from merlin.compile.scheduling.convolution_output_groups import plan_rolling_convolution_output_group
    from merlin.compile.scheduling.dense_array_retention import DenseArrayRetentionAdmission
    from merlin.compile.scheduling.window_transfer import (
        ConvolutionComputeEpochLaw,
        ConvolutionInputRetentionLaw,
        ConvolutionLoadIssueLaw,
        ConvolutionReadoutEpochLaw,
        ConvolutionReadoutTransferLaw,
        ConvolutionResidentInputRows,
    )
    from merlin.targetgen.isa_dialect_gen import generate_package
    from merlin.targetgen.isa_spec import Argument, EncodingField, Instruction, IsaType, ResolvedISA

    if contract["schema"] != "merlin.toy_convolution_adapter.v1" or contract["domain"] != "sequential-toy-only":
        raise ValueError("toy contracts cannot qualify a physical target")
    g, m, laws = contract["source"], contract["machine"], contract["laws"]
    if m["pending"] < 1 or not 1 <= len(contract["alternatives"]) <= 8:
        raise ValueError("toy execution requires bounded pending capacity and alternatives")
    b, h, w, ci, co = g["shape"]
    kh, kw = g["kernel"]
    if b*h*w*ci*co*kh*kw > 100_000:
        raise ValueError("use the ordinary numerical owner beyond the small toy example")
    caps = dict(array=(m["primitive"],)*2, operand=dict(rows=m["operand_rows"]),
        accumulator=dict(rows=m["accumulator_rows"]))
    W = epochs._load("conv_window_recipe")
    source = W.plan(caps, g)
    ids = (digest(m), digest(g["readout"]), source.identity, "sequential-toy-interpreter-only")
    schedule = plan_convolution_schedule(source,
        ConvolutionScheduleChoice(None, True, laws["compute"]["max_compute_pairs"]),
        retention_law=ConvolutionInputRetentionLaw(*ids, **laws["retention"]),
        issue_law=ConvolutionLoadIssueLaw(*ids, **laws["issue"]),
        epoch_law=ConvolutionComputeEpochLaw(*ids, **laws["compute"]))
    rhs = plan_convolution_rhs_residency(schedule, operand_rows=m["operand_rows"],
        rhs_base=m["rhs_base"], law=ConvolutionRhsResidencyLaw(*ids, **laws["rhs"]))
    inputs = ConvolutionResidentInputRows(source.plan.window, source.identity, m["primitive"],
        m["input_base"], m["row_pitch"], m["slots"], m["input_base"], m["input_private_rows"],
        m["stage_rows"], halo_value=0 if any(g["pad"]) else None)
    readout = ConvolutionReadoutEpochLaw(*ids, **laws["readout"])
    array = DenseArrayRetentionAdmission(*ids[:2], **laws["array"])
    transfer = ConvolutionReadoutTransferLaw(*ids[:2], source.plan.identity, ids[3],
        column_states=((0, co, digest(g["readout"])),), **laws["transfer"])
    if transfer.block_rows != source.plan.primitive:
        raise ValueError("toy interpreter needs its declared primitive block map")
    selected, schemas, semantics = [], {}, {}
    names = [choice["name"] for choice in contract["alternatives"]]
    if len(set(names)) != len(names) or any(not name.isascii() or not name.isidentifier() for name in names):
        raise ValueError("toy alternatives need unique identifier names")
    for choice in contract["alternatives"]:
        group = (plan_rolling_convolution_output_group(schedule, rhs, inputs,
            max_row_blocks=choice["max_row_blocks"], max_output_rows=choice["max_output_rows"])
            if choice["group"] else None)
        if choice["group"] and group is None:
            selected.append((choice, None))  # retain the incumbent, no claimed application
            continue
        options = dict(rhs=rhs, input_rows=inputs, readout_law=readout, output_group=group,
            array_law=array if group else None, grouped_transport=choice["grouped_transport"],
            readout_transfer_law=transfer if choice["wide_readout"] else None)
        def record(prefix, kind, fields):
            name = choice["name"] + "_" + prefix + kind
            names = tuple(sorted(fields))
            if name in schemas and schemas[name] != names:
                raise ValueError("toy effect signature changed")
            schemas[name] = names
            semantics[name] = (kind, names, prefix == "complete_")
        epochs.emit_checked_readout_epoch(schedule, ScfControl(), lambda k, f: record("effect_", k, f),
            complete=lambda k, f: record("complete_", k, f), **options)
        selected.append((choice, options))
    # Synthetic discriminators are not physical opcodes. SSA fields remain values,
    # never assumed register numbers or a fabricated target encoding allocation.
    argtype = IsaType("integer", 64, True, None)
    instructions = tuple(Instruction(name, tuple(Argument(n, argtype, "scalar", 1) for n in names),
        (), (), (EncodingField(0, 8, index, None),), '{"status":"unknown"}', '{"status":"unknown"}')
        for index, (name, names) in enumerate(sorted(schemas.items())))
    model = ResolvedISA("toy_effects", "toy_effects", 8, digest(m), digest(schemas), instructions)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    generation = generate_package(model, directory / "generated-isa")
    import importlib.util
    spec = importlib.util.spec_from_file_location("toy_generated_instructions",
        directory / "generated-isa/instruction_dialect.py")
    generated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generated)
    image = [n % 7 - 3 for n in range(b*h*w*ci)]
    weights = [[(k+c) % 5 - 2 for c in range(co)] for k in range(kh*kw*ci)]
    expected = {key: quantize(value, g["readout"]) for key, value in W.reference(source, image, weights).items()}
    alternatives = []
    for choice, options in selected:
        if options is None:
            alternatives.append(dict(name=choice["name"], status="retain_incumbent"))
            continue
        control = ScfControl()
        tables = {prefix: {name[len(choice["name"])+len(prefix)+1:]: (name, {field: field for field in names})
            for name, names in schemas.items() if name.startswith(choice["name"] + "_" + prefix)}
            for prefix in ("effect_", "complete_")}
        lower, complete = epochs.generated_effect_callbacks(generated.create, control.add, control.value,
            effects=tables["effect_"], completions=tables["complete_"])
        applied = epochs.emit_checked_readout_epoch(schedule, control, lower, complete=complete, **options)
        control.entry.add_op(func.ReturnOp())
        module = ModuleOp([func.FuncOp("example", ([], []), Region(control.entry))])
        module.verify()
        machine = ToyMachine(source, inputs, image, weights, g["readout"], m["pending"])
        execute_ir(module, generated, semantics, machine)
        if machine.pending or machine.output != expected:
            raise ValueError("generated SCF/ISA execution differs from complete original source")
        artifact = directory / (choice["name"] + ".target.mlir")
        artifact.write_text(str(module) + "\n")
        alternatives.append(dict(name=choice["name"], status="applied", application=applied["application"],
            consumer=applied["emission"], artifact=artifact.name,
            artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            exact_source_check=dict(status="PASS", elements=len(expected), scope="toy generated IR execution"),
            observed_effects=dict(machine.counts), cycles="UNKNOWN"))
    receipt = dict(schema="merlin.toy_convolution_adapter_receipt.v1", contract_sha256=digest(contract),
        source_sha256=source.identity, generated_isa=generation, alternatives=alternatives,
        target_qualification=False, selected_alternative=None, performance_claim=False)
    (directory / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


def execute_ir(module, generated, semantics, machine):
    """Use upstream SCF/arith execution and explicit toy instruction semantics."""
    from xdsl.dialects import arith
    from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
    from xdsl.interpreters.arith import ArithFunctions
    from xdsl.interpreters.func import FuncFunctions
    from xdsl.interpreters.scf import ScfFunctions

    # The upstream interpreter has no implementations for these integer ops.
    # All indices are checked i64 values; comparisons/loops use its stock code.
    methods = {
        "run_remui": impl(arith.RemUIOp)(lambda self, i, op, a: ((a[0] % 2**64) % (a[1] % 2**64),)),
        "run_divui": impl(arith.DivUIOp)(lambda self, i, op, a: ((a[0] % 2**64) // (a[1] % 2**64),)),
        "run_extui": impl(arith.ExtUIOp)(lambda self, i, op, a: (a[0] & 1,)),
    }
    for name, (kind, fields, completion) in semantics.items():
        def bind(instruction, kind, names, completion):
            def execute(self, interpreter, op, args):
                machine.issue(kind, dict(zip(names, args, strict=True)), completion=completion)
                return ()
            return impl(generated.INSTRUCTIONS[instruction])(execute)
        methods["run_" + name] = bind(name, kind, fields, completion)
    implementations = register_impls(type("ToyInstructions", (InterpreterFunctions,), methods))()
    interpreter = Interpreter(module)
    for functions in (ArithFunctions(), ScfFunctions(), FuncFunctions(), implementations):
        interpreter.register_implementations(functions)
    interpreter.call_op("example", ())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
        epilog="Run conv_epoch_recipe.py --write-toy-contract FILE, then --adapter-contract FILE --output DIR.")
    parser.parse_args()
    parser.print_help()
