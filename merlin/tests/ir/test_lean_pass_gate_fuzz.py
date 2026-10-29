"""Fuzz the Lean pass gate with random tensor programs, against an independent numpy oracle.

Each seeded case generates a random dataflow program (elementwise ``linalg.generic`` over function
inputs, constants, per-channel dequantization in the capture's ``quant_ext`` form, shared values and
dead values) and checks two properties:

* completeness -- the gate accepts the real output of every validated pass on the program, or, when it
  does not, the real output is identical to this file's independent rendering of the expected
  output and that is refused too: a checker gap (recorded and reported) rather than a pass bug. Any
  other refusal is a failure;
* soundness -- at each checked boundary (production DCE/CSE and standalone numerical cases) a
  mutated pass output that the oracle can tell apart from the program on random inputs is never
  accepted. A mutation the oracle cannot tell apart is counted, not asserted.

Only DCE and CSE are registered compiler gates. Numerical cases below call the
mathematical checker directly for handoff coverage; they do not gate compilation.

The oracle evaluates the generator's own program representation in numpy; it shares no code with
the passes or the checker. ``MERLIN_FUZZ_SEED`` and ``MERLIN_FUZZ_CASES`` scale the run; a failure
names its seed and saves the program.
"""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from test_lean_pass_gate import job

from merlin.llvmlower import leading_cse
from merlin.llvmlower.passes_xdsl import (
    lower_quant_ext,
    parse_mlir_text,
    preprocess_text_textual,
    prune_dead_pure_tensor_ops,
)
from merlin.verify import gate_checks
from merlin.verify.gate_errors import GateError
from merlin.xdsl_dialects._common import text as module_text

SEED = int(os.environ.get("MERLIN_FUZZ_SEED", "20261028"))
CASES = int(os.environ.get("MERLIN_FUZZ_CASES", "3"))
R, C = 4, 6
F = f"tensor<{R}x{C}xf32>"
BINARY = ("arith.addf", "arith.subf", "arith.mulf", "arith.maximumf")
NUMPY = {"arith.addf": np.add, "arith.subf": np.subtract, "arith.mulf": np.multiply, "arith.maximumf": np.maximum}
IDENTITY = "affine_map<(d0, d1) -> (d0, d1)>"
COLUMN = "affine_map<(d0, d1) -> (d1)>"
SCALAR = "affine_map<(d0, d1) -> ()>"


@dataclass
class Node:
    kind: str  # "binary" | "negate" | "constant" | "dequant" (per-channel, i32) | "dequant_t" (per-tensor, i64)
    op: str = ""
    args: list[str] = field(default_factory=list)
    value: float = 0.0
    lowered: bool = False  # a dequant printed as its linalg.generic lowering
    subtract_reversed: bool = False  # lowering bug model: (zp - w)
    scale_added: bool = False  # lowering bug model: (w - zp) + scale
    low: int = -128  # dequant_t quant_min: PyTorch ignores it, captures carry -128 and -127


@dataclass
class Program:
    inputs: int
    nodes: dict[str, Node]
    order: list[str]
    outputs: list[str]

    def live(self) -> list[str]:
        needed, todo = set(), list(self.outputs)
        while todo:
            name = todo.pop()
            if name in needed or name not in self.nodes:
                continue
            needed.add(name)
            todo += self.nodes[name].args
        return [name for name in self.order if name in needed]


def generate(rnd: random.Random) -> Program:
    inputs = rnd.randint(2, 3)
    values = [f"x{i}" for i in range(inputs)]
    nodes, order = {}, []
    for index in range(rnd.randint(5, 9)):
        name = f"v{index}"
        roll = rnd.random()
        if roll < 0.15:
            node = Node("constant", value=rnd.choice([0.5, 1.5, -2.0, 3.0]))
        elif roll < 0.22:
            node = Node("dequant", args=["w", "s", "z"])
        elif roll < 0.3:
            node = Node("dequant_t", args=["w", "st", "zt"], low=rnd.choice([-128, -127]))
        elif roll < 0.4:
            node = Node("negate", args=[rnd.choice(values)])
        else:
            node = Node("binary", op=rnd.choice(BINARY), args=[rnd.choice(values), rnd.choice(values)])
        if rnd.random() < 0.2 and order:  # an exact duplicate of an earlier node
            node = copy.deepcopy(nodes[rnd.choice(order)])
        nodes[name] = node
        order.append(name)
        values.append(name)
    outputs = rnd.sample(order[len(order) // 2 :], k=min(2, len(order) - len(order) // 2))
    return Program(inputs, nodes, order, outputs)


def evaluate(program: Program, env: dict[str, np.ndarray]) -> list[np.ndarray]:
    vals = dict(env)
    for name in program.order:
        node = program.nodes[name]
        if node.kind == "constant":
            vals[name] = np.full((R, C), node.value, np.float32)
        elif node.kind == "negate":
            vals[name] = -vals[node.args[0]]
        elif node.kind == "binary":
            vals[name] = NUMPY[node.op](vals[node.args[0]], vals[node.args[1]]).astype(np.float32)
        elif node.kind == "dequant_t":
            # PyTorch v2.10.0 per-tensor: (input.to(f32) - zero_point) * scale, all in f32.
            wf, zf = vals["w"].astype(np.float32), np.float32(vals["zt"])
            difference = (zf - wf) if node.subtract_reversed else (wf - zf)
            vals[name] = (difference + vals["st"]) if node.scale_added else (difference * vals["st"])
        else:
            w = vals["w"].astype(np.int32)
            difference = (vals["z"] - w) if node.subtract_reversed else (w - vals["z"])
            converted = difference.astype(np.float32)
            vals[name] = (converted + vals["s"]) if node.scale_added else (converted * vals["s"])
    return [vals[o] for o in program.outputs]


def random_inputs(rnd: random.Random, program: Program) -> dict[str, np.ndarray]:
    env = {
        f"x{i}": np.asarray([[rnd.uniform(-2, 2) for _ in range(C)] for _ in range(R)], np.float32)
        for i in range(program.inputs)
    }
    env["w"] = np.asarray([[rnd.randint(-128, 127) for _ in range(C)] for _ in range(R)], np.int8)
    env["s"] = np.asarray([rnd.uniform(0.01, 0.5) for _ in range(C)], np.float32)
    env["z"] = np.asarray([rnd.randint(-5, 5) or 1 for _ in range(C)], np.int32)
    env["st"] = np.float32(rnd.uniform(0.01, 0.5))
    env["zt"] = np.int64(rnd.choice([rnd.randint(-5, 5) or 1, rnd.randint(2**31, 2**40)]))  # past i32 too
    return env


def distinguishable(rnd: random.Random, a: Program, b: Program) -> bool:
    for _ in range(4):
        env = random_inputs(rnd, a)
        if any(not np.array_equal(x, y) for x, y in zip(evaluate(a, env), evaluate(b, env), strict=True)):
            return True
    return False


def generic(name: str, ins: list[tuple[str, str, str]], body: list[str]) -> str:
    maps = ", ".join([m for _, _, m in ins] + [IDENTITY])
    names = ", ".join(v for v, _, _ in ins)
    types = ", ".join(t for _, t, _ in ins)
    args = ", ".join(
        f"%i{k}: {(t[t.rindex('x') + 1 : -1] if 'x' in t else t[len('tensor<') : -1]) if t.startswith('tensor') else t}"
        for k, (_, t, _) in enumerate(ins)
    )
    return "\n".join(
        [
            f"  %e_{name} = tensor.empty() : {F}",
            f'  %{name} = linalg.generic {{indexing_maps = [{maps}], iterator_types = ["parallel", "parallel"]}}'
            f" ins({names} : {types}) outs(%e_{name} : {F}) {{",
            f"  ^bb0({args}, %o: f32):",
            *(f"    {line}" for line in body),
            f"  }} -> {F}",
        ]
    )


def render(program: Program, *, dead_empty: bool = True) -> str:
    ref = {f"x{i}": f"%x{i}" for i in range(program.inputs)} | {
        "w": "%w",
        "s": "%s",
        "z": "%z",
        "st": "%st",
        "zt": "%zt",
    }
    lines = []
    for name in program.order:
        node = program.nodes[name]
        a = [ref.get(x, f"%{x}") for x in node.args]
        if node.kind == "constant":
            lines.append(f"  %{name} = arith.constant dense<{node.value}> : {F}")
        elif node.kind == "negate":
            lines.append(generic(name, [(a[0], F, IDENTITY)], ["%r = arith.negf %i0 : f32", "linalg.yield %r : f32"]))
        elif node.kind == "binary":
            lines.append(
                generic(
                    name,
                    [(a[0], F, IDENTITY), (a[1], F, IDENTITY)],
                    [f"%r = {node.op} %i0, %i1 : f32", "linalg.yield %r : f32"],
                )
            )
        elif node.kind == "dequant_t" and not node.lowered:
            lines.append(
                f'  %{name} = "quant_ext.dequantize_per_tensor"(%w, %st, %zt) '
                f"<{{quant_min = {node.low} : i64, quant_max = 127 : i64}}>"
                f" : (tensor<{R}x{C}xi8>, tensor<f32>, tensor<i64>) -> {F}"
            )
        elif node.kind == "dequant_t":
            difference = "%c = arith.subf %b, %a : f32" if node.subtract_reversed else "%c = arith.subf %a, %b : f32"
            scaled = "%d = arith.addf %c, %i1 : f32" if node.scale_added else "%d = arith.mulf %c, %i1 : f32"
            lines.append(
                generic(
                    name,
                    [
                        ("%w", f"tensor<{R}x{C}xi8>", IDENTITY),
                        ("%st", "tensor<f32>", SCALAR),
                        ("%zt", "tensor<i64>", SCALAR),
                    ],
                    [
                        "%a = arith.sitofp %i0 : i8 to f32",
                        "%b = arith.sitofp %i2 : i64 to f32",
                        difference,
                        scaled,
                        "linalg.yield %d : f32",
                    ],
                )
            )
        elif not node.lowered:
            lines.append(
                f'  %{name} = "quant_ext.dequantize_per_channel"(%w, %s, %z) <{{axis = 1 : i64, input_dtype = "i8"}}>'
                f" : (tensor<{R}x{C}xi8>, tensor<{C}xf32>, tensor<{C}xi32>) -> {F}"
            )
        else:
            difference = "%b = arith.subi %i2, %a : i32" if node.subtract_reversed else "%b = arith.subi %a, %i2 : i32"
            scaled = "%d = arith.addf %c, %i1 : f32" if node.scale_added else "%d = arith.mulf %c, %i1 : f32"
            lines.append(
                generic(
                    name,
                    [
                        ("%w", f"tensor<{R}x{C}xi8>", IDENTITY),
                        ("%s", f"tensor<{C}xf32>", COLUMN),
                        ("%z", f"tensor<{C}xi32>", COLUMN),
                    ],
                    [
                        "%a = arith.extsi %i0 : i8 to i32",
                        difference,
                        "%c = arith.sitofp %b : i32 to f32",
                        scaled,
                        "linalg.yield %d : f32",
                    ],
                )
            )
    if dead_empty:
        lines.append(f"  %unused = tensor.empty() : {F}")
    params = ", ".join(
        [f"%x{i}: {F}" for i in range(program.inputs)]
        + [f"%w: tensor<{R}x{C}xi8>", f"%s: tensor<{C}xf32>", f"%z: tensor<{C}xi32>"]
        + ["%st: tensor<f32>", "%zt: tensor<i64>"]
    )
    results = ", ".join(F for _ in program.outputs)
    returned = ", ".join(ref.get(o, f"%{o}") for o in program.outputs)
    return (
        f"func.func @forward({params}) -> ({results}) {{\n"
        + "\n".join(lines)
        + f"\n  return {returned} : {results}\n}}\n"
    )


def pruned(program: Program) -> Program:
    keep = program.live()
    return Program(program.inputs, {n: program.nodes[n] for n in keep}, keep, list(program.outputs))


def lowered(program: Program) -> Program:
    result = copy.deepcopy(program)
    for node in result.nodes.values():
        if node.kind in ("dequant", "dequant_t"):
            node.lowered = True
    return result


def deduplicated(program: Program) -> Program:
    """Every node that repeats an earlier one (same kind and arguments) replaced by the earlier one."""
    alias: dict[str, str] = {}
    seen: dict[tuple, str] = {}
    nodes, order = {}, []
    for name in program.order:
        node = copy.deepcopy(program.nodes[name])
        node.args = [alias.get(a, a) for a in node.args]
        key = (node.kind, node.op, tuple(node.args), node.value, node.lowered, node.low)
        if key in seen:
            alias[name] = seen[key]
            continue
        seen[key] = name
        nodes[name] = node
        order.append(name)
    return Program(program.inputs, nodes, order, [alias.get(o, o) for o in program.outputs])


def mutate(rnd: random.Random, program: Program) -> tuple[str, Program] | None:
    mutant = copy.deepcopy(program)
    live = mutant.live()
    if not live:
        return None
    name = rnd.choice(live)
    node = mutant.nodes[name]
    values = [f"x{i}" for i in range(mutant.inputs)] + [
        n for n in mutant.order if mutant.order.index(n) < mutant.order.index(name)
    ]
    if node.kind == "binary":
        choice = rnd.choice(["op", "swap", "rewire"])
        if choice == "op":
            node.op = rnd.choice([o for o in BINARY if o != node.op])
        elif choice == "swap":
            node.args.reverse()
        else:
            node.args[rnd.randrange(2)] = rnd.choice(values)
        return f"{choice} {name}", mutant
    if node.kind == "negate":
        node.args[0] = rnd.choice(values)
        return f"rewire {name}", mutant
    if node.kind == "constant":
        node.value += 1.0
        return f"constant {name}", mutant
    if node.lowered:
        if rnd.random() < 0.5:
            node.subtract_reversed = True
            return f"dequant subtraction reversed {name}", mutant
        node.scale_added = True
        return f"dequant scale added {name}", mutant
    return None


def check(tmp_path: Path, tools, label: str, before: str, after: str, tag: str) -> str:
    """'accepted' or the gate error's class name; never raises for a refusal."""
    directory = tmp_path / tag
    directory.mkdir()
    item = job(directory, before, after, tools)
    item["label"] = label
    try:
        gate_checks.tensor_refinement(item)
        return "accepted"
    except GateError as error:
        (directory / "refusal.txt").write_text(str(error))
        return type(error).__name__


def generic_text(source: str, transform=None) -> tuple[str, str]:
    module = parse_mlir_text(source)
    before = module_text(module, generic=True)
    if transform is not None:
        transform(module)
    return before, module_text(module, generic=True)


@pytest.mark.parametrize("case", range(CASES))
def test_fuzzed_programs(tmp_path, pass_gate_tools, case):
    tools = pass_gate_tools
    seed = SEED + case
    rnd = random.Random(seed)
    program = generate(rnd)
    source = render(program)
    (tmp_path / f"program-{seed}.mlir").write_text(source)
    where = f"seed {seed}, program saved at {tmp_path}"

    # Completeness: each real pass output is accepted, or it is exactly the independently rendered
    # expected output and the refusal is a checker gap that output shares.
    gaps = []

    def complete(label, before, real_after, reference_after, tag):
        if check(tmp_path, tools, label, before, real_after, tag) == "accepted":
            return
        reference = check(tmp_path, tools, label, before, reference_after, tag + "-reference")
        assert reference != "accepted", (
            f"{label}: the real output is refused but the expected one is accepted ({where})"
        )
        # Only a refusal of the expected output itself is a checker gap; otherwise a pass bug could hide.
        assert generic_text(real_after)[0] == generic_text(reference_after)[0], (
            f"{label}: the real output is refused and differs from the expected one ({where})"
        )
        gaps.append(label)

    prune_before, prune_after = generic_text(source, prune_dead_pure_tensor_ops)
    assert prune_before != prune_after, where
    complete(
        "prune_dead_pure_tensor_ops",
        prune_before,
        prune_after,
        generic_text(render(pruned(program), dead_empty=False))[0],
        "prune",
    )
    quant_source = render(pruned(program), dead_empty=False)
    quant_before, quant_after = generic_text(quant_source, lower_quant_ext)
    if quant_before != quant_after:
        complete(
            "lower_quant_ext", quant_before, quant_after, render(lowered(pruned(program)), dead_empty=False), "quant"
        )
    textual_after, _ = preprocess_text_textual(source)
    complete("preprocess_text_textual", source, textual_after, render(lowered(program)), "textual")
    # The pipeline's leading cse, by MLIR itself, on the whole-model route's preprocessed form.
    cse_source = render(lowered(program))
    (tmp_path / "cse-apply").mkdir()
    cse_after = leading_cse.apply(cse_source, "cse", tmp_path / "cse-apply", "cse")
    if cse_after != cse_source:
        complete("cse", cse_source, cse_after, render(deduplicated(lowered(program))), "cse")
    if gaps:
        (tmp_path / "checker-gaps.txt").write_text("\n".join(gaps))

    # Soundness: an output the oracle can tell apart is never accepted.
    counts = {"rejected": 0, "indistinguishable": 0}
    for boundary, before, expected, dead_empty in (
        ("prune", source, pruned(program), False),
        ("quant", quant_source, lowered(pruned(program)), False),
        ("textual", source, lowered(program), True),
        ("cse", cse_source, deduplicated(lowered(program)), True),
    ):
        for attempt in range(2):
            mutated = mutate(rnd, expected)
            if mutated is None:
                continue
            what, mutant = mutated
            verdict = check(
                tmp_path,
                tools,
                boundary,
                before,
                render(mutant, dead_empty=dead_empty),
                f"{boundary}-mutant-{attempt}",
            )
            if distinguishable(rnd, expected, mutant):
                assert verdict != "accepted", f"UNSOUND: {boundary} mutation '{what}' was accepted ({where})"
                counts["rejected"] += 1
            else:
                counts["indistinguishable"] += 1
    print(f"seed {seed}: {counts}, checker gaps: {gaps or 'none'}")


def test_deleting_a_dead_dequantization_is_proved(tmp_path, pass_gate_tools):
    """Found by fuzzing: erasing a dead modeled dequantization (ledger A9) is a proved refinement."""
    program = Program(
        2,
        {"v0": Node("binary", op="arith.addf", args=["x0", "x1"]), "v1": Node("dequant", args=["w", "s", "z"])},
        ["v0", "v1"],
        ["v0"],
    )
    before, after = generic_text(render(program, dead_empty=False), prune_dead_pure_tensor_ops)
    assert before != after
    assert check(tmp_path, pass_gate_tools, "prune_dead_pure_tensor_ops", before, after, "dead") == "accepted"


def test_a_dead_cone_before_live_code_is_proved(tmp_path, pass_gate_tools):
    """Found by fuzzing: a dead constant feeding a dead generic, placed before live code whose
    tensor.empty is identical to the dead one's, is proved whichever comes first."""
    add = lambda a, b: Node("binary", op="arith.addf", args=[a, b])  # noqa: E731
    nodes = {"v0": Node("constant", value=0.5), "v1": add("v0", "x0"), "v2": add("x0", "x1")}
    for order in (["v0", "v1", "v2"], ["v2", "v0", "v1"]):
        program = Program(2, nodes, order, ["v2"])
        before, after = generic_text(render(program, dead_empty=False), prune_dead_pure_tensor_ops)
        assert before != after
        verdict = check(tmp_path, pass_gate_tools, "prune_dead_pure_tensor_ops", before, after, "-".join(order))
        assert verdict == "accepted", order


def test_a_live_op_deleted_as_dead_is_still_rejected(tmp_path, pass_gate_tools):
    """The liveness-guided script only proposes erasures; deleting a used value is still refused."""
    add = lambda a, b: Node("binary", op="arith.addf", args=[a, b])  # noqa: E731
    program = Program(2, {"v0": add("x0", "x1"), "v1": add("v0", "x1")}, ["v0", "v1"], ["v1"])
    wrong = Program(2, {"v1": add("x0", "x1")}, ["v1"], ["v1"])  # v0 dropped, its use rewired
    before = generic_text(render(program, dead_empty=False))[0]
    after = generic_text(render(wrong, dead_empty=False))[0]
    assert check(tmp_path, pass_gate_tools, "prune_dead_pure_tensor_ops", before, after, "wrong") == "GateRejected"
