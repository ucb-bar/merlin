"""Lower a validated session into the existing bounded C dispatch pipeline.

Each selected stage is already compiled. Recurrent edges become distinct
buffer versions, so the ordinary lifetime allocator can reuse their storage.
This forms host orchestration, not a fused device program or a model evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from merlin.common.yaml import load_yaml
from merlin.frontends.linalg_mlir import parse_mlir_file
from merlin.llvmlower.session_bundle import MultiProgramSession, Program, load
from merlin.runtime.program import build_program
from merlin.xdsl_dialects.lowering.arena_plan import ArenaRequirements
from merlin.xdsl_dialects.lowering.dispatch_program import Buffer, DispatchProgram, Node
from merlin.xdsl_dialects.lowering.global_plan_emission import dispatch_digest

from .dispatch_c import DispatchCall, emit_program_c


@dataclass(frozen=True)
class SessionImplementation:
    """A synchronous wrapper's ordered ports, identified by capture argument index.

    Weights and constants may be bound in its context. The caller must verify
    this selection against the stage compiler's public interface before use.
    The wrapper uses the normal DispatchCall C ABI.
    """

    symbol: str
    input_arguments: tuple[int, ...]
    placement: str


def _identity(program: Program) -> tuple[int, ...]:
    from xdsl.dialects import func

    module = parse_mlir_file(program.bundle / "model.mlir")
    module.verify()
    functions = [op for op in module.body.block.ops if isinstance(op, func.FuncOp)]
    if (
        len(tuple(module.body.block.ops)) != 1 or len(functions) != 1
        or functions[0].sym_name.data != "forward" or len(functions[0].body.blocks) != 1
    ):
        raise ValueError("an uncompiled stage must be a proved tensor identity")
    function = functions[0]
    operations = tuple(function.body.block.ops)
    if len(operations) != 1 or not isinstance(operations[0], func.ReturnOp):
        raise ValueError("an uncompiled stage must contain only a return")
    arguments = tuple(function.args)
    if any(value not in arguments for value in operations[0].arguments):
        raise ValueError("identity stage returns a value other than its arguments")
    return tuple(arguments.index(value) for value in operations[0].arguments)


def _updates(program: Program) -> tuple[tuple[str, int, int], ...]:
    path = program.bundle / "session_contract.yaml"
    child = load_yaml(path) if path.is_file() else {}
    result = []
    for state in child.get("states", ()):
        incoming, outgoing = state.get("input_arg"), state.get("output_index")
        if (
            type(incoming) is not int or not 0 <= incoming < len(program.inputs)
            or type(outgoing) is not int or not 0 <= outgoing < len(program.outputs)
            or program.inputs[incoming] != program.outputs[outgoing]
        ):
            raise ValueError("recurrent state needs matching input and output tensor types")
        result.append((state["name"], incoming, outgoing))
    if len({row[0] for row in result}) != len(result) or len({row[1] for row in result}) != len(result):
        raise ValueError("recurrent state names and input ports must be distinct")
    return tuple(result)


def emit_session_dispatch(
    session: MultiProgramSession,
    implementations: dict[str, SessionImplementation],
    *,
    name: str,
    capacity: int,
    max_invocations: int = 4096,
):
    """Emit a C/H driver using checked stage order, state edges and arena planning.

    Missing implementations are accepted only for a structurally proved identity.
    Calls are synchronous, and repeated stages reuse the same wrapper/context.
    Final outputs include the last stage's results followed by carried states.
    Error recovery and buffer ownership follow emit_program_c; no device or
    reference implementation is supplied by this orchestration transform.
    """
    if (
        type(max_invocations) is not int or max_invocations < 1
        or sum(p.steps for p in session.programs) > max_invocations
        or not set(implementations) <= set(session.program_names)
    ):
        raise ValueError("session exceeds the declared invocation budget or names an unknown stage")
    if session != load(session.root, runtime_inputs=False):
        raise ValueError("session descriptor or tensor signatures changed after loading")
    buffers, nodes, calls, final, states, contexts = {}, [], {}, {}, {}, []
    initial, argument_count = {}, 0
    incoming = {(b.target_program, b.target_input): (b.source_program, b.source_output) for b in session.bindings}

    def buffer(key, signature, kind, argument=None):
        shape, dtype = signature
        buffers[key] = Buffer(key, list(shape), dtype, kind, argument)
        return key

    for stage_index, stage in enumerate(session.programs):
        implementation = implementations.get(stage.name)
        forwarded = None if implementation else _identity(stage)
        ports = implementation.input_arguments if implementation else tuple(dict.fromkeys(forwarded))
        if (
            len(set(ports)) != len(ports)
            or any(type(i) is not int or not 0 <= i < len(stage.inputs) for i in ports)
        ):
            raise ValueError("compiled stage input ports differ from its captured signature")
        updates = _updates(stage)
        if not {row[1] for row in updates} <= set(ports):
            raise ValueError("compiled stage omits a recurrent input")
        arguments = {}
        for port in ports:
            edge = incoming.get((stage.name, port))
            if edge is not None:
                arguments[port] = final[edge[0]][edge[1]]
            else:
                key = f"stage_{stage_index}_arg_{port}"
                arguments[port] = buffer(key, stage.inputs[port], "arg", argument_count)
                initial[key] = {"program": stage.name, "input_arg": port}
                argument_count += 1
        if implementation:
            context_index = len(contexts)
            contexts.append({"index": context_index, "program": stage.name, "symbol": implementation.symbol})
        for step in range(stage.steps):
            if forwarded is not None:
                outputs = tuple(arguments[i] for i in forwarded)
            else:
                outputs = tuple(
                    buffer(f"stage_{stage_index}_step_{step}_out_{i}", signature, "intermediate")
                    for i, signature in enumerate(stage.outputs)
                )
                calls[len(nodes)] = DispatchCall(
                    implementation.symbol, implementation.placement, context_index=context_index,
                )
                nodes.append(Node("dispatch", implementation.symbol, [arguments[i] for i in ports], list(outputs)))
            for state_name, port, output in updates:
                arguments[port] = outputs[output]
                states[state_name] = outputs[output]
        final[stage.name] = outputs
    results = [*final[session.program_names[-1]], *states.values()]
    graph = DispatchProgram(name, list(range(argument_count)), buffers, nodes, results)
    program = build_program(
        graph,
        capability="compiled_session",
        memory_requirements=ArenaRequirements(
            capacity, tuple(2 * i for i in range(len(nodes))), tuple(2 * i + 1 for i in range(len(nodes))),
            dispatch_digest=dispatch_digest(graph),
        ),
    )
    driver, files = emit_program_c(program, name, calls)
    description = {
        "driver": driver,
        "program": program.to_dict(),
        "inputs": initial,
        "stage_outputs": final,
        "states": states,
        "contexts": contexts,
        "stage_source_sha256": {
            stage.name: sha256((stage.bundle / "model.mlir").read_bytes()).hexdigest()
            for stage in session.programs
        },
        "execution": "not_run",
        "device_residency": "determined by compiled stage implementations",
    }
    return description, files
