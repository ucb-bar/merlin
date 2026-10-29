"""Consume checked mixed-region products in an ordinary target C build.

This helper accepts a ConnectedProgram from compile_connected_region. It does
not provide an instruction implementation, target pointer ABI or launch runtime.
An automatic author and an offline compiler use the same returned products.
The neutral byte-buffer execute() tests are separate from guest code generation.

For caller-owned, synchronous buffers, prefer the shared
merlin.compile.connected_host.emit_connected_host_c sequence builder below.
"""

from __future__ import annotations


def emit_borrowed_connected_c(program, *, emit_stage, symbol, backing_origins):
    """Use shared checked binding and sequence generation.

    emit_stage(invocation, symbol) returns ConnectedCStage. It implements the
    documented byte-view ABI using the actual invocation.product (dense) or
    invocation.images (pointwise), with explicit logical-window relocation and
    completion. No target implementation is supplied by this example.

    The returned receipt gives backing order, window offsets and every live
    output. The C entrypoint preflights all bindings before calling any stage.
    Its status is execution status, never an independent numerical verdict.
    """
    from merlin.compile.connected_host import emit_connected_host_c

    return emit_connected_host_c(
        program, emit_stage, symbol=symbol, backing_origins=backing_origins
    )


def emit_connected_c(program, *, instruction_set, emit_pointwise, emit_sequence):
    """Emit native dense functions, pointwise entrypoints and an explicit sequence.

    instruction_set(invocation) supplies the caller's target InstructionSet.
    emit_pointwise(invocation, symbol) emits C loading/launching the exact checked
    invocation.images. emit_sequence(program, entries) emits the caller's ordinary
    pointer binding, intermediate allocation and sequential completion; entries
    are (symbol, invocation) pairs in source order. invocation.bindings retains
    complete input/output windows and byte orders; program.region retains SSA
    live-ins/live-outs. No implementation or qualification is inferred.

    For an LLVM caller, use invocation.product.emit_llvm(builder, lower_call)
    on dense entries instead. Counted loops remain owned by the shared emitter;
    target callbacks supply instruction calls and runtime pointer bindings.
    """
    if not all(callable(callback) for callback in (instruction_set, emit_pointwise, emit_sequence)):
        raise ValueError("connected C emission requires explicit target instruction and runtime emitters")
    invocations = program.ordered_invocations()
    definitions, entries = [], []
    for index, invocation in enumerate(invocations):
        symbol = f"connected_stage_{index}"
        if invocation.images is None:
            source = invocation.product.emit_c(instruction_set(invocation), symbol=symbol)
        else:
            source = emit_pointwise(invocation, symbol)
        if not isinstance(source, str) or not source.strip():
            raise ValueError("target stage emitter returned no C entrypoint")
        definitions.append(source)
        entries.append((symbol, invocation))
        program.ordered_invocations()
    sequence = emit_sequence(program, tuple(entries))
    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError("target runtime emitter returned no bound launch sequence")
    program.ordered_invocations()
    return "\n\n".join((*definitions, sequence))
