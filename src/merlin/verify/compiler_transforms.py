"""Explicit source SCF/storage transforms at the ordinary compiler entry.

Transformation selection is separate from proof selection, allowing identical
gated/ungated compiler comparisons. All changed selected boundaries are held by
the existing owned speculative gate before the compiler returns its products.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .compiler_loop_intake import TILE, UNROLL

BUFFER = "buffer_reuse"
ENV_UNROLL = "MERLIN_LOOP_UNROLL_FACTOR"
ENV_TILE = "MERLIN_LOOP_TILE_SIZE"
ENV_ALLOCATOR = "MERLIN_BUFFER_REUSE_ALLOCATOR"
ALLOCATOR = "total_private_unobservable"


@dataclass(frozen=True)
class Options:
    unroll_factor: int | None = None
    tile_size: int | None = None
    buffer_allocator: str | None = None
    memory_source: bool = False

    def __post_init__(self):
        if type(self.memory_source) is not bool:
            raise ValueError("memory source selection requires an explicit boolean")
        if self.unroll_factor is not None and (type(self.unroll_factor) is not int or not 2 <= self.unroll_factor <= 4):
            raise ValueError("formal unrolling requires factor2..4")
        if self.tile_size is not None and (type(self.tile_size) is not int or not 2 <= self.tile_size <= 16):
            raise ValueError("formal ordered tiling requires size2..16")
        if self.buffer_allocator is not None and (
            self.buffer_allocator != ALLOCATOR or self.tile_size is None or self.unroll_factor is not None
        ):
            raise ValueError(
                "buffer reuse requires explicit total_private_unobservable allocator and tiling; no unroll"
            )

    @property
    def enabled(self):
        return (
            self.unroll_factor is not None
            or self.tile_size is not None
            or self.buffer_allocator is not None
            or self.memory_source
        )


def from_environment():
    def integer(name):
        raw = os.environ.get(name, "").strip()
        if not raw:
            return None
        if not raw.isdecimal():
            raise ValueError(f"{name} requires a positive integer")
        return int(raw)

    return Options(integer(ENV_UNROLL), integer(ENV_TILE), os.environ.get(ENV_ALLOCATOR) or None)


def apply(source, directory, *, options=None, index_bits=64):
    from merlin.transforms import loops, normalization, planning, unrolling
    from merlin.transforms.certificates import ir_text, parse_ir

    from . import pass_validation

    options = from_environment() if options is None else options
    if type(options) is not Options:
        raise ValueError("typed compiler transformation options required")
    if not options.enabled:
        return source, None
    if index_bits != 64:
        raise ValueError("formal source transformations require explicit index64")
    module = parse_ir(source)
    buffers, names, identities = [ir_text(module)], [], []
    changing_buffers = [buffers[0]]

    def run(kind, action):
        before = ir_text(module)
        action()
        module.verify()
        after = ir_text(module)
        if before != after:
            names.append(kind)
            changing_buffers.append(after)
        else:
            identities.append(kind)
        buffers.append(after)

    if options.buffer_allocator is not None:
        run("shape_view_normalization", lambda: normalization.normalize_shapes(module))
        run(BUFFER, lambda: planning.reuse_storage(module, planning.plan_storage(module)))
        run(TILE, lambda: loops.segment_loops(module, (options.tile_size,), index_bits=64))
        check_name = "compiler_buffer_refinement"
        required = tuple(name for name in names if name in (BUFFER, TILE))
    else:
        if options.unroll_factor is not None:
            run(UNROLL, lambda: unrolling.unroll_ordered_loops(module, (options.unroll_factor,), index_bits=64))
        if options.tile_size is not None:
            run(TILE, lambda: loops.segment_loops(module, (options.tile_size,), index_bits=64))
        check_name = "compiler_loop_refinement"
        required = tuple(names)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(buffers):
        (directory / f"stage{i}.mlir").write_text(text)
    validation = pass_validation.current()
    selected = [name for name in required if pass_validation.selected(name)]
    if selected and names:
        # The memory proof covers the whole actual4 chain. Scalar loop proofs
        # additionally retain all-entry adjacent theorems for each actual pass.
        # Exact identity stages preserve the complete scalar boundary bytes.
        proof_buffers = buffers if options.buffer_allocator is not None else changing_buffers
        validation.gate.submit(
            check_name,
            before=proof_buffers[0],
            after=proof_buffers[-1],
            label="+".join(selected),
            params={
                "sdk_bin": str(validation.tools.sdk_bin),
                "index_bits": 64,
                "factor": options.unroll_factor or 2,
                "tile": options.tile_size or 3,
                "kinds": names,
                "selected": selected,
                "intermediates": proof_buffers[1:-1],
                "allocator": options.buffer_allocator,
            },
        )
    report = {
        "schema": "merlin.compiler-transforms.v1",
        "changed": names,
        "unchanged": identities,
        "selected_checks": selected,
        "directory": str(directory),
        "buffer_allocator": options.buffer_allocator,
        "memory_source": options.memory_source or options.buffer_allocator is not None,
    }
    return buffers[-1], report


def memory_lowering_pipeline():
    """Explicit memref input already owns its alloc/dealloc operations.

    The tensor ownership-inference pipeline cannot accept those explicit frees.
    This route lowers the existing scalar/SCF/memref program directly; it does
    not rerun tensor bufferization or infer a second allocation lifetime.
    """
    return ",".join(
        (
            "canonicalize",
            "cse",
            "convert-scf-to-cf",
            "convert-math-to-llvm",
            "convert-arith-to-llvm",
            "convert-index-to-llvm",
            "finalize-memref-to-llvm",
            "convert-func-to-llvm",
            "convert-cf-to-llvm",
            "reconcile-unrealized-casts",
        )
    )


def _buffers(job):
    return (Path(job["before"]).read_text(), *job["params"]["intermediates"], Path(job["after"]).read_text())


def loop_check(job):
    import json

    from .compiler_loop_intake import prepare
    from .gate_errors import GateRejected, GateUnavailable, GateUnsupported
    from .loop_transform_gate import NotEstablished, check, validate_completion
    from .loop_transform_intake import Unavailable, Unsupported

    params = job["params"]
    try:
        prepared = prepare(
            _buffers(job),
            kinds=tuple(params["kinds"]),
            factor=params["factor"],
            tile=params["tile"],
            index_bits=params["index_bits"],
        )
        completion = check(
            prepared, lean=Path(params["sdk_bin"]) / "lean", destination=Path(job["directory"]) / "proof", timeout=300
        )
        validate_completion(completion, prepared)
        report = json.loads(Path(completion.report).read_text())
    except Unsupported as exc:
        raise GateUnsupported(str(exc)) from exc
    except NotEstablished as exc:
        raise GateRejected(str(exc)) from exc
    except Unavailable as exc:
        raise GateUnavailable(str(exc)) from exc
    return {
        "claim": report["relation"],
        "premises": report["premises"],
        "receipt": completion.report,
        "axioms": report["audit"]["axioms_used"],
        "boundaries": {
            name: "Merlin.GraphGateProfileAdjacent.complete" + ("2" if name == UNROLL else "3")
            for name in params["kinds"]
        },
        "native_fidelity": False,
        "general_tiling": False,
    }


def buffer_check(job):
    import json

    from .buffer_reuse_gate import check, validate_completion
    from .buffer_reuse_intake import ROOT, prepare_chain
    from .gate_errors import GateRejected, GateUnavailable, GateUnsupported

    params = job["params"]
    if params["index_bits"] != 64 or params["allocator"] != ALLOCATOR:
        raise GateUnsupported("explicit index64 total private allocator required")
    try:
        prepared = prepare_chain(_buffers(job), allocator=params["allocator"], index_bits=64)
    except (ValueError, KeyError, TypeError) as exc:
        raise GateUnsupported(str(exc)) from exc
    result = check(
        prepared, lean=Path(params["sdk_bin"]) / "lean", destination=Path(job["directory"]) / "proof", timeout=300
    )
    if result.status != "checked":
        error = GateRejected if result.status == "not_established" else GateUnavailable
        raise error(result.reason)
    validate_completion(result.completion, prepared)
    report = json.loads(Path(result.completion.report).read_text())
    return {
        "claim": prepared.relation,
        "premises": {"allocator": ALLOCATOR},
        "receipt": result.completion.report,
        "theorems": [ROOT],
        "axioms": report["audit"]["axioms_used"],
        "covered_chain": params["kinds"],
        "compiler_enforcement": True,
        "native_allocator_fidelity": False,
        "full_error_or_world_equivalence": False,
    }
