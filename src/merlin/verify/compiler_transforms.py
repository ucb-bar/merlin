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
ENV_UNROLL_DEPTH = "MERLIN_LOOP_UNROLL_DEPTH"
ENV_TILE_DEPTH = "MERLIN_LOOP_TILE_DEPTH"
ENV_ALLOCATOR = "MERLIN_BUFFER_REUSE_ALLOCATOR"
ENV_MACC = "MERLIN_BF16_F32ACC"
ALLOCATOR = "total_private_unobservable"
MACC = "bf16_f32acc_draft_contract"


@dataclass(frozen=True)
class Options:
    unroll_factor: int | None = None
    tile_size: int | None = None
    buffer_allocator: str | None = None
    memory_source: bool = False
    macc_f32acc: bool = False
    unroll_depth: int = 1
    tile_depth: int = 1

    def __post_init__(self):
        if type(self.memory_source) is not bool:
            raise ValueError("memory source selection requires an explicit boolean")
        if type(self.macc_f32acc) is not bool:
            raise ValueError("MACC transform selection requires an explicit boolean")
        if self.unroll_factor is not None and (type(self.unroll_factor) is not int or self.unroll_factor < 2):
            raise ValueError("formal unrolling requires an integer factor at least two")
        if self.tile_size is not None and (type(self.tile_size) is not int or self.tile_size < 2):
            raise ValueError("formal ordered tiling requires an integer size at least two")
        for name, depth, parameter in (
            ("unroll", self.unroll_depth, self.unroll_factor),
            ("tile", self.tile_depth, self.tile_size),
        ):
            if type(depth) is not int or depth < 1:
                raise ValueError(f"{name} depth requires a positive integer")
            if depth != 1 and parameter is None:
                raise ValueError(f"{name} depth requires its transformation parameter")
        if self.buffer_allocator is not None and (self.buffer_allocator != ALLOCATOR or self.unroll_factor is not None):
            raise ValueError("buffer reuse requires explicit total_private_unobservable allocator; no unroll")
        # Public resource policy, not a theorem bound. Strip mining emits a
        # constant-size body; actual signed-index safety is checked in Lean.
        if self.buffer_allocator is not None and self.tile_size is not None and self.tile_size > 65536:
            raise ValueError("private buffer tile resource policy permits sizes up to 65536")
        if self.buffer_allocator is not None and self.tile_depth != 1:
            raise ValueError("private buffer initializer tiling requires depth one")

    @property
    def enabled(self):
        return (
            self.unroll_factor is not None
            or self.tile_size is not None
            or self.buffer_allocator is not None
            or self.memory_source
            or self.macc_f32acc
        )


def from_environment():
    def integer(name):
        raw = os.environ.get(name, "").strip()
        if not raw:
            return None
        if not raw.isdecimal():
            raise ValueError(f"{name} requires a positive integer")
        return int(raw)

    macc = os.environ.get(ENV_MACC, "")
    if macc not in ("", "0", "1"):
        raise ValueError(f"{ENV_MACC} requires 0 or 1")
    return Options(
        integer(ENV_UNROLL),
        integer(ENV_TILE),
        os.environ.get(ENV_ALLOCATOR) or None,
        macc_f32acc=macc == "1",
        unroll_depth=integer(ENV_UNROLL_DEPTH) if os.environ.get(ENV_UNROLL_DEPTH, "").strip() else 1,
        tile_depth=integer(ENV_TILE_DEPTH) if os.environ.get(ENV_TILE_DEPTH, "").strip() else 1,
    )


def _at_depth(module, depth, parameter):
    """Select one actual nesting depth without allocating for an absent depth."""
    from xdsl.dialects.scf import ForOp

    for operation in module.walk():
        if type(operation) is not ForOp:
            continue
        nesting = 1
        parent = operation.parent_op()
        while parent is not None:
            nesting += type(parent) is ForOp
            parent = parent.parent_op()
        if nesting == depth:
            return (1,) * (depth - 1) + (parameter,)
    return ()


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
    if options.macc_f32acc:
        from merlin.frontends.linalg_mlir import parse_mlir_text

        module = parse_mlir_text(source)
    else:
        module = parse_ir(source)
    macc_changed, macc_unchanged = [], []
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if options.macc_f32acc:
        from merlin.llvmlower.passes_xdsl import lower_bf16_matmul_f32acc

        before_macc = ir_text(module)
        pass_validation.observe(MACC, module, lower_bf16_matmul_f32acc)
        module.verify()
        after_macc = ir_text(module)
        (directory / "macc-before.mlir").write_text(before_macc)
        (directory / "macc-after.mlir").write_text(after_macc)
        (macc_changed if before_macc != after_macc else macc_unchanged).append(MACC)
    buffers, names, identities = [ir_text(module)], [], []
    changing_buffers = [buffers[0]]
    validation = pass_validation.current()
    from .compiler_coverage import boundary
    from .compiler_profile import SourceOutsideProfile

    def loop_admission(raw, kind):
        from .compiler_loop_source_profile import admit_source

        return pass_validation.source_admission(
            kind,
            raw,
            predicate=lambda source: admit_source(
                source,
                kind=kind,
                factor=options.unroll_factor or 2,
                tile=options.tile_size or 3,
                depth=options.unroll_depth if kind == UNROLL else options.tile_depth,
                index_bits=index_bits,
            ),
        )

    selected_loops = [
        kind
        for kind, parameter in ((UNROLL, options.unroll_factor), (TILE, options.tile_size))
        if parameter is not None and pass_validation.selected(kind)
    ]
    initial_profiles = {}
    buffer_admission = None
    if options.buffer_allocator is not None:
        if pass_validation.selected(BUFFER) or (options.tile_size is not None and pass_validation.selected(TILE)):
            from .compiler_buffer_source_profile import admit_source

            buffer_admission = pass_validation.source_admission(
                BUFFER, buffers[0], predicate=lambda raw: admit_source(raw, options=options)
            )
    else:
        initial_profiles = {kind: loop_admission(buffers[0], kind) for kind in selected_loops}
    from .compiler_loop_source_profile import GENERIC_PROFILE, WORD_PROFILE

    grouped = len(selected_loops) == 2 and all(initial_profiles[kind] == WORD_PROFILE for kind in selected_loops)
    stages = []

    def run(kind, action):
        before = ir_text(module)
        admission = None
        if options.buffer_allocator is None and kind in initial_profiles:
            admission = initial_profiles[kind] if grouped or not stages else loop_admission(before, kind)
        try:
            action()
            module.verify()
        except BaseException:
            excluded = buffer_admission if options.buffer_allocator is not None else admission
            if isinstance(excluded, SourceOutsideProfile):
                boundary(kind, before, None, profile_id=excluded.profile_id, exclusion=excluded)
            raise
        after = ir_text(module)
        stages.append((kind, before, after, admission))
        if before != after:
            names.append(kind)
            changing_buffers.append(after)
        else:
            identities.append(kind)
        buffers.append(after)

    if options.buffer_allocator is not None:
        run("shape_view_normalization", lambda: normalization.normalize_shapes(module))
        run(BUFFER, lambda: planning.reuse_storage(module, planning.plan_storage(module)))
        if options.tile_size is not None:
            run(TILE, lambda: loops.segment_loops(module, (options.tile_size,), index_bits=64))
        required = tuple(name for name in names if name in (BUFFER, TILE))
        if names and not required:
            # Normalization is part of this selected storage bundle even when
            # reuse is identity and no initializer tiling was requested.
            required = (BUFFER,)
    else:
        if options.unroll_factor is not None:
            run(
                UNROLL,
                lambda: unrolling.unroll_ordered_loops(
                    module, _at_depth(module, options.unroll_depth, options.unroll_factor), index_bits=64
                ),
            )
        if options.tile_size is not None:
            run(
                TILE,
                lambda: loops.segment_loops(
                    module, _at_depth(module, options.tile_depth, options.tile_size), index_bits=64
                ),
            )
        required = tuple(names)
    for i, text in enumerate(buffers):
        (directory / f"stage{i}.mlir").write_text(text)
    selected = [name for name in required if pass_validation.selected(name)]

    def enqueue(check, proof_buffers, kinds, selected_names, admission):
        label = "+".join(selected_names)
        if isinstance(admission, SourceOutsideProfile):
            boundary(label, proof_buffers[0], proof_buffers[-1], profile_id=admission.profile_id, exclusion=admission)
            return
        if proof_buffers[0] == proof_buffers[-1]:
            boundary(label, proof_buffers[0], proof_buffers[-1], profile_id=admission)
            return
        pass_validation.require_canonical_handler(validation, check)
        job = validation.gate.submit(
            check,
            before=proof_buffers[0],
            after=proof_buffers[-1],
            label=label,
            params={
                "sdk_bin": str(validation.tools.sdk_bin),
                "index_bits": 64,
                "factor": options.unroll_factor or 2,
                "tile": options.tile_size or 3,
                "kinds": kinds,
                "selected": selected_names,
                "intermediates": proof_buffers[1:-1],
                "allocator": options.buffer_allocator,
            },
        )
        boundary(label, proof_buffers[0], proof_buffers[-1], profile_id=admission, job=job)

    if buffer_admission is not None:
        selected_storage = [
            name
            for name in (BUFFER, TILE)
            if pass_validation.selected(name) and (name != TILE or options.tile_size is not None)
        ]
        # The memory proof covers the complete actual storage chain, including
        # normalization and identity stages. It grants one whole-chain row.
        enqueue("compiler_buffer_refinement", buffers, names, selected_storage, buffer_admission)
    elif options.buffer_allocator is None and grouped and names:
        enqueue("compiler_loop_refinement", changing_buffers, names, selected, WORD_PROFILE)
        for kind, before, after, admission in stages:
            if kind in selected_loops and before == after:
                boundary(kind, before, after, profile_id=admission)
    elif options.buffer_allocator is None:
        for kind, before, after, admission in stages:
            if admission is not None:
                check = (
                    "compiler_loop_generic_refinement" if admission == GENERIC_PROFILE else "compiler_loop_refinement"
                )
                enqueue(check, [before, after], [kind], [kind], admission)
    report = {
        "schema": "merlin.compiler-transforms.v1",
        "changed": macc_changed + names,
        "unchanged": macc_unchanged + identities,
        "selected_checks": [name for name in macc_changed if pass_validation.selected(name)] + selected,
        "directory": str(directory),
        "buffer_allocator": options.buffer_allocator,
        "memory_source": options.memory_source or options.buffer_allocator is not None,
        "unroll_depth": options.unroll_depth,
        "tile_depth": options.tile_depth,
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


@dataclass(frozen=True)
class _TransformInputs:
    buffers: tuple[str, ...]
    paths: tuple[Path, ...]
    refs: tuple[dict, ...]
    request_ref: dict | None
    params_json: str


def _input_snapshot(job):
    import json

    from .configuration_elision_bridge import canonical, digest, file_ref
    from .gate_errors import GateUnavailable

    try:
        paths = tuple(Path(job[key]) for key in ("before", "after"))
        refs = tuple(file_ref(path) for path in paths)
        raw = tuple(path.read_bytes() for path in paths)
        if any(digest(data) != ref["sha256"] or len(data) != ref["bytes"] for data, ref in zip(raw, refs, strict=True)):
            raise ValueError("actual transform buffers differ from the captured references")
        before, after = (data.decode("utf-8") for data in raw)
        middle = tuple(job["params"]["intermediates"])
        params_json = canonical(job["params"])
        if any(type(value) is not str for value in middle):
            raise ValueError("literal actual intermediate buffers required")
        request_path = Path(job["directory"]) / "request.json"
        request_ref = None
        if request_path.exists():
            request_ref = file_ref(request_path)
            request_bytes = request_path.read_bytes()
            if digest(request_bytes) != request_ref["sha256"] or json.loads(request_bytes) != job:
                raise ValueError("actual request differs from the independently read transform parameters")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise GateUnavailable("transform inputs changed during independent read") from error
    return _TransformInputs((before, *middle, after), paths, refs, request_ref, params_json)


def _recheck_inputs(inputs, job):
    from .configuration_elision_bridge import canonical, file_ref
    from .gate_errors import GateUnavailable

    try:
        if (
            tuple(Path(job[key]) for key in ("before", "after")) != inputs.paths
            or canonical(job["params"]) != inputs.params_json
            or tuple(job["params"]["intermediates"]) != inputs.buffers[1:-1]
            or any(file_ref(path) != ref for path, ref in zip(inputs.paths, inputs.refs, strict=True))
            or (inputs.request_ref is not None and file_ref(inputs.request_ref["path"]) != inputs.request_ref)
        ):
            raise ValueError("original transform input or request changed")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise GateUnavailable("transform inputs changed after proof") from error


def loop_check(job):
    import json

    from .compiler_loop_intake import prepare
    from .gate_errors import GateRejected, GateUnavailable, GateUnsupported
    from .loop_transform_gate import NotEstablished, check, validate_completion
    from .loop_transform_intake import Unavailable, Unsupported

    params = job["params"]
    try:
        inputs = _input_snapshot(job)
        prepared = prepare(
            inputs.buffers,
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
        _recheck_inputs(inputs, job)
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
    inputs = _input_snapshot(job)
    try:
        prepared = prepare_chain(inputs.buffers, allocator=params["allocator"], index_bits=64)
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
    _recheck_inputs(inputs, job)
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
