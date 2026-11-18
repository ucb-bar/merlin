"""Numerical boundary policy for the whole-model mesh reference executor.

The policy is target-neutral: the selected datapath binding supplies the operand
and accumulator formats and whether subnormal operands are flushed.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from merlin.compiler.partition.dispatch_program import _check_dispatch_definition
from merlin.compiler.partition.outline import OutlineError, _external_declarations
from merlin.runtime.dispatch.whole_model import (
    _classify_mesh_contraction,
    _classify_mesh_matmul,
    _compile_kernel_so,
    _eval_view,
    _has_contraction,
    _kernel_io,
    _oracle_unreachable,
    mesh_datapath,
)

# ---- dispatch_numeric --------------------------------------------------------------------------


def boundary_scale(x, fmt: str) -> float:
    """Power-of-two scale placing the largest finite operand inside ``fmt``."""
    from merlin.common import fp8_formats as FF

    mx = float(np.abs(x).max()) if x.size else 0.0
    if not (mx > 0.0) or not np.isfinite(mx):
        return 1.0
    _min_normal, max_finite = FF.normal_range(fmt)
    return float(2.0 ** math.ceil(math.log2(mx / max_finite)))


def _representability(x, fmt: str, *, flush: bool) -> dict:
    """Count subnormal, flushed and saturating values at the selected boundary."""
    from merlin.common import fp8_formats as FF

    min_normal, max_finite = FF.normal_range(fmt)
    ax = np.abs(np.asarray(x, dtype=np.float64))
    subnormal = int(np.count_nonzero((ax > 0.0) & (ax < min_normal)))
    return {
        "n_values": int(ax.size),
        "n_subnormal": subnormal,
        "n_flushed_to_zero": subnormal if flush else 0,
        "n_saturating": int(np.count_nonzero(ax > max_finite)),
    }


def float_boundary_operands(a, b, binding) -> tuple[list, list, float, dict]:
    """Scale float operands and report what their selected format cannot carry.

    Power-of-two scaling preserves mantissas. If the resulting reduction could
    overflow the selected accumulator, back off the scales in whole powers of
    two and retain the remaining loss in the representability record.
    """
    from merlin.common import fp8_formats as FF

    fmt = binding.operand_dtype
    af, bf = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    sa, sb = boundary_scale(af, fmt), boundary_scale(bf, fmt)

    ma = float(np.abs(af).max()) if af.size else 0.0
    mb = float(np.abs(bf).max()) if bf.size else 0.0
    try:
        _min_acc, max_acc = FF.normal_range(binding.accum_dtype)
    except KeyError:  # integer or unresolvable accumulator: no float cap
        max_acc = None
    if max_acc is not None and ma > 0.0 and mb > 0.0:
        k = max(1, int(af.shape[1])) if af.ndim == 2 else 1
        headroom = max_acc / 2.0
        worst = (ma / sa) * (mb / sb) * k
        if np.isfinite(worst) and worst > headroom:
            back = math.ceil(math.log2(worst / headroom))
            sa *= float(2.0 ** ((back + 1) // 2))
            sb *= float(2.0 ** (back // 2))

    qa, qb = af / sa, bf / sb
    flush = bool(getattr(binding, "subnormal_operand_flush", False))
    rec = {
        "operand_dtype": fmt,
        "scale_a": sa,
        "scale_b": sb,
        "a": _representability(qa, fmt, flush=flush),
        "b": _representability(qb, fmt, flush=flush),
    }
    return qa.tolist(), qb.tolist(), sa * sb, rec


# ---- dispatch_placement ------------------------------------------------------------------------
# Honor selected outline placement without assigning hardware execution roles.
#
# These are structural caller requirements, not an issuer of device capabilities,
# command acceptance/completion, storage ownership or synchronization semantics.


class DispatchRuntimeError(RuntimeError):
    pass


def _selection(row):
    if (
        type(row.index) is not int
        or type(row.symbol) is not str
        or not row.symbol
        or type(row.n_operands) is not int
        or type(row.result_types) is not list
        or any(type(value) is not str for value in row.result_types)
        or row.placement is not None
        and (type(row.placement) is not str or not row.placement.strip())
        or row.group is not None
        and (type(row.group) is not int or row.group < 0)
    ):
        raise DispatchRuntimeError("runtime dispatch has malformed original placement/ABI membership")
    return row.index, row.symbol, row.n_operands, tuple(row.result_types), row.placement, row.group


class DispatchPlacements:
    """Bind the original static calls/table and recheck the actual selected call.

    A repeated dynamic call uses its original static selection. A successful
    host dispatch may satisfy only an unspecified or explicitly host selection.
    The existing device runner's result supplies no hardware authority here.
    """

    def __init__(self, outlined, driver, *, host_only):
        self.outlined = outlined
        self.rows = {}
        try:
            declarations = _external_declarations(outlined.module, outlined.external_symbols)
            calls = [
                op for op in driver.walk() if op.name == "func.call" and op.callee.string_value() not in declarations
            ]
            if len(calls) != len(outlined.dispatches):
                raise DispatchRuntimeError("runtime dispatch table omitted or added original calls")
            for index, (call, row) in enumerate(zip(calls, outlined.dispatches, strict=True)):
                selected = _selection(row)
                if row.index != index or call.callee.string_value() != row.symbol:
                    raise DispatchRuntimeError("runtime dispatch table differs from original call order/symbol")
                # Copy the complete ordered result type slots; the caller's mutable
                # table cannot later remove the selected nonhost requirement.
                snapshot = replace(row, result_types=list(row.result_types))
                _check_dispatch_definition(outlined.module, call, snapshot)
                self.rows[call] = snapshot, selected
                if host_only:
                    self.require_host(call)
        except OutlineError as error:
            raise DispatchRuntimeError(str(error)) from error
        self.host_symbols = frozenset(row.symbol for row, _selected in self.rows.values() if row.placement == "host")

    def verify_call(self, call):
        if call not in self.rows:
            raise DispatchRuntimeError("runtime call has no original defined dispatch selection")
        row, selected = self.rows[call]
        if (
            row.index >= len(self.outlined.dispatches)
            or _selection(self.outlined.dispatches[row.index]) != selected
            or call.callee.string_value() != row.symbol
        ):
            raise DispatchRuntimeError("runtime original dispatch selection changed")
        try:
            _check_dispatch_definition(self.outlined.module, call, row)
        except OutlineError as error:
            raise DispatchRuntimeError(str(error)) from error
        return row.placement

    def is_device(self, call):
        return self.verify_call(call) not in (None, "host")

    def require_host(self, call):
        if self.is_device(call):
            row = self.rows[call][0]
            raise DispatchRuntimeError(
                f"dispatch {row.symbol!r} explicitly requires placement {row.placement!r}; "
                "host execution is unavailable"
            )


# ---- dispatch_execute --------------------------------------------------------------------------
# Outlined-kernel execution loop of the whole-model host reference executor.
#
# :func:`execute` walks an outlined model's ``@forward`` and evaluates each view op, scalar op and
# kernel call with the numeric boundary and placement policy selected for it.


def execute(
    outline_result,
    arg_arrays: list[np.ndarray],
    workdir: str | Path,
    entry: str = "forward",
    cache_dir: str | Path | None = None,
    tap=None,
    qinner: dict | None = None,
    kernel_backend: str | None = None,
    mesh_target: str | None = None,
    mesh_package: str | None = None,
    counters: dict | None = None,
    numeric_policy: dict | None = None,
) -> list[np.ndarray]:
    """Run the outlined model on bound arguments; return the driver's result arrays.

    ``cache_dir`` persists compiled kernel ``.so``s keyed by kernel-body hash, so repeated
    runs (debugging, the multi-step action loop) skip recompilation. ``tap(op, value)`` is
    an optional callback invoked after every node with its produced value — the per-op
    bisection hook.
    """
    import hashlib
    import platform

    from merlin.compiler.dialects.xdsl_context import text as to_text
    from merlin.compiler.lower.model.oracle import HostModel, compile_host, extract_kernel, host_scalar_result_dtype

    workdir = Path(workdir)
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    module = outline_result.module
    driver = next(op for op in module.walk() if op.name == "func.func" and op.sym_name.data == entry)
    block = driver.body.blocks[0]
    placements = DispatchPlacements(outline_result, driver, host_only=kernel_backend != "mesh")

    env: dict[int, Any] = {}
    for arg, arr in zip(block.args, arg_arrays):
        env[id(arg)] = arr

    cache: dict[str, Any] = {}  # kernel body text -> HostModel
    compiled: dict[str, Any] = {}  # symbol -> HostModel
    n_unique = 0

    # --- mesh routing: run each matmul kernel on the target accelerator MESH (real activations) ----------
    # Same structural classifier picks the plain matmul kernels; each is executed on the target's mesh
    # oracle via operand injection (run_matmul_on_mesh) with the LIVE activation/weight arrays, and the
    # mesh output threads forward — so the whole model runs with its matmul layers on the real systolic
    # array. A layer that does not already arrive in the mesh's own operand format is converted at the
    # boundary: quantized for an integer datapath (the drop is expected quantization error, covered by the
    # whole-model tolerance), cast for a float one. Which of those applies is DERIVED from the target.
    mesh_routes: dict[str, dict] = {}
    mesh_dp = None  # the target's CorpusBinding, whole (see mesh_datapath)
    # Kernel bodies by symbol, for the host-lane contraction count below. Built LAZILY and cached: the
    # `kfns` map is bound inside the `kernel_backend == "mesh"` branch, so it does not exist on a plain host run -- reaching for one there is a
    # NameError, not a missing count. One walk, and only if a host kernel actually executes.
    _host_kfn_cache: dict = {}

    def _host_kernel_fn(sym: str):
        if not _host_kfn_cache:
            _host_kfn_cache.update(
                {o.sym_name.data: o for o in module.walk() if o.name == "func.func" and "$kernel_" in o.sym_name.data}
            )
        return _host_kfn_cache.get(sym)

    mesh_counts: dict = counters if counters is not None else {}  # per-CALL, not
    native_architecture = {"machine": platform.machine(), "system": platform.system(), "basis": "executing_process"}
    if kernel_backend != "mesh":
        mesh_counts.setdefault("target_executed", False)
    # a module-global function attribute: run_suite grades capsules on a thread pool.
    if kernel_backend == "mesh":
        if not mesh_target:
            raise DispatchRuntimeError("kernel_backend='mesh' requires mesh_target=<target>")
        mesh_dp = mesh_datapath(mesh_target, numeric_policy=numeric_policy)
        grouped = {d.symbol: d for d in outline_result.dispatches if d.group is not None}
        # The host lane materializes f32; the mesh may also take its own format directly.
        _mesh_et = mesh_dp.mlir_dtype(mesh_dp.operand_dtype)
        _accept = ("f32", _mesh_et) if _mesh_et != "f32" else ("f32",)
        kfns = {
            op.sym_name.data: op for op in module.walk() if op.name == "func.func" and "$kernel_" in op.sym_name.data
        }
        for sym, kfn in kfns.items():
            # An explicitly host-placed group must not be promoted merely because its body happens
            # to match the standalone matmul classifier.
            if sym in placements.host_symbols:
                continue
            # a bare linalg.matmul, or the linalg.generic a quantization rewrite leaves behind
            route = _classify_mesh_matmul(kfn, _accept) or _classify_mesh_contraction(kfn, _accept)
            if route is not None:
                mesh_routes[sym] = route
        # The standalone mesh executor consumes two operands and returns their contraction. It has
        # no binding for an outlined group's additional stages (bias, activation, readout, etc.).
        # Never run an accelerator-placed group on the host by accident, or treat its contraction
        # alone as execution of the whole group. Only a contraction-only group may use that route.
        unsupported_groups = [
            d.symbol
            for d in grouped.values()
            if d.placement != "host" and (d.symbol not in mesh_routes or d.stages != ["contraction"])
        ]
        if unsupported_groups:
            raise DispatchRuntimeError(
                "accelerator-placed outlined groups have no complete mesh executor: "
                + ", ".join(sorted(unsupported_groups))
            )
        # Matmul-family kernels the classifier REJECTED (bias-fused, transposed, batched-generic,
        # non-f32/i8, operand computed in-kernel). They never reach the mesh branch and so are invisible
        # to `mesh_fell_back`; counting them keeps the coverage claim honest.
        _unrouted = sum(1 for sym, kfn in kfns.items() if sym not in mesh_routes and _has_contraction(kfn))
        # mesh_unavailable seeded to 0 alongside the rest: absent it reads as UNKNOWN downstream,
        # which is the right answer for "nobody could tell" but the wrong one for "none occurred".
        mesh_counts.update(
            mesh_ran=0,
            mesh_fell_back=0,
            mesh_unrouted_matmuls=_unrouted,
            mesh_routed=len(mesh_routes),
            mesh_unavailable=0,
            mesh_route_symbols=sorted(mesh_routes),
            dispatch_ledger=[],
        )
        execute.last_mesh_routed = len(mesh_routes)
        execute.mesh_ran = 0
        execute.mesh_fell_back = 0
        # A LAYER THE ORACLE COULD NOT MEASURE IS NOT A LAYER THE MESH REFUSED. Kept apart from
        # mesh_fell_back because the two license opposite conclusions: a refusal says something about the
        # backend's shape space, an unavailable oracle says only that our measurement did not finish.
        # Conflating them reported "the model fell back to the host" for a whole model whose every layer
        # the mesh runs correctly -- the simulator had simply timed out.
        execute.mesh_unavailable = 0
        execute.mesh_fallbacks = []  # per-layer reasons, so a fallback is actionable
        execute.mesh_unavailable_detail = []
        # Layers whose capacity_fit obligation the RUNTIME discharged on the backend's behalf. A
        # whole-model pass that needed this is a statement about runtime+backend, not about the backend.
        execute.mesh_capacity_fit_delegated = []
        # What the mesh's operand format could actually hold, summed over every layer it ran. The whole
        # atlas divergence was invisible because nobody counted this; a run now reports it. The SAME dict
        # rides this call's counters, so a reader takes it off the run's own result rather than off a
        # module-global attribute that a concurrent grade would clobber.
        execute.mesh_operand_repr = {
            "operand_dtype": mesh_dp.operand_dtype,
            "subnormal_operand_flush": bool(mesh_dp.subnormal_operand_flush),
            "n_values": 0,
            "n_subnormal": 0,
            "n_flushed_to_zero": 0,
            "n_saturating": 0,
        }
        if mesh_dp.integer:
            # Only the FLOAT boundary counts representability; an integer datapath saturates rather than
            # underflowing and takes the quantizing branch below. Say that, instead of shipping four
            # zeroes that read as a clean bill of health this check never issued.
            execute.mesh_operand_repr = {
                "operand_dtype": mesh_dp.operand_dtype,
                "applicable": False,
                "note": "integer datapath: operands are quantized to the mesh's width at the boundary; "
                "the subnormal/saturation accounting applies to a float datapath only",
            }
        mesh_counts["mesh_operand_repr"] = execute.mesh_operand_repr
        mesh_counts["mesh_fallbacks"] = execute.mesh_fallbacks
        # Ride this run's own counters, not a module-global attribute a concurrent grade would clobber.
        # Same list object, so the appends below land in both.
        mesh_counts["mesh_capacity_fit_delegated"] = execute.mesh_capacity_fit_delegated

    def _record_dispatch(symbol: str, lane: str, **evidence) -> None:
        """Record completed dynamic calls; the ledger grants no command/effect authority."""
        ledger = mesh_counts.setdefault("dispatch_ledger", [])
        ledger.append({"ordinal": len(ledger), "symbol": symbol, "lane": lane, "status": "pass", **evidence})

    def kernel_model(symbol: str):
        if symbol in compiled:
            return compiled[symbol]
        km = extract_kernel(module, symbol)
        key = to_text(km)
        model = cache.get(key)
        nonlocal n_unique
        if model is None:
            if cache_dir is not None:
                import shutil

                digest = hashlib.sha1(key.encode()).hexdigest()[:16]
                so = cache_dir / f"{digest}.so"
                if not so.is_file():
                    compile_host(km, workdir / symbol.replace("$", "_"))
                    shutil.copy2(workdir / symbol.replace("$", "_") / "model_host.so", so)
                model = HostModel.load(str(so), scalar_result_dtype=host_scalar_result_dtype(km))
            else:
                model = compile_host(km, workdir / symbol.replace("$", "_"))
            cache[key] = model
            n_unique += 1
        compiled[symbol] = model
        return model

    def precompile_parallel():
        """Compile all unique kernels concurrently into ``cache_dir`` before execution.

        Lazy per-call compilation is serial — for large models (pi05 ~4k kernels) the
        lowering+clang invocations dominate wall-clock. Each kernel is lowered+compiled in a
        separate PROCESS (xDSL's parser/lowering is not thread-safe — concurrent parses race;
        processes isolate it), then execution only *loads* the cached ``.so``s. The dedup +
        cache check runs in the main process (cheap); only the per-kernel lower+clang fans
        out. No-op without a cache_dir; ``MERLIN_COMPILE_WORKERS=1`` forces serial."""
        if cache_dir is None:
            return
        import concurrent.futures
        import os as _os

        from xdsl.dialects.builtin import ModuleOp, StringAttr

        # Pre-index kernel funcs by symbol in ONE walk -- extract_kernel walks the whole
        # module per call (O(module)), which is O(symbols x module) overall and dominates
        # wall-clock for large graphs (pi05 ~4k symbols x 13k ops). Lookup + clone instead.
        kfuncs = {
            op.sym_name.data: op for op in module.walk() if op.name == "func.func" and "$kernel_" in op.sym_name.data
        }

        def _ktext(sym: str) -> str:
            clone = kfuncs[sym].clone()
            clone.properties["sym_name"] = StringAttr(entry)
            clone.properties.pop("sym_visibility", None)
            return to_text(ModuleOp([clone]))

        todo: dict[str, tuple] = {}  # digest -> (kernel_text, symbol); deduped by body
        seen: set[str] = set()
        for op in driver.walk():
            if op.name != "func.call":
                continue
            sym, _ = _kernel_io(op)
            if sym in seen or sym not in kfuncs or placements.is_device(op):
                continue
            seen.add(sym)
            ktext = _ktext(sym)
            digest = hashlib.sha1(ktext.encode()).hexdigest()[:16]
            if digest not in todo and not (cache_dir / f"{digest}.so").is_file():
                todo[digest] = (ktext, sym)
        if not todo:
            return

        jobs = [
            (ktext, str(workdir / sym.replace("$", "_")), str(cache_dir / f"{digest}.so"))
            for digest, (ktext, sym) in todo.items()
        ]
        workers = int(_os.environ.get("MERLIN_COMPILE_WORKERS", "0")) or max(1, (_os.cpu_count() or 4))
        workers = min(len(jobs), workers)
        if workers <= 1:
            for j in jobs:
                _compile_kernel_so(j)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
                list(ex.map(_compile_kernel_so, jobs))

    precompile_parallel()

    from xdsl.dialects.builtin import TensorType

    from merlin.compiler.lower.model.oracle import ScalarArg

    def run_call(op):
        placements.verify_call(op)
        symbol, outs = _kernel_io(op)
        host_lane = "native_cpu"
        host_evidence: dict = {}
        mroute = mesh_routes.get(symbol)
        if mroute is not None:  # run this matmul LAYER on the target mesh (real data)
            from merlin.runtime.dispatch.mesh import run_matmul_on_mesh

            a = np.ascontiguousarray(env[id(op.operands[mroute["a"]])])
            b = np.ascontiguousarray(env[id(op.operands[mroute["b"]])])
            mesh_out = None
            _obs: dict = {}
            binding = mesh_dp or mesh_datapath(mesh_target, numeric_policy=numeric_policy)
            op_dt, acc_dt, integer = binding.operand_dtype, binding.accum_dtype, bool(binding.integer)
            mesh_et = binding.mlir_dtype(op_dt)
            if a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]:  # a plain 2-D contraction
                if mroute["in_dtype"] == mesh_et:  # already in the mesh's format — no boundary convert
                    qa, qb, scale = a.tolist(), b.tolist(), 1.0
                elif integer:
                    # f32 -> per-tensor symmetric integer at the mesh boundary. The clip bound comes from
                    # the datapath's own width, not from a literal 127: a narrower or wider integer mesh
                    # would otherwise be fed operands saturated to some other unit's range.
                    from merlin.development.targetgen.grading.golden_inputs import dtype_bits as _dbits

                    _bits = _dbits(op_dt)
                    lim = float(2 ** (_bits - 1) - 1)
                    af, bf = a.astype(np.float64), b.astype(np.float64)
                    sa = float(np.abs(af).max()) / lim or 1.0
                    sb = float(np.abs(bf).max()) / lim or 1.0
                    qa = np.clip(np.rint(af / sa), -lim, lim).astype(np.int64).tolist()
                    qb = np.clip(np.rint(bf / sb), -lim, lim).astype(np.int64).tolist()
                    scale = sa * sb
                else:
                    # Float datapath: the mesh takes the values themselves, but only after a power-of-two
                    # shift into the range its operand format can actually hold. Rounding TO the format is
                    # still the ORACLE's job -- doing that here would grade the target against our idea of
                    # its rounding rather than the hardware's -- and a power of two does not round: it
                    # moves the exponent and leaves every mantissa bit where it was.
                    qa, qb, scale, _repr = float_boundary_operands(a, b, binding)
                    _acc = mesh_counts.get("mesh_operand_repr")
                    if _acc is not None:
                        for side in ("a", "b"):
                            for key, val in _repr[side].items():
                                _acc[key] = _acc.get(key, 0) + val
                # `package` is the SUBMISSION under test. Without it the mesh path resolves the target's
                # DEFAULT package and quietly certifies a different compiler than the one being graded.
                # `observed` is this call's OWN record of what the mesh path did -- which tiler chose the
                # extent, why it declined, whether the runtime had to discharge a capacity_fit obligation
                # the backend owes. Read from the call rather than a module global, so a concurrent grade
                # cannot attribute one model's decline to another model's layer.
                # THE BACKEND UNDER TEST MUST REACH THE MESH. Without it this resolved to the
                # target's DEFAULT package -- which for a target that ships none is no oracle at all, so
                # every layer returned None with no verdict and was recorded as a host fallback. That is
                # how a model whose layers the mesh runs correctly reported "15 of 15 fell back": not a
                # refusal, not even a timeout, simply nothing to ask.
                mesh_out = run_matmul_on_mesh(
                    mesh_target,
                    qa,
                    qb,
                    operand_dtype=op_dt,
                    accum_dtype=acc_dt,
                    package=mesh_package,
                    observed=_obs,
                    numeric_policy=numeric_policy,
                )
                _cf = _obs.get("capacity_fit")
                if _cf is not None:
                    _d = getattr(execute, "mesh_capacity_fit_delegated", None)
                    if _d is not None and len(_d) < 64:
                        # WHICH tiler discharged it travels with the record. "the layer did not fit"
                        # and "the backend only built one tile" are different facts about the backend,
                        # and only the second says its shape space is uncovered -- so the score can
                        # report the backend's own coverage apart from runtime+backend.
                        _d.append(
                            {
                                "kernel": symbol,
                                "lhs": list(a.shape),
                                "rhs": list(b.shape),
                                "required_elems": _cf.get("required_elems"),
                                "capacity_elems": _cf.get("capacity_elems"),
                                # WHICH tiler chose the extent: a capacity fact read out of the RTL,
                                # or a probe that halved until the backend stopped refusing. Those are
                                # different provenance claims and the record used to say neither.
                                "tile_source": _cf.get("tile_source"),
                                # ...and WHY it tiled at all (working-set capacity vs a declared
                                # primitive tile). A separate question from which tiler ran, so both
                                # ride the record — neither answers the other.
                                "tiled_by": _cf.get("tiled_by"),
                            }
                        )
                if mesh_out is not None:
                    om = np.array(mesh_out, np.float64) * scale
                    env[id(op.results[0])] = om.reshape(outs[0][0]).astype(outs[0][1])
                    mesh_counts["mesh_ran"] = mesh_counts.get("mesh_ran", 0) + 1
                    execute.mesh_ran = getattr(execute, "mesh_ran", 0) + 1
                    _record_dispatch(
                        symbol,
                        "on_mesh",
                        selected_placement=placements.verify_call(op),
                        lhs_shape=list(a.shape),
                        rhs_shape=list(b.shape),
                        oracle_evidence=_obs.get("oracle_evidence"),
                        trace_check=_obs.get("trace_check"),
                        artifact_identity=_obs.get("artifact_identity"),
                        cert_run_id=_obs.get("cert_run_id"),
                        capacity_fit=_obs.get("capacity_fit"),
                    )
                    if tap is not None:
                        tap(op, [env[id(op.results[0])]])
                    return
            placements.require_host(op)
            # A layer the mesh could not run falls back to the host kernel and is NOT counted as
            # mesh-executed — honest accounting, never a faked mesh result. RECORD WHY. The count alone
            # says a model failed its must_accelerate gate without saying what to fix, which is the same
            # fail-silent shape the rest of this path was built to avoid: a whole-model capsule reported
            # "35 of 37 layers on the mesh, 2 fell back" and nothing anywhere named the two.
            mesh_counts["mesh_fell_back"] = mesh_counts.get("mesh_fell_back", 0) + 1
            if not (a.ndim == 2 and b.ndim == 2):
                _why = f"operand rank {a.ndim}x{b.ndim}: the mesh boundary takes a 2-D contraction"
            elif a.shape[1] != b.shape[0]:
                _why = f"inner dims disagree: {a.shape} @ {b.shape}"
            else:
                # The oracle path records its OWN refusal reason; prefer it, because "unsynthesizable at
                # this shape" and "the oracle was unreachable" are different repairs. THIS call's own
                # `observed` verdict comes first -- it belongs to this layer and nothing else can have
                # overwritten it -- then the module-global last-refusal note, then a generic message that
                # covers both causes and names neither.
                _recorded = (locals().get("_obs") or {}).get("decline")
                if not _recorded:
                    try:
                        from merlin.cli.compile_workload import _MESH_REFUSAL

                        _recorded = _MESH_REFUSAL.get("reason")
                    except Exception:  # noqa: BLE001
                        _recorded = None
                _why = (
                    f"mesh oracle declined {a.shape} @ {b.shape} ({op_dt}/{acc_dt}): {str(_recorded)[:300]}"
                    if _recorded
                    else f"mesh oracle returned no result for {a.shape} @ {b.shape} "
                    f"({op_dt}/{acc_dt}) — unsynthesizable at this shape, or the oracle "
                    f"was unreachable"
                )
            # SEPARATE "the oracle could not tell us" FROM "the mesh refused". The host kernel runs
            # either way -- the model must still compute -- but only the second is evidence about the
            # backend. `_oracle_unreachable` keys on the oracle's OWN reported cause, never on a guess.
            # Carried on the counters dict, not the module-global function attributes this check first
            # used: concurrent grades clobber those, and a model verdict now depends on the count.
            _decl = str((locals().get("_obs") or {}).get("decline") or "")
            if _oracle_unreachable(_decl):
                # charged above as a fallback; it is not one -- move it to its own bucket
                mesh_counts["mesh_fell_back"] = max(0, mesh_counts.get("mesh_fell_back", 0) - 1)
                mesh_counts["mesh_unavailable"] = mesh_counts.get("mesh_unavailable", 0) + 1
                _ud = mesh_counts.setdefault("mesh_unavailable_detail", [])
                if len(_ud) < 64:
                    _ud.append({"kernel": symbol, "lhs": list(a.shape), "rhs": list(b.shape), "reason": _why})
                _unavailable = True
                host_lane = "mesh_unavailable"
            else:
                _unavailable = False
                host_lane = "host_fallback"
            host_evidence = {"lhs_shape": list(a.shape), "rhs_shape": list(b.shape), "mesh_decline": _why}
            if not _unavailable:
                # Name the layer, not just the count. "4 layers fell back" gives a reader nothing to act on;
                # the SHAPES say whether the cause is one extent the backend cannot take or four different
                # ones, which is the difference between a tiling gap and a capability gap.
                try:
                    _sh = (
                        f"{a.shape[0]}x{a.shape[1]}x{b.shape[1]}"
                        if (a.ndim == 2 and b.ndim == 2)
                        else f"rank{a.ndim}x{b.ndim}"
                    )
                except Exception:  # noqa: BLE001
                    _sh = "unknown"
                mesh_counts.setdefault("mesh_fallback_shapes", []).append(f"{_sh}: {_why}")
                # Capture the operands HERE, where they are already numpy arrays, rather than deeper in the
                # oracle path where an earlier attempt to do it failed silently. A layer that fails inside a
                # model and runs standalone differs only in the values it was handed.
                try:
                    import tempfile as _tf

                    _n = len(mesh_counts["mesh_fallback_shapes"])
                    _p = Path(_tf.gettempdir()) / f"fallback_operands_{_n}.npz"
                    np.savez_compressed(_p, A=a, W=b)
                    mesh_counts.setdefault("mesh_fallback_operands", []).append(str(_p))
                except Exception as _oe:  # noqa: BLE001
                    mesh_counts.setdefault("mesh_fallback_operands", []).append(
                        f"capture failed: {type(_oe).__name__}: {_oe}"
                    )
                execute.mesh_fell_back = getattr(execute, "mesh_fell_back", 0) + 1
                _fb = getattr(execute, "mesh_fallbacks", None)
                if _fb is not None and len(_fb) < 64:  # bounded: a diagnostic, not a full trace
                    _fb.append({"kernel": symbol, "lhs": list(a.shape), "rhs": list(b.shape), "reason": _why})
        placements.require_host(op)
        model = kernel_model(symbol)
        scalar_slots = [index for index, r in enumerate(op.results) if not isinstance(r.type, TensorType)]
        expected_scalar = str(op.results[scalar_slots[0]].type) if len(scalar_slots) == 1 else None
        if len(scalar_slots) > 1 or getattr(model, "scalar_result_dtype", None) != expected_scalar:
            raise DispatchRuntimeError("host kernel scalar result ABI differs from the original call")
        args, keep = [], []
        for o in op.operands:
            if isinstance(o.type, TensorType):
                a = np.ascontiguousarray(env[id(o)])
                keep.append(a)
                args.append((a.ctypes.data, a.shape))
            else:
                args.append(ScalarArg(env[id(o)], str(o.type)))  # by-value scalar arg
        out_arrays = [np.zeros(sh, dt) for sh, dt in outs]
        args += [(o.ctypes.data, o.shape) for r, o in zip(op.results, out_arrays) if isinstance(r.type, TensorType)]
        # THE HOST LANE ACTUALLY EXECUTING, which is the one thing nothing counted. `lane_report`
        # corrected `on_mesh` against per-layer mesh accounting but had no equivalent for
        # `scalar_rvv_lane`, so a capsule could satisfy a required host lane on the strength of a
        # ROUTING PLAN that never ran -- the same defect ("a routing plan is not an execution") that
        # was fixed on the mesh side and left open on this one.
        #
        # Contractions are counted SEPARATELY on purpose: the host lane is populated overwhelmingly by
        # norms and activations, so a bare kernel count is satisfied by any model at all and would be
        # as hollow as the plan-only evidence it replaces.
        #
        # Into the per-call `counters` dict, never onto a module-global: run_suite grades on a thread
        # pool and concurrent grades clobber module attributes -- and this count now feeds a verdict.
        scalar_result = model(args)
        if scalar_slots:
            if scalar_result is None:
                raise DispatchRuntimeError("host kernel did not return its original scalar result")
            slot = scalar_slots[0]
            out_arrays[slot] = np.asarray(scalar_result, dtype=outs[slot][1])
        mesh_counts["native_host_executed"] = True
        mesh_counts["host_kernels_ran"] = mesh_counts.get("host_kernels_ran", 0) + 1
        _hostfn = _host_kernel_fn(symbol)
        if _hostfn is not None and _has_contraction(_hostfn):
            mesh_counts["host_contractions_ran"] = mesh_counts.get("host_contractions_ran", 0) + 1
        for r, o in zip(op.results, out_arrays):
            env[id(r)] = o
        _record_dispatch(
            symbol,
            host_lane,
            selected_placement=placements.verify_call(op),
            placement="host",
            executor="native_cpu",
            runtime_architecture=native_architecture,
            target_executed=False,
            **host_evidence,
        )
        if tap is not None:
            tap(op, out_arrays)

    def run_scf_for(op):
        lb, ub, step = (int(env[id(op.operands[i])]) for i in range(3))
        body = op.regions[0].blocks[0]
        ivar, iter_args = body.args[0], list(body.args[1:])
        acc = [env[id(o)] for o in op.operands[3:]]  # iter_args inits
        for i in range(lb, ub, step):
            env[id(ivar)] = i
            for a, v in zip(iter_args, acc):
                env[id(a)] = v
            acc = run_block(body) or acc  # scf.yield -> new acc
        for r, v in zip(op.results, acc):
            env[id(r)] = v

    def run_scf_if(op):
        cond = env[id(op.operands[0])]
        regions = op.regions
        region = regions[0] if cond else regions[1]
        if region.blocks:
            res = run_block(region.blocks[0])
            for r, v in zip(op.results, res or []):
                env[id(r)] = v

    def _dead_after(blk):
        """Map op-index -> [value-ids] whose LAST use is that op, so the driver can free
        intermediate tensors from ``env`` as soon as they're dead. Without this the numpy
        interpreter keeps every intermediate alive for the whole forward -- fatal for large
        graphs (pi05 ~13k ops). Region-nested uses count against the containing top-level op
        (``op.walk()`` recurses), so a value captured by an scf body stays live until the
        scf op. Disabled when ``tap`` is set (the bisection hook may inspect values later)."""
        last: dict[int, int] = {}
        for i, op in enumerate(blk.ops):
            for sub in op.walk():
                for operand in sub.operands:
                    last[id(operand)] = i
        dead: dict[int, list[int]] = {}
        for vid, i in last.items():
            dead.setdefault(i, []).append(vid)
        return dead

    def run_block(blk, prune=False):
        """Execute a block's ops; returns operand values on a terminator (return/yield)."""
        dead = _dead_after(blk) if prune else {}
        for i, op in enumerate(blk.ops):
            if op.name in ("func.return", "scf.yield"):
                return [env[id(o)] for o in op.operands]
            if op.name == "func.call":
                run_call(op)
            elif op.name == "scf.for":
                run_scf_for(op)
            elif op.name == "scf.if":
                run_scf_if(op)
            else:
                val = _eval_view(op, env)
                # Bind torchao subclass inner tensors (int_data/scale) that m2m elided to an
                # uninitialized empty tagged prov.quant_inner -> the real quantized data.
                if qinner and op.name == "tensor.empty":
                    qi = op.attributes.get("prov.quant_inner")
                    key = getattr(qi, "data", None)
                    if key is not None and key in qinner:
                        val = np.ascontiguousarray(qinner[key])
                env[id(op.results[0])] = val
                if tap is not None:
                    tap(op, val)
            for vid in dead.get(i, ()):  # free now-dead intermediates
                env.pop(vid, None)
        return None

    results = run_block(block, prune=(tap is None))
    execute.last_unique_kernels = n_unique
    if results is None:
        raise DispatchRuntimeError("driver had no func.return")
    return results
