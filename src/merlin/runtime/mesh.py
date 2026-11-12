"""Existing recursive/tiled mesh execution and original selected invocation hooks.

Certification algorithms live in verification.mesh and arithmetic references in
verification.mesh_reference. The compatibility boundary passes actual selected
callbacks; it does not replace globals in another owner or register new variants.
"""

from __future__ import annotations

from pathlib import Path as Path

from merlin.compiler.selection import mesh as _selection
from merlin.compiler.selection.capacity import (
    _accumulator_capacity_elems,
    _capacity_fit_tile,
    _operand_store_capacity_elems,
    capacity_fit,
    declared_primitive_tile,
)
from merlin.development.targetgen.operation_numerics import integer_split_k_limit
from merlin.verification.mesh_reference import _accum_rel_tolerance as _accum_rel_tolerance
from merlin.verification.mesh_reference import _reference_on_datapath as _reference_on_datapath

from .mesh_backend import (
    _MESH_RUN_SEQ as _MESH_RUN_SEQ,
)
from .mesh_backend import (
    _built_mesh_package as _built_mesh_package,
)
from .mesh_backend import (
    _mesh_invocation_id as _mesh_invocation_id,
)
from .mesh_backend import (
    _mesh_layer_id,
)
from .mesh_backend import (
    _refuse as _refuse,
)
from .mesh_backend import (
    _resolve_oot_mesh_simulator as _resolve_oot_mesh_simulator,
)

# Original invocation hooks are explicit bindings used by the bodies below.
# Independent source-lowering consumers use the actual selection owner directly.
_default_oot_package = _selection._default_oot_package
_mesh_tile_binding = _selection._mesh_tile_binding
tile_builder_op = _selection.tile_builder_op
_MESH_NTILE_WIDTH: dict[tuple, int] = {}


def run_matmul_on_mesh(
    target: str,
    A: list,
    W: list,
    *,
    operand_dtype: str | None = None,
    accum_dtype: str | None = None,
    numeric_policy: dict | None = None,
    simulator: str | None = None,
    package: str | None = None,
    epilogue: list | None = None,
    acc_scale: float | None = None,
    timeout: int = 900,
    observed: dict | None = None,
    _tiled: bool = False,
    _rescaled: bool = False,
) -> list | None:
    """Execute ``A @ W`` on the target's mesh oracle with the REAL operand values INJECTED (not
    materialized-from-name), and return the mesh's output tensor (nested list) — or ``None`` when this
    target has no reachable mesh path. ``A`` / ``W`` are the layer's real activations / weights.

    ``epilogue`` (e.g. ``["acc_scale"]``) applies the accumulator epilogue the layer commits with; an
    ``acc_scale`` epilogue re-quantizes the i32 accumulator to the target's narrow requant dtype (the
    target-declared ``requant_output_dtype``, e.g. i8 for an integer mesh) by the float ``acc_scale`` +
    saturating cast, so the layer's output is a same-dtype activation that can feed the NEXT mesh layer —
    this is what makes an int8 matmul CHAIN executable on the mesh end to end (the requant handoff).

    TARGET-AGNOSTIC: it dispatches on the target's DERIVED ``endpoint_kind`` (from the contract, never a
    literal) to that target's OWN oracle — a systolic/RoCC target (``inline_asm_insn``) through its generated
    OOT-package cert path, a self-hosted-ISA target (``external_backend``) through the program oracle whose
    RTL cosim mlc DERIVES from the target. The per-target kernel + harness come from the GENERATED package;
    this tool only builds the merlin_iface interface, injects the operands, and reads the output back.
    Fail-closed (``None``) on an unavailable oracle or an endpoint with no mesh path (never a fabricated
    result)."""
    from merlin.development.targetgen import corpus_spec as CS
    from merlin.development.targetgen.oracle_policy import _SIM_ORACLES
    from merlin.development.targetgen.oracle_policy import selected_endpoint as _endpoint_of
    from merlin.development.targetgen.oracle_policy import selected_sim_via as _bespoke_sim_via

    endpoint, model_ext = _endpoint_of(target)
    M, K, N = len(A), len(A[0]), len(W[0])
    # An acc_scale requant commits the accumulator back to the operand's narrow dtype so the layer's
    # output can feed the next mesh layer (the int8 chain handoff); tell the binding that narrow dtype.
    requant_out = (operand_dtype or "i8") if (epilogue and "acc_scale" in epilogue) else None
    binding = _mesh_tile_binding(
        target, operand_dtype, accum_dtype, requant_output_dtype=requant_out, numeric_policy=numeric_policy
    )

    # PAD TO THE MESH TILE EDGE. A generated package is entitled to reject a sub-tile extent, and a real
    # model is full of them: every matmul layer of an 8-token sequence has M=8 against a tile edge of 32
    # (atlas) or 16 (gemmini), so the mesh refused all 15 layers and the dispatch runtime silently fell
    # back to the host kernel for every one -- a whole model that reported "on mesh" while running
    # entirely on the CPU. Zero-padding is EXACT for a contraction: the padded rows of A and columns of W
    # contribute 0 to every retained output element, and the K padding multiplies zeros against zeros.
    # The result is sliced back to the true extent, so callers see the shape they asked for.
    _D = int(binding.tile_dim or 1)

    def _up(x):
        return ((int(x) + _D - 1) // _D) * _D if _D > 1 else int(x)

    _m_true, _n_true = M, N  # the extent the caller asked for, restored on the way out
    _Mp, _Kp, _Np = _up(M), _up(K), _up(N)
    _padded = (_Mp, _Kp, _Np) != (M, K, N)
    if _padded:
        import numpy as _np

        _A = _np.zeros((_Mp, _Kp), dtype=_np.float64)
        _A[:M, :K] = _np.asarray(A, dtype=_np.float64)
        _W = _np.zeros((_Kp, _Np), dtype=_np.float64)
        _W[:K, :N] = _np.asarray(W, dtype=_np.float64)
        A, W = _A.tolist(), _W.tolist()
        M, K, N = _Mp, _Kp, _Np

    def _build(m: int, k: int, n: int) -> str:
        e = {
            "name": "mesh_layer",
            "op": "matmul",
            "kind": "op",
            "source_role": "mesh_layer_real_operands",
            "source_reference": f"whole-model matmul layer {m}x{k}x{n} on the mesh with real operands",
            "M": m,
            "K": k,
            "N": n,
            "lhs": "A0",
            "weight": "W",
            "out": "Y0",
        }
        if epilogue:
            e["epilogue"] = list(epilogue)
            if acc_scale is not None:
                e["acc_scale"] = float(acc_scale)
        return CS.build(e, binding)[1]

    mlir = _build(M, K, N)

    # DISPATCH ORDER is deliberate (mirrors capsule_runner.oracle_adapters): a target whose contract
    # DECLARES an EXCLUSIVE bespoke sim (a self-hosted SIMT core graded on its own emitted kernel by its
    # own oracle, e.g. a cyclotron backend) is routed FIRST — its endpoint is ALSO external_backend,
    # but the arc command-buffer program oracle grades the WRONG artifact for a SIMT kernel, so the bespoke
    # executor must take precedence. Only when no exclusive sim is declared does the endpoint kind pick the
    # path. Derived from the contract's sim_via + the _SIM_ORACLES registry — never a target-name branch.
    def _unpad(out):
        """Slice the padded mesh result back to the extent the caller asked for."""
        if out is None or not _padded:
            return out
        return [row[:_n_true] for row in out[:_m_true]]

    so = _SIM_ORACLES.get(_bespoke_sim_via(target))

    def _dispatch(_mlir, _A, _W, *, _call_observed: dict | None = None) -> list | None:
        """One mesh call at whatever extent it is given."""
        _obs = observed if _call_observed is None else _call_observed
        if so is not None and so.exclusive:
            if _obs is not None:
                _obs["path"] = "bespoke_sim"
            return _matmul_via_bespoke_sim(
                target,
                _mlir,
                _A,
                _W,
                package=package,
                timeout=timeout,
                layer_id=_mesh_layer_id(len(_A), len(_A[0]), len(_W[0]), binding, epilogue, acc_scale),
                observed=_obs,
            )
        if endpoint in (None, "inline_asm_insn", "upstream_target"):
            if _obs is not None:
                _obs["path"] = "oot_cert"
            return _matmul_via_oot_cert(
                target,
                _mlir,
                _A,
                _W,
                simulator=simulator,
                package=package,
                layer_id=_mesh_layer_id(len(_A), len(_A[0]), len(_W[0]), binding, epilogue, acc_scale),
                timeout=timeout,
                observed=_obs,
            )
        if endpoint == "external_backend":
            if _obs is not None:
                _obs["path"] = "program_oracle"
            return _matmul_via_program_oracle(
                target,
                _mlir,
                _A,
                _W,
                model_ext=model_ext,
                package=package,
                timeout=timeout,
                operand_dtype=binding.operand_dtype,
                observed=_obs,
            )
        return None  # no mesh-execution path derived for this endpoint kind

    # BLOCK A LAYER THAT DOES NOT FIT ON CHIP, rather than declining it. The working set of a matmul is
    # the weight tile K*N plus the activation tile M*K; past the target's scratchpad the mesh returns
    # nothing and the whole layer falls back to the host. Measured on lstmnetvit/gemmini: K and N are each
    # fine alone (1x16x512 and 1x512x16 both run) and together they are not (1x512x512 declines), so two
    # of 37 layers fell back and the model failed its must_accelerate gate at 35/37.
    #
    # `_capacity_fit_tile` already computes the fitting extent from RTL-derived facts, and `_mesh_verify`
    # already uses it to shrink the tile it CERTIFIES. Only the execution path did not, so the tile record
    # said "runs at this shape" about a shape the model never got to run. Same tiler, both paths.
    #
    # Splitting is EXACT on an integer datapath: partial products over a K split sum in the i32
    # accumulator, and an N or M split is independent columns/rows. It is NOT applied when an epilogue is
    # declared -- an acc_scale requant must see the whole accumulation, not each K block -- so such a
    # layer still declines, now with that as the stated reason.
    _cap = _operand_store_capacity_elems(target, binding.operand_dtype)
    _mt, _kt, _nt = M, K, N
    _n_sub = 1
    _tiled_by = None
    try:
        _integer_limit = integer_split_k_limit(numeric_policy, tile_dim=_D, full_k=K) if binding.integer else None
    except ValueError as exc:
        if observed is not None:
            observed["decline"] = str(exc)
        return None
    if _integer_limit is not None:
        import numpy as _np

        _min_operand = -(1 << (_integer_limit["signed_operand_bits"] - 1))
        _max_operand = (1 << (_integer_limit["signed_operand_bits"] - 1)) - 1
        _lhs_values, _rhs_values = _np.asarray(A, dtype=_np.float64), _np.asarray(W, dtype=_np.float64)
        if any(
            not _np.isfinite(values).all()
            or not _np.array_equal(values, _np.rint(values))
            or _np.any(values < _min_operand)
            or _np.any(values > _max_operand)
            for values in (_lhs_values, _rhs_values)
        ):
            if observed is not None:
                observed["decline"] = "selected integer mesh operands are not integral signed-width values"
            return None
    if _integer_limit is not None and K > _integer_limit["max_padded_k"]:
        if _integer_limit["tile_k"] < _D:
            if observed is not None:
                observed["decline"] = "selected internal MAC width cannot safely execute one padded mesh tile"
            return None
        _kt = _integer_limit["tile_k"]
        _tiled_by = "internal_mac_width"
    _acc_cap = _accumulator_capacity_elems(target, binding.accum_dtype)
    if _cap:
        _cm, _ck, _cn, _n_sub = _capacity_fit_tile(M, K, N, max(1, _D), _cap, _acc_cap)
        _mt, _kt, _nt = _cm, min(_kt, _ck), _cn
        if (_cm, _ck, _cn) != (M, K, N):
            _tiled_by = "capacity_and_internal_mac_width" if _tiled_by else "capacity"
    else:
        # NO CLASSIFIABLE OPERAND STORE -> the capacity tiler cannot even be evaluated, and this branch
        # used to end there: the whole extent went to a backend that may implement one tile, which
        # answers with a program that writes nothing. Fall back to what the backend DECLARES it built.
        # Measured: on the target whose 39 SRAMs mlc declines to classify, `capacity_elems` is None, so
        # the residency tiler was structurally unreachable and every multi-tile layer failed there.
        _decl = declared_primitive_tile(package)
        if _decl:
            _dm, _dk, _dn = (max(1, min(v, e)) for v, e in zip(_decl, (M, K, N)))
            if (_dm, _dk, _dn) != (M, K, N):
                _mt, _kt, _nt = _dm, min(_kt, _dk), _dn
                _n_sub = (-(-M // _mt)) * (-(-K // _kt)) * (-(-N // _nt))
                _tiled_by = "declared_primitive_tile_and_internal_mac_width" if _tiled_by else "declared_primitive_tile"
    if _integer_limit is not None and ((_kt + _D - 1) // _D) * _D > _integer_limit["max_padded_k"]:
        _kt = (_integer_limit["max_padded_k"] // _D) * _D
        if _kt < 1:
            if observed is not None:
                observed["decline"] = "selected integer tile cannot meet both residency and padded MAC bounds"
            return None
    if (_mt, _kt, _nt) != (M, K, N):
        if epilogue:
            if observed is not None:
                _why = (
                    f"exceeds the on-chip working set ({_cap} elems)"
                    if _tiled_by == "capacity"
                    else f"exceeds the backend's declared primitive tile {list(_decl)}"
                )
                observed["decline"] = (
                    f"{M}x{K}x{N} {_why} and declares epilogue {list(epilogue)}; "
                    f"an accumulator epilogue cannot be split across K blocks"
                )
            return None
        import numpy as _np

        _dtype = _np.int64 if _integer_limit is not None else _np.float64
        _An, _Wn = _np.asarray(A, dtype=_dtype), _np.asarray(W, dtype=_dtype)
        _acc = _np.zeros((M, N), dtype=_np.int64 if _integer_limit is not None else _np.float64)
        _sub_mlir: dict[tuple[int, int, int], str] = {}
        _primitive_calls: list[dict] = []
        _n_sub = (-(-M // _mt)) * (-(-K // _kt)) * (-(-N // _nt))
        if observed is not None:
            # ATTRIBUTION. The runtime is discharging an obligation the contract places on the TARGET
            # BACKEND. Blocking here keeps whole-model work moving, but a result produced this way is
            # evidence about our runtime plus their backend -- not evidence that their backend handles
            # a layer this size. Unrecorded, it reads as the latter.
            # WHICH tiler engaged is part of the attribution, not a detail: "the layer did not fit"
            # and "the backend only built one tile" are different facts about the backend, and only the
            # second one says the shape space is uncovered.
            observed["blocked"] = {
                "tile": [_mt, _kt, _nt],
                "n_subtiles": _n_sub,
                "capacity_elems": _cap,
                "tiled_by": _tiled_by,
            }
            observed["capacity_fit"] = {
                **capacity_fit(target, M, K, N, binding.operand_dtype, _D, binding.accum_dtype),
                # WHICH TILER CHOSE THIS EXTENT. Derived here: the fitting extent was computed from
                # the target's own RTL-derived operand-store capacity. The other tiler below finds a
                # width by probing until the backend stops refusing, and "derived from a hardware
                # capacity fact" is a much stronger claim than "found by probing" -- the record used
                # to make neither, so a reader could not tell them apart.
                "tile_source": "capacity_fact",
                "discharged_by": "merlin runtime (host-side residency tiling)",
                # WHETHER THE SPLIT IS NUMERICALLY FREE depends on the accumulator, and it is not free
                # everywhere. Over an integer accumulator the K-partials sum exactly, so a blocked layer
                # is bit-identical to an unblocked one. Over a FLOAT accumulator it is not: the hardware
                # would reduce all K terms in its own format, while blocking reduces each K block there
                # and then sums the blocks on the host. That difference is charged to the runtime too,
                # so a float target's blocked layer is not quotable as "what the accelerator computes".
                "split_exact": bool(binding.integer),
                "accum_dtype": binding.accum_dtype,
                "tiled_by": _tiled_by,
                "note": (
                    "the target backend did not satisfy capacity_fit at this extent; the runtime "
                    "split the layer so it could run. A whole-model pass resting on this is a "
                    "statement about the runtime + backend together."
                    if _tiled_by == "capacity"
                    else "the backend DECLARED it implements a single "
                    f"{_mt}x{_kt}x{_nt} tile, so the runtime drove the loop nest over this larger "
                    "extent. The backend lowered one tile; the generalization over M/K/N is the "
                    "RUNTIME'S, and this result is NOT evidence that the backend generalizes."
                )
                + (
                    ""
                    if binding.integer
                    else f" This datapath accumulates in {binding.accum_dtype}, so the split also "
                    f"CHANGES THE REDUCTION ORDER (per-block on the device, cross-block on the "
                    f"host) and the layer's numerics are not what the device alone would give."
                ),
            }
            if _tiled_by == "internal_mac_width":
                # A numeric split does not discharge a capacity-fit obligation.
                observed.pop("capacity_fit", None)
        for m0 in range(0, M, _mt):
            for n0 in range(0, N, _nt):
                for k0 in range(0, K, _kt):
                    a = _An[m0 : m0 + _mt, k0 : k0 + _kt]
                    w = _Wn[k0 : k0 + _kt, n0 : n0 + _nt]
                    shp = (a.shape[0], a.shape[1], w.shape[1])
                    if shp not in _sub_mlir:
                        _sub_mlir[shp] = _build(*shp)
                    _call_obs: dict = {}
                    part = _dispatch(_sub_mlir[shp], a.tolist(), w.tolist(), _call_observed=_call_obs)
                    if part is None:
                        if observed is not None:
                            observed["decline"] = _call_obs.get("decline", "mesh primitive did not complete")
                        return None  # fail closed: a partial sum is not a result
                    _part = _np.asarray(part, dtype=_np.float64)
                    if _part.shape != (a.shape[0], w.shape[1]) or not _np.isfinite(_part).all():
                        if observed is not None:
                            observed["decline"] = "mesh primitive returned malformed partial-result shape or values"
                        return None
                    if _integer_limit is not None:
                        if not _np.array_equal(_part, _np.rint(_part)):
                            if observed is not None:
                                observed["decline"] = "integer mesh primitive returned a non-integral partial result"
                            return None
                        if (
                            _np.any(_part < -(1 << 31))
                            or _np.any(_part > (1 << 31) - 1)
                            or _np.any(_np.abs(_part) > shp[1] * _integer_limit["maximum_product"])
                        ):
                            if observed is not None:
                                observed["decline"] = "integer mesh primitive exceeds its exact i32/K bound"
                            return None
                        _part = _part.astype(_np.int64)
                    _acc[m0 : m0 + a.shape[0], n0 : n0 + w.shape[1]] += _part
                    if _integer_limit is not None:
                        _primitive_calls.append(
                            {
                                "offsets": [m0, k0, n0],
                                "shape": list(shp),
                                "worst_case_padded_shape": [_up(axis) for axis in shp],
                                "oracle_evidence": _call_obs.get("oracle_evidence"),
                                "trace_check": _call_obs.get("trace_check"),
                                "artifact_identity": _call_obs.get("artifact_identity"),
                                "cert_run_id": _call_obs.get("cert_run_id"),
                            }
                        )
                    if observed is not None:
                        observed.update(_call_obs)
        if _integer_limit is not None:
            if _np.any(_acc < -(1 << 31)) or _np.any(_acc > (1 << 31) - 1):
                if observed is not None:
                    observed["decline"] = "integer mesh split exceeded i32 after exact aggregation"
                return None
            if observed is not None:
                observed["integer_split_k"] = {
                    "schema": "merlin.integer_split_k.v1",
                    "padded_shape": [M, K, N],
                    "mesh_tile_dim": _D,
                    "tile": [_mt, _kt, _nt],
                    "max_padded_k_per_dispatch": _integer_limit["max_padded_k"],
                    "mac_result_bits": _integer_limit["mac_result_bits"],
                    "signed_operand_bits": _integer_limit["signed_operand_bits"],
                    "aggregation": "exact_i64_sum_checked_i32_before_epilogue",
                    "primitive_calls": _primitive_calls,
                }
        return _unpad(_acc.tolist())

    # EVALUATE THE OBLIGATION ON EVERY MESH PATH, not just one of them. This check used to sit inside
    # the RoCC/oot cert path, so a self-hosted-ISA target -- whose layers leave through the program
    # oracle -- had no obligation evaluated at all, and an oversized layer there declined with the same
    # uninformative "the oracle returned nothing" that gemmini produced before the check existed.
    # Evaluating it here covers bespoke_sim / oot_cert / program_oracle from one place, and only at the
    # FULL extent: the blocked path above already attributes each sub-tile it chose to fit.
    if observed is not None:
        try:
            # THE ACCUMULATOR DTYPE IS PASSED, and leaving it out was not cosmetic. It defaults to
            # 8 bits, so the accumulator term was evaluated at 4x the capacity this target
            # actually has (65536 elements against 16384), and the obligation recorded a fit for
            # layers the tiler standing beside it was already splitting. Measured: (96,64)@(64,512)
            # -- a shape whose own regression docstring records spike aborting on it -- reported
            # holds=True without it and holds=False with it. The binding has carried the real
            # accumulate type all along; the tiler passes it and these two sites did not.
            observed["capacity_fit_check"] = capacity_fit(
                target, M, K, N, binding.operand_dtype, _D, binding.accum_dtype
            )
        except Exception:  # noqa: BLE001 — unresolvable target: no obligation known
            pass
    out = _dispatch(mlir, A, W)
    if out is not None and _integer_limit is not None and not epilogue:
        import numpy as _np

        _array = _np.asarray(out, dtype=_np.float64)
        if (
            _array.shape != (M, N)
            or not _np.isfinite(_array).all()
            or not _np.array_equal(_array, _np.rint(_array))
            or _np.any(_array < -(1 << 31))
            or _np.any(_array > (1 << 31) - 1)
            or _np.any(_np.abs(_array) > K * _integer_limit["maximum_product"])
        ):
            if observed is not None:
                observed["decline"] = "integer mesh direct primitive exceeds its exact i32/K bound"
            return None
        if observed is not None:
            observed["integer_split_k"] = {
                "schema": "merlin.integer_split_k.v1",
                "padded_shape": [M, K, N],
                "mesh_tile_dim": _D,
                "tile": [M, K, N],
                "max_padded_k_per_dispatch": _integer_limit["max_padded_k"],
                "mac_result_bits": _integer_limit["mac_result_bits"],
                "signed_operand_bits": _integer_limit["signed_operand_bits"],
                "aggregation": "exact_i64_sum_checked_i32_before_epilogue",
                "primitive_calls": [
                    {
                        "offsets": [0, 0, 0],
                        "shape": [M, K, N],
                        "worst_case_padded_shape": [M, K, N],
                        "oracle_evidence": observed.get("oracle_evidence"),
                        "trace_check": observed.get("trace_check"),
                        "artifact_identity": observed.get("artifact_identity"),
                        "cert_run_id": observed.get("cert_run_id"),
                    }
                ],
            }
    if out is not None or _tiled:
        return _unpad(out)  # already a tile of a split layer: do not recurse further
    # Past the return above, the mesh DECLINED this extent. Name why, before either tiler is tried:
    # the obligation evaluated a moment ago already predicted it, or it could not be evaluated at all.
    if observed is not None:
        _cf = observed.get("capacity_fit_check") or {}
        if _cf.get("holds") is False:
            observed["contract_violation"] = {
                **_cf,
                "discharged_by": None,
                "detail": (
                    f"the target backend does not satisfy the capacity_fit obligation this "
                    f"interface requires: the contraction needs {_cf['required_elems']} "
                    f"resident elements against a declared capacity of {_cf['capacity_elems']}. "
                    f"The lowering must block for residency, not only tile the iteration space."
                ),
            }
        elif _cf.get("holds") is None:
            # SAY THAT THE OBLIGATION COULD NOT BE EVALUATED. A target that declares no on-chip operand
            # capacity gets `holds: None`, which is correct (never assume it holds) but silent -- and a
            # silent None is how a whole class of failures came to be reported as an unreachable oracle.
            # Recording it means a decline on such a target names the missing fact instead of hiding it.
            observed["capacity_fit_unevaluable"] = {
                **_cf,
                "detail": (
                    "this target declares no on-chip operand-store capacity, so the capacity_fit "
                    "obligation could not be evaluated and this decline is UNATTRIBUTED: it may "
                    "or may not be a residency failure. Declare the operand store in the "
                    "target's RTL facts to make the obligation decidable."
                ),
            }

    # The datapath has a magnitude floor: a layer whose operands are small is refused, and one doubling
    # of either operand flips it. Retry once with both operands rescaled to a power of two near 1.0 and
    # the result divided back. Exact -- a power-of-two factor moves only the exponent -- so this changes
    # which layers run and not what they compute.
    if not _rescaled:
        try:
            import math as _math

            _amax = max((abs(v) for row in A for v in row), default=0.0)
            _wmax = max((abs(v) for row in W for v in row), default=0.0)
            if _amax > 0.0 and _wmax > 0.0:
                _i = -int(_math.floor(_math.log2(_amax)))
                _j = -int(_math.floor(_math.log2(_wmax)))
                if _i or _j:
                    _sa, _sw = 2.0**_i, 2.0**_j
                    _As = [[v * _sa for v in row] for row in A]
                    _Ws = [[v * _sw for v in row] for row in W]
                    _scaled = run_matmul_on_mesh(
                        target,
                        _As,
                        _Ws,
                        operand_dtype=operand_dtype,
                        accum_dtype=accum_dtype,
                        numeric_policy=numeric_policy,
                        simulator=simulator,
                        package=package,
                        timeout=timeout,
                        epilogue=epilogue,
                        acc_scale=acc_scale,
                        observed=observed,
                        _rescaled=True,
                    )
                    if _scaled is not None:
                        _inv = 1.0 / (_sa * _sw)
                        return _unpad([[v * _inv for v in row] for row in _scaled])
        except Exception:  # noqa: BLE001 — rescale is an optimisation, not a gate
            pass

    # EMPIRICAL TILER -- the fallback, reached only when the DERIVED one above could not act. The two
    # are complementary, not rivals: the blocking path above computes a fitting extent from the target's
    # own RTL-derived operand-store capacity and runs first, but `_operand_store_capacity_elems` returns
    # None for a target that declares no such fact. There is then nothing to derive a tile size FROM, and
    # inventing one would be a target literal in disguise. So find the width this backend ACCEPTS by
    # halving, then run the layer as N-tiles of that width: splitting along N is exact, because
    # C[:, a:b] is A @ W[:, a:b]. Costs a few refused probes ONCE per target.
    D = int(getattr(binding, "tile_dim", 0) or 0)
    if D <= 0:
        return None

    def _probed(extent, n_sub, axis):
        """Record that a PROBE, not a capacity fact, chose this extent.

        Both tilers write `observed["capacity_fit"]`, and the layer record downstream reads
        `tile_source` off it. "Derived from an RTL scratchpad capacity" and "found by halving until the
        backend stopped refusing" are very different provenance claims about the same number.
        """
        if observed is None:
            return
        observed["capacity_fit"] = {
            "holds": None,
            "required_elems": None,
            "capacity_elems": None,
            "tile_source": "probed",
            "tile": list(extent),
            "n_subtiles": int(n_sub),
            "split_axis": axis,
            "discharged_by": "merlin runtime (host-side residency tiling)",
            # An N or M split leaves every output element a single full-K reduction on the device, so
            # unlike the K-blocking the derived tiler does, this stays exact on a float accumulator too.
            "split_exact": True,
            "accum_dtype": binding.accum_dtype,
            "note": (
                f"this target declares no on-chip operand-store capacity, so no tile size could be "
                f"DERIVED; the extent above is the largest this backend accepted when probed by "
                f"halving. The numerics are the device's own (an {axis}-split is exact), but the "
                f"EXTENT is an empirical finding about this backend, not a derived property of the "
                f"hardware -- and the layer ran because the runtime split it."
            ),
        }

    _rows_kw = dict(
        numeric_policy=numeric_policy,
        operand_dtype=operand_dtype,
        accum_dtype=accum_dtype,
        simulator=simulator,
        package=package,
        timeout=timeout,
        epilogue=epilogue,
        acc_scale=acc_scale,
        observed=observed,
        M=M,
        record=_probed,
    )
    if N <= D:
        return _unpad(_mesh_rows(target, A, W, **_rows_kw))  # nothing to split along N
    # Keyed by K as well: the accepted width is bounded by the WORKING SET (K*w + M*K), so a width
    # discovered at K=128 is too wide at K=344 and its tiles are refused in turn. Measured: caching
    # without K left 4 of 15 small_llama layers on the host for exactly that reason.
    key = (target, operand_dtype or "", accum_dtype or "", K)
    width = _MESH_NTILE_WIDTH.get(key)
    if width is None:
        w = N
        while w > D:
            w = max(D, (w // 2 // D) * D or D)
            probe = run_matmul_on_mesh(
                target,
                A,
                [row[:w] for row in W],
                operand_dtype=operand_dtype,
                accum_dtype=accum_dtype,
                numeric_policy=numeric_policy,
                simulator=simulator,
                package=package,
                timeout=timeout,
                epilogue=epilogue,
                acc_scale=acc_scale,
                observed=observed,
                _tiled=True,
            )
            if probe is not None:
                width = w
                break
        if width is None:
            # Even one tile-dim-wide column is refused, so the M*K term is what does not fit.
            return _unpad(_mesh_rows(target, A, W, **_rows_kw))
        _MESH_NTILE_WIDTH[key] = width
    cols: list[list] = []
    for a in range(0, N, width):
        b = min(a + width, N)
        piece = run_matmul_on_mesh(
            target,
            A,
            [row[a:b] for row in W],
            operand_dtype=operand_dtype,
            accum_dtype=accum_dtype,
            numeric_policy=numeric_policy,
            simulator=simulator,
            package=package,
            timeout=timeout,
            epilogue=epilogue,
            acc_scale=acc_scale,
            observed=observed,
            _tiled=True,
        )
        if piece is None:
            return None  # a tile the discovered width should have covered
        cols.append(piece)
    _probed((M, K, width), (N + width - 1) // width, "N")
    return _unpad([sum((c[r] for c in cols), []) for r in range(M)])


def _mesh_rows(
    target,
    A,
    W,
    *,
    operand_dtype,
    accum_dtype,
    numeric_policy,
    simulator,
    package,
    timeout,
    epilogue,
    acc_scale,
    M,
    observed=None,
    record=None,
):
    """Run a refused layer as row blocks. The working set is ``K*N + M*K``; N-tiling shrinks the first
    term and M-tiling the second, so a layer whose N is already at or below the tile dim (a projection
    down to a couple of columns, say) can only be helped by splitting M. Exact: C[a:b, :] is A[a:b, :] @ W.
    """
    if M <= 1:
        return None
    height = M
    while height > 1:
        height = max(1, height // 2)
        probe = run_matmul_on_mesh(
            target,
            A[:height],
            W,
            operand_dtype=operand_dtype,
            accum_dtype=accum_dtype,
            numeric_policy=numeric_policy,
            simulator=simulator,
            package=package,
            timeout=timeout,
            epilogue=epilogue,
            acc_scale=acc_scale,
            observed=observed,
            _tiled=True,
        )
        if probe is not None:
            break
    else:
        return None
    rows: list = []
    for a in range(0, M, height):
        piece = run_matmul_on_mesh(
            target,
            A[a : a + height],
            W,
            operand_dtype=operand_dtype,
            accum_dtype=accum_dtype,
            numeric_policy=numeric_policy,
            simulator=simulator,
            package=package,
            timeout=timeout,
            epilogue=epilogue,
            acc_scale=acc_scale,
            observed=observed,
            _tiled=True,
        )
        if piece is None:
            return None
        rows.extend(piece)
    if record is not None:
        record((height, len(A[0]) if A else 0, len(W[0]) if W else 0), (M + height - 1) // height, "M")
    return rows


def _certify_tile_via_executor(target, mlir, *, m, k, n, binding, timeout, numeric_policy=None) -> dict:
    """Retain the original selected callbacks at the runtime boundary."""
    from merlin.verification.mesh import _certify_tile_via_executor as implementation

    return implementation(
        target,
        mlir,
        m=m,
        k=k,
        n=n,
        binding=binding,
        timeout=timeout,
        numeric_policy=numeric_policy,
        _executor=run_matmul_on_mesh,
    )


def _mesh_verify(plan: dict, *, target: str, package: str | None, timeout: int, numeric_policy=None) -> dict:
    """Retain the original selected callbacks at the runtime boundary."""
    from merlin.verification.mesh import _mesh_verify as implementation

    return implementation(
        plan,
        target=target,
        package=package,
        timeout=timeout,
        numeric_policy=numeric_policy,
        _package_selector=_default_oot_package,
        _tile_binding=_mesh_tile_binding,
        _tile_builder=tile_builder_op,
        _tile_certifier=_certify_tile_via_executor,
    )


def _matmul_via_oot_cert(
    target, mlir, A, W, *, simulator, package, timeout, layer_id: str = "mesh_layer", observed: dict | None = None
) -> list | None:
    """Retain the original selected callbacks at the runtime boundary."""
    from merlin.runtime.mesh_backend import _matmul_via_oot_cert as implementation

    return implementation(
        target,
        mlir,
        A,
        W,
        simulator=simulator,
        package=package,
        timeout=timeout,
        layer_id=layer_id,
        observed=observed,
        _package_selector=_default_oot_package,
    )


def _matmul_via_program_oracle(
    target, mlir, A, W, *, model_ext, package, timeout, operand_dtype=None, observed: dict | None = None
) -> list | None:
    """Retain the original selected callbacks at the runtime boundary."""
    from merlin.runtime.mesh_backend import _matmul_via_program_oracle as implementation

    return implementation(
        target,
        mlir,
        A,
        W,
        model_ext=model_ext,
        package=package,
        timeout=timeout,
        operand_dtype=operand_dtype,
        observed=observed,
        _package_selector=_default_oot_package,
    )


def _matmul_via_bespoke_sim(
    target, mlir, A, W, *, package, timeout, layer_id: str = "mesh_layer", observed: dict | None = None
) -> list | None:
    """Retain the original selected callbacks at the runtime boundary."""
    from merlin.runtime.mesh_backend import _matmul_via_bespoke_sim as implementation

    return implementation(
        target,
        mlir,
        A,
        W,
        package=package,
        timeout=timeout,
        layer_id=layer_id,
        observed=observed,
        _package_selector=_default_oot_package,
    )


for _legacy_function in (
    run_matmul_on_mesh,
    _mesh_rows,
    _certify_tile_via_executor,
    _mesh_verify,
    _matmul_via_oot_cert,
    _matmul_via_program_oracle,
    _matmul_via_bespoke_sim,
):
    _legacy_function.__module__ = "merlin.runtime.mesh"
del _legacy_function
