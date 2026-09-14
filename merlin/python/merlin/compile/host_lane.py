"""The RVV host lane: which package a compile uses for each datatype, and its provenance pin.

``default_package`` resolves the certified champion for a ``--dtype`` (``_DTYPE_STRATEGY`` maps the token to
the ``dtype_strategy`` packages declare) and refuses one that has drifted from the lane pinned in the
provenance registry; ``host_lane_identity`` is the record every compile carries of the lane it used.
"""
from __future__ import annotations

from pathlib import Path

from .bundles import ir_scalar_dtype


#: knobs.yaml ``dtype_strategy`` for each ``--dtype``, used to pick a champion package of the
#: RIGHT datatype. An fp32 schedule applied to an int8 workload builds a silently wrong
#: datapath rather than failing, so this must never fall back across datatypes.
#:
#: These strings must match what packages actually declare (``mining.tuning_agent``'s strategy
#: set: fp32, int8_w8a8, bf16_f32acc, fp16_f32acc). "fp16"/"fp8" matched nothing, so
#: ``select_champion`` raised and the fallback below handed back ``hand_v0`` — an **fp32**
#: package — which is exactly the cross-datatype substitution the paragraph above forbids.
#: NOTE ``fp8`` is deliberately ABSENT. A dtype only belongs here when the SCALAR/RVV lane has a
#: datapath for it, and it does not for fp8: the widening rewrite that gives narrow floats an f32
#: accumulator (``passes_xdsl.lower_bf16_matmul_f32acc``) is typed on the xDSL builtin float types,
#: which have no fp8 member at all, and every fp8 recapture's ``model.mlir`` is 100% f32 (the bundles
#: are weight-only fake-quant, so the fp8 never reaches the IR). fp8 IS a real MESH operand format --
#: it travels as ``operation.attributes.dtype`` through ``routing_dtype`` and executes on the target's
#: matrix unit -- but that is a different lane from the one this map configures. Mapping it to a
#: strategy string no package can legally declare produced "no package declares dtype_strategy='fp8'",
#: which reads as a missing artifact and sent readers off to mint a package that cannot exist.
_DTYPE_STRATEGY = {"int8": "int8_w8a8", "fp32": "fp32", "fp16": "fp16_f32acc",
                   "bf16": "bf16_f32acc"}



def host_lane_pin_name(strategy: str) -> str:
    """The provenance-registry artifact that pins the host lane for ``strategy``."""
    return f"rvv_host_lane_{strategy}"


def host_lane_identity(package_dir: "str | Path") -> dict:
    """``{package, package_sha256, n_files, dtype_strategy, pinned_as}`` for a host-lane package.

    Recorded on EVERY compile, under the same key and subkeys the grading path writes, so a graded run
    and an ordinary one produce comparable records and a result that used an unpinned lane is
    detectable afterwards instead of indistinguishable.
    """
    from pathlib import Path as _P
    from merlin.benchharness import hash_tree
    from merlin.common.provenance import load_artifacts

    d = _P(package_dir)
    out: dict = {"package": str(d), "package_sha256": None, "n_files": None,
                 "dtype_strategy": None, "pinned_as": None}
    try:
        hashed = hash_tree(d)
        out["package_sha256"] = hashed.get("sha256")
        out["n_files"] = hashed.get("n_files")
    except Exception:                              # noqa: BLE001 -- an unhashable package is not a digest
        pass
    try:
        from ..mining.registry import load_rvv_package
        out["dtype_strategy"] = load_rvv_package(d).dtype_strategy
    except Exception:                              # noqa: BLE001
        pass
    try:
        for name, art in load_artifacts().items():
            resolved = art.resolve()
            if resolved is not None and _P(resolved).resolve() == d.resolve():
                out["pinned_as"] = name
                break
    except Exception:                              # noqa: BLE001 -- an unreadable registry is not a pin
        pass
    return out


def _verified_against_the_pinned_lane(package_dir: str, strategy: str) -> str:
    """Refuse a certified champion that has drifted from the lane pinned for its precision.

    ``select_champion`` picks whatever currently ranks highest. That is the right answer for tuning and
    the wrong one for reproducing a graded result: no fp32 package declares ``publication.champion``,
    so the fp32 choice falls out of ``_rank_key``'s newest-wins tie-break, and minting one more fp32
    package silently redirects every unpinned compile. REFUSE rather than warn -- the failure mode is
    that nothing is printed at all.

    An undeclared pin is a warning, not a refusal: this registry is opt-in per lane, and requiring a pin
    that nobody has written yet would break every dtype that has one package and no declaration.
    """
    from pathlib import Path as _P
    from merlin.common.provenance import PinsError, load_artifacts, verify_artifact

    name = host_lane_pin_name(strategy)
    try:
        declared = load_artifacts().get(name)
    except PinsError:
        declared = None
    if declared is None:
        print(f"[merlin-compile] host lane for dtype_strategy={strategy!r} is UNPINNED: no artifact "
              f"{name!r} in the provenance registry, so this compile's host compiler is whatever "
              f"currently ranks highest and is not reproducible from the registry.", flush=True)
        return package_dir
    pinned = declared.resolve()
    if pinned is not None and _P(pinned).resolve() != _P(package_dir).resolve():
        raise SystemExit(
            f"[merlin-compile] the certified champion for dtype_strategy={strategy!r} has DRIFTED from "
            f"the pinned host lane.\n  champion: {package_dir}\n  pinned as {name!r}: {pinned}\n"
            f"One of the two is wrong: either promote the new package into the registry (updating its "
            f"digest), or stop promoting it. Refusing to compile against a host lane the registry does "
            f"not name, because a graded capsule and this compile would then use different compilers.")
    check = verify_artifact(name)
    if check.matches is False:
        raise SystemExit(
            f"[merlin-compile] the pinned host lane {name!r} has been EDITED: its tree digest is "
            f"{(check.digest or '')[:16]} but the registry declares {(declared.digest or '')[:16]}. "
            f"Re-record the digest deliberately, or restore the package; a silently-changed host "
            f"compiler makes every result built with it unattributable.")
    return package_dir

def default_package(dtype: str, *, bundle: "Path | None" = None) -> str:
    """The package `merlin-compile` uses when `--package` is not given.

    Resolves the CERTIFIED CHAMPION for this datatype via ``targetgen.publish.select_champion``
    rather than hard-coding a name. The previous default was ``hand_v0``/``hand_v0_int8`` — the
    FROZEN, hand-authored, UNOPTIMIZED control that exists to be the before/after baseline. Every
    default invocation therefore shipped the slowest package in the repo while the tuned ones sat
    unused. Falls back to the hand baseline only when no package of this dtype is certified, and
    says so.
    """
    from ..mining.tuning_agent import _DTYPE_STRATEGIES
    from ..targetgen.publish import PublishError, select_champion
    strategy = _DTYPE_STRATEGY.get(dtype)
    if strategy is None and bundle is not None:
        # The requested dtype names how the model was QUANTIZED; it does not name what the compiled IR
        # carries, and only the latter decides which scalar datapath is correct. A weight-only fp8 capture
        # emits f32 tensors end to end, so the f32 package IS its datapath -- not a cross-datatype
        # substitution but the derived one. Read it off the bundle rather than refusing.
        derived = ir_scalar_dtype(bundle)
        if derived is not None and derived in _DTYPE_STRATEGY:
            print(f"[merlin-compile] --dtype {dtype} has no scalar/RVV datapath; the bundle's IR carries "
                  f"{derived}, so the scalar lane uses the {derived} package. ({dtype} remains the MESH "
                  f"operand format, routed and executed on the matrix unit.)", flush=True)
            dtype, strategy = derived, _DTYPE_STRATEGY[derived]
    if strategy is None:
        raise SystemExit(
            f"[merlin-compile] --dtype {dtype} has no scalar/RVV datapath (known: "
            f"{', '.join(sorted(_DTYPE_STRATEGY))}). If {dtype} is a MESH operand format, it belongs in "
            f"the capsule's operation.attributes.dtype (threaded as routing_dtype and executed on the "
            f"matrix unit), and the scalar lane should declare the dtype its IR actually carries. Only "
            f"add it here alongside a lowering that gives {dtype} a scalar datapath.")
    if strategy not in _DTYPE_STRATEGIES:
        # A map entry naming a strategy the knob validator rejects can never be satisfied by ANY package.
        # Diagnose it as the configuration error it is rather than as a missing artifact.
        raise SystemExit(
            f"[merlin-compile] _DTYPE_STRATEGY maps --dtype {dtype} to dtype_strategy {strategy!r}, which "
            f"is not a strategy packages may declare ({', '.join(sorted(_DTYPE_STRATEGIES))}); no package "
            f"can ever satisfy it. Fix the map, or add {strategy!r} to mining.tuning_agent.")
    try:
        sel = select_champion("rvv", dtype_strategy=strategy)
        return _verified_against_the_pinned_lane(str(sel.package_dir), strategy)
    except PublishError:
        # Fall back only WITHIN the datatype. The frozen hand baselines exist for fp32 and int8
        # only, so any other dtype has no same-datatype control to fall back to — and handing
        # back an fp32 schedule there would build a silently wrong datapath (see _DTYPE_STRATEGY).
        # Fail with the fix instead: mint and certify a package for this strategy.
        fallback = {"int8": "hand_v0_int8", "fp32": "hand_v0"}.get(dtype)
        if fallback is None:
            raise SystemExit(
                f"[merlin-compile] no package declares dtype_strategy={strategy!r} (for --dtype "
                f"{dtype}), and there is no {dtype} baseline to fall back to. Refusing to "
                f"substitute a package of a different datatype. Either pass --package explicitly "
                f"or mint one: see docs/guides/targetgen.md.") from None
        print(f"[merlin-compile] no certified {strategy} package; falling back to the frozen "
              f"baseline {fallback} (this is the UNOPTIMIZED control)", flush=True)
        from ..common.artifacts import artifacts_dir
        return str(artifacts_dir() / "targets" / "rvv" / fallback)
