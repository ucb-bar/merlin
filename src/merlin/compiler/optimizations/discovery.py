"""Resolve existing feature names without importing every legacy implementation."""

from __future__ import annotations

from merlin.common.paths import module_source_path
from merlin.compiler.optimizations.blocks import (
    ensure_v3_microkernel,
    ensure_v3_perop_microkernel,
    ensure_v3_scalable_microkernel,
)
from merlin.compiler.optimizations.pointwise import _vec_noncontraction_point, ensure_vec_noncontraction
from merlin.compiler.selection import shared as _shared_features

_LEVER_MODULES: "dict[str, str] | None" = None


def _single_name_lever_modules() -> "dict[str, str]":
    """Lever name -> module basename, for every sibling module declaring lever names as constants.

    Discovered by PARSING the sibling sources rather than importing them: at this point in the
    package every lever module imports THIS one to call :func:`register`, so importing them all
    eagerly is circular. Parsing is also structural, not a line-match -- the module-level
    ``FEATURE = "..."`` assignment is read out of the AST, and a module only qualifies if it also
    defines ``ensure_registered`` at module level, which is the thing that will be called.

    A module that declares no ``FEATURE`` (a family module carrying a ``FEATURE_PREFIX``, or a plain
    helper) simply does not appear, so families keep resolving through their own prefix/arity rules.

    ``<X>_FEATURE`` counts too, not only the bare ``FEATURE``. A module may own more than one point --
    ``requant_fuse`` registers the traversal removal and the variant that also reshapes the epilogue
    loop, so the board can price the reshape separately -- and ``perop_blocks`` has always owned
    ``CONV_ARM_FEATURE`` with no way to resolve it here. Reading every module-level string constant
    whose name ends in ``FEATURE`` keeps the mapping DERIVED, which is the property this function
    exists for; a name whose ``ensure_registered`` does not actually register it still falls through
    to the family rules below, because the caller re-checks the registry.
    """
    global _LEVER_MODULES
    if _LEVER_MODULES is None:
        import ast as _ast

        found: dict[str, str] = {}
        for src in sorted(module_source_path("merlin.llvmlower").parent.glob("*.py")):
            if src.name == "impr_features.py":
                continue
            try:
                tree = _ast.parse(src.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue  # unreadable/unparseable sibling is not a lever; never guess one
            features: list[str] = []
            has_ensure = False
            for node in tree.body:  # MODULE level only -- a nested FEATURE is not the lever's name
                if (
                    isinstance(node, _ast.Assign)
                    and isinstance(node.value, _ast.Constant)
                    and isinstance(node.value.value, str)
                    and any(
                        isinstance(t, _ast.Name) and (t.id == "FEATURE" or t.id.endswith("_FEATURE"))
                        for t in node.targets
                    )
                ):
                    features.append(node.value.value)
                elif isinstance(node, _ast.FunctionDef) and node.name == "ensure_registered":
                    has_ensure = True
            if has_ensure:
                for feature in features:
                    found.setdefault(feature, src.stem)
        _LEVER_MODULES = found
    return _LEVER_MODULES


def _try_lazy_register(name: str) -> bool:
    """Auto-register a v3 micro-kernel tuning point from its NAME (accum_resident_v3_<MR>_<NR>_<KC>).

    The register block is a continuous, beam-tunable knob space, but the lowering runs in a SUBPROCESS
    that re-imports this module — so a point registered on demand in the parent is invisible there and
    the feature fails to resolve (observed: every dynamically-registered MR failed at K2 while the
    pre-registered grid worked). Deriving the point from the name makes resolution reproducible in ANY
    process, which is what actually makes the space continuous.

    The same holds for the two other on-demand v3 families, which were previously NOT derivable from
    their name (their prefix is ``accum_resident_v3vl_`` / ``accum_resident_v3p_``, so the
    ``accum_resident_v3_`` test rejected them): a package that names one of those points directly in
    ``compiler_features`` -- rather than reaching it through a ``microkernel`` knob block, which
    registers it as a side effect of resolving -- failed with an "unknown impr feature" KeyError.
    Each family is registered from its own arity, so a name is either derivable or an honest error."""
    from merlin.compiler.optimizations.registry import _REGISTRY, register

    try:
        shared = _shared_features.get_shared_feature(name)
    except KeyError:
        pass
    else:
        register(shared)
        return True
    # A SINGLE-NAME lever lives in its own module and resolves by importing that module, rather than
    # through the arity table below. It has to resolve HERE and not only where it is ranked:
    # `wholemodel_proposer` registers it for the PROPOSAL path, but a package that names it in
    # `compiler_features` reaches `normalize` through `k1.build_k1_binary`, which imports no proposer
    # -- and through the lowering SUBPROCESS, which re-imports this module and sees no registration
    # the parent made at run time. Both raise "unknown impr feature" without this.
    #
    # This was a HAND-MAINTAINED list of per-module hooks, and it drifted exactly as such lists do:
    # it named two modules while THIRTEEN declared a lever. The other eleven resolved only by
    # ACCIDENT -- an unrelated module happening to import theirs (`pipeline` pulls in `prov_cse` and
    # `im2col_pack`), or a call-time `ensure_registered()` buried in `k1.build_k1_binary` (that is
    # what `prepack_weight_layout` relies on). Accidental resolution is not resolution: a fresh
    # process that imports only THIS module could not resolve seven of the thirteen, and the lowering
    # subprocess is exactly such a process. Removing an unrelated import, or reaching a lever by a
    # path that skips `build_k1_binary`, silently turns a named lever into "unknown impr feature" --
    # or, where a caller swallows that, into a lever reported as applied and never applied.
    # The mapping is DERIVED from the modules themselves so that resolution does not depend on who
    # else happened to be imported first.
    _module = _single_name_lever_modules().get(name)
    if _module is not None:
        import importlib

        importlib.import_module(f"merlin.llvmlower.{_module}").ensure_registered()
        return name in _REGISTRY
    # The parallel-GRAIN family carries its threshold in the name and lives in its own module, so
    # it is resolved first and by import rather than through the arity table below.
    from merlin.llvmlower.parallel_grain import FEATURE_PREFIX as _PG_PREFIX

    if name.startswith(_PG_PREFIX):
        from merlin.llvmlower.parallel_grain import ensure_registered as _pg_ensure

        try:
            _pg_ensure(int(name[len(_PG_PREFIX) :]))
        except ValueError:
            return False
        return name in _REGISTRY
    # Per-region OpenMP team policy is another name-derived continuous family.  Resolve it here so
    # a package and the lowering subprocess do not depend on the proposer having imported it first.
    from merlin.llvmlower.parallel_team import FEATURE_PREFIX as _PT_PREFIX

    if name.startswith(_PT_PREFIX):
        from merlin.llvmlower.parallel_team import ensure_registered as _pt_ensure

        try:
            _pt_ensure(int(name[len(_PT_PREFIX) :]))
        except ValueError:
            return False
        return name in _REGISTRY
    from merlin.llvmlower.residual_parallel import FEATURE_PREFIX as _RP_PREFIX

    if name.startswith(_RP_PREFIX):
        from merlin.llvmlower.residual_parallel import ensure_registered as _rp_ensure

        try:
            _rp_ensure(int(name[len(_RP_PREFIX) :]))
        except ValueError:
            return False
        return name in _REGISTRY
    point = _vec_noncontraction_point(name)
    if point is not None:
        # The non-contraction family's points are DERIVABLE FROM THE NAME (lanes and max rank), so
        # they resolve in any process -- which is the whole reason this hook exists: a point the
        # parent registered at run time is invisible to the lowering subprocess that re-imports this
        # module, and the build fails with "unknown impr feature" rather than with the lever off.
        ensure_vec_noncontraction(*point)
        return name in _REGISTRY
    parts = name.split("_")
    tails: dict[str, tuple[int, "Callable[..., str]"]] = {
        "accum_resident_v3_": (3, ensure_v3_microkernel),  # MR, NR, KC
        "accum_resident_v3vl_": (3, ensure_v3_scalable_microkernel),  # MR, NR, KC
        "accum_resident_v3p_": (5, ensure_v3_perop_microkernel),  # MR_mm, NR_mm, MR_bmm, NR_bmm, KC
    }
    for prefix, (arity, make) in tails.items():
        if not name.startswith(prefix):
            continue
        args = parts[len(prefix.rstrip("_").split("_")) :]
        if len(args) != arity:
            return False
        try:
            make(*(int(a) for a in args))
        except ValueError:
            return False
        return name in _REGISTRY
    return False
