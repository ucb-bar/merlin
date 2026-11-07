"""Compatibility facade for named compiler features.

Scalar and shared feature definitions live in :mod:`shared_feature_registry`.
Executable RVV features and schedules live in the selected rvv-mlir provider.
This module retains the old import surface for existing forks; importing it does
not load RVV, so a scalar compiler has no vector-provider dependency.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .feature_registry import CompositionError, FeatureRegistry, ImprFeature
from .shared_feature_registry import (
    NAMED_INT8_CONTRACTION_NAME,
    QUANTIZE_BEFORE_GATHER_NAME,
    eager_features,
    get_shared_feature,
)
from .shared_features import (
    _PROMOTE_STACK_DEFAULT_BYTES,
    FUSE_AFTER_GENERALIZE_NAME,
    FUSE_ELEMENTWISE_ANCHOR,
    FUSE_ELEMENTWISE_NAME,
    FUSE_ELEMENTWISE_STAGE,
    PROMOTE_STACK_BYTES_ENV,
    PROMOTE_STACK_NAME,
    _fuse_after_generalize,
    _fuse_elementwise_pipeline,
    _promote_buffers_to_stack,
    _promote_stack_bytes,
    _reuse_tensor_destination,
)

__all__ = [
    "CompositionError",
    "FeatureRegistry",
    "ImprFeature",
    "NAMED_INT8_CONTRACTION_NAME",
    "QUANTIZE_BEFORE_GATHER_NAME",
    "_PROMOTE_STACK_DEFAULT_BYTES",
    "FUSE_AFTER_GENERALIZE_NAME",
    "FUSE_ELEMENTWISE_ANCHOR",
    "FUSE_ELEMENTWISE_NAME",
    "FUSE_ELEMENTWISE_STAGE",
    "PROMOTE_STACK_BYTES_ENV",
    "PROMOTE_STACK_NAME",
    "_fuse_after_generalize",
    "_fuse_elementwise_pipeline",
    "_promote_buffers_to_stack",
    "_promote_stack_bytes",
    "_reuse_tensor_destination",
]

# Retained names are import compatibility, not standalone scalar registrations.
OPU_MATMUL_NAME = "opu_matmul"
BUFFERIZED_RESULT_IDENTITY_NAME = "canonicalize_bufferized_result_identity"
SCALAR_SQUARED_SUM_NAME = "scalar_squared_sum_accumulator"
VECTORIZE_AMAX_REDUCTION_NAME = "vectorize_amax_reduction"

_REGISTRY: dict[str, ImprFeature] = {feature.name: feature for feature in eager_features()}


def _provider():
    from .backend_spi import load_vector_codegen

    return load_vector_codegen("rvv")


def _provider_module(name: str):
    backend = _provider()
    root = backend.__class__.__module__.rsplit(".", 1)[0]
    return importlib.import_module(f"{root}.{name}")


def _try_lazy_register(name: str) -> bool:
    if name in _REGISTRY:
        return True
    try:
        feature = get_shared_feature(name)
    except KeyError:
        pass
    else:
        _REGISTRY[name] = feature
        return True
    try:
        feature = _provider().registry.get(name)
    except (KeyError, OSError, ValueError):
        return False
    _REGISTRY[name] = feature
    return True


def _features() -> FeatureRegistry:
    registry = FeatureRegistry(lazy_register=_try_lazy_register)
    registry.entries = _REGISTRY
    return registry


def register(feature: ImprFeature) -> ImprFeature:
    return _features().register(feature)


def get(name: str) -> ImprFeature:
    return _features().get(name)


def _known_provider_names() -> None:
    """Materialize the provider's declared compatibility catalog on demand."""
    try:
        backend = _provider()
    except (OSError, ValueError):
        return
    module = importlib.import_module(backend.__class__.__module__)
    catalog = Path(module.__file__).parent / "tests" / "feature_names_v1.txt"
    names = catalog.read_text().splitlines() if catalog.is_file() else backend.feature_names()
    for name in names:
        if name != OPU_MATMUL_NAME:
            _try_lazy_register(name)


def known() -> list[str]:
    _known_provider_names()
    return sorted(_REGISTRY)


def normalize(features) -> frozenset[str]:
    return _features().normalize(features)


def apply_pipeline(passes: list[str], features: frozenset[str]) -> list[str]:
    return _features().apply_pipeline(passes, features)


def apply_schedule(schedule_text: str, features: frozenset[str]) -> str:
    return _features().apply_schedule(schedule_text, features)


def apply_cflags(cflags, features: frozenset[str]) -> list[str]:
    return _features().apply_cflags(cflags, features)


def ensure_promote_stack(nbytes: int) -> str:
    n = int(nbytes)
    if n <= 0 or n != nbytes:
        raise ValueError(f"stack-promotion cap must be a positive integer, got {nbytes!r}")
    name = f"{PROMOTE_STACK_NAME}_{n}"
    if name not in _REGISTRY:
        register(
            ImprFeature(
                name=name,
                action_class="PASS",
                description=f"Promote bufferization allocations of at most {n} bytes to the stack.",
                edit_pipeline=lambda passes, _n=n: _promote_buffers_to_stack(passes, cap=_n),
            )
        )
    return name


def _single_name_lever_modules() -> dict[str, str]:
    """Preserve the old inspection API for shared single-name feature modules."""
    import ast

    found: dict[str, str] = {}
    for source in sorted(Path(__file__).parent.glob("*.py")):
        if source.name == Path(__file__).name:
            continue
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        names = [
            node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and any(
                isinstance(target, ast.Name) and (target.id == "FEATURE" or target.id.endswith("_FEATURE"))
                for target in node.targets
            )
        ]
        if any(isinstance(node, ast.FunctionDef) and node.name == "ensure_registered" for node in tree.body):
            for name in names:
                found.setdefault(name, source.stem)
    return found


def _bound_call(name: str, *args, **kwargs):
    backend = _provider()
    renamed = {
        "MR": "mr",
        "NR": "nr",
        "KC": "kc",
        "MR_mm": "mr_mm",
        "NR_mm": "nr_mm",
        "MR_bmm": "mr_bmm",
        "NR_bmm": "nr_bmm",
    }
    kwargs = {renamed.get(key, key): value for key, value in kwargs.items()}
    result = getattr(backend, name)(*args, **kwargs)
    if isinstance(result, str):
        _try_lazy_register(result)
    return result


def ensure_v3_microkernel(MR: int, NR: int, KC: int) -> str:
    return _bound_call("ensure_v3_microkernel", MR, NR, KC)


def ensure_v3_kblocked_microkernel(MR: int, NR: int, KC: int) -> str:
    return _bound_call("ensure_v3_kblocked_microkernel", MR, NR, KC)


def ensure_v3_unrolled_microkernel(MR: int, NR: int, KC: int) -> str:
    return _bound_call("ensure_v3_unrolled_microkernel", MR, NR, KC)


def ensure_v3_scalable_microkernel(MR: int, NR: int, KC: int) -> str:
    return _bound_call("ensure_v3_scalable_microkernel", MR, NR, KC)


def ensure_v3_perop_microkernel(MR_mm, NR_mm, MR_bmm, NR_bmm, KC: int) -> str:
    return _bound_call("ensure_v3_perop_microkernel", MR_mm, NR_mm, MR_bmm, NR_bmm, KC)


def ensure_vec_noncontraction(lanes: int, max_rank: int = 4) -> str:
    module = _provider_module("vector_features")
    result = module.ensure_vec_noncontraction(_provider().registry, lanes, max_rank)
    _try_lazy_register(result)
    return result


def ensure_lmul_group(*, operand_bits: int, acc_bits: int, vlen=None, max_group_elems=None) -> str:
    module = _provider_module("grid_features")
    result = module.ensure_lmul_group(
        _provider().registry,
        operand_bits=operand_bits,
        acc_bits=acc_bits,
        vlen=vlen,
        max_group_elems=max_group_elems,
    )
    _try_lazy_register(result)
    return result


def ensure_lmul_group_for_elem_types(a: str, b: str, c: str, *, vlen=None, max_group_elems=None) -> str:
    module = _provider_module("grid_features")
    result = module.ensure_lmul_group_for_elem_types(
        _provider().registry, a, b, c, vlen=vlen, max_group_elems=max_group_elems
    )
    _try_lazy_register(result)
    return result


def ensure_mrpad_for_elem_types(a: str, b: str, c: str, MR=4, NR=16, NR_bmm=8) -> str:
    module = _provider_module("accumulator_grid")
    result = module.ensure_mrpad_for_elem_types(_provider().registry, a, b, c, MR, NR, NR_bmm)
    _try_lazy_register(result)
    return result


def ensure_perop_block(table, kc: int, pairs=(), vec_epilogue: bool = False) -> str:
    return _bound_call("ensure_perop_block", table, kc, pairs, vec_epilogue)


def perop_mr_sentinel(mr_cap: int) -> str:
    return _bound_call("perop_mr_sentinel", mr_cap)


def perop_nr_sentinel(nr_cap: int) -> str:
    return _bound_call("perop_nr_sentinel", nr_cap)


_PROVIDER_MODULES = (
    "active_features",
    "accumulator_grid",
    "dynamic_accumulator",
    "grid_features",
    "other_vector_features",
    "perop_features",
    "vector_features",
    "reduce_vec",
    "im2col_pack",
    "weight_panel",
)

# A handful of old constants are metadata for compatibility tests and fork manifests.
# Executable definitions are resolved from the provider by __getattr__ below.
_COMPAT_CONSTANTS: dict[str, Any] = {
    "ACCUM_RESIDENT_V3_NAMES": [
        "accumulator_resident_microkernel_v3",
        "accum_resident_v3_4_16_32",
        "accum_resident_v3_8_16_16",
        "accum_resident_v3_4_32_16",
        "accumulator_resident_wholemodel_vf",
        "accumulator_resident_wholemodel_vf_bmmpad",
        "accumulator_resident_wholemodel_vf_mr4",
        "accumulator_resident_wholemodel_vf_mrpad",
    ],
    "LMUL_GROUP_NAMES": tuple(f"lmul_group_m{n}" for n in (1, 2, 4, 8)),
    "MRPAD_NAME": "accumulator_resident_wholemodel_vf_mrpad",
    "MRPAD_INT8_NAME": "accumulator_resident_wholemodel_vf_mrpad_i8_i8_i32_mr4_nr16_nb8",
    "MRPAD_INT8_TILES": tuple(
        f"accumulator_resident_wholemodel_vf_mrpad_i8_i8_i32_mr{mr}_nr{nr}_nb{nb}"
        for mr, nr, nb in (
            (1, 16, 8),
            (1, 32, 8),
            (1, 64, 16),
            (2, 16, 8),
            (4, 16, 8),
            (8, 16, 8),
            (4, 32, 8),
            (4, 64, 16),
            (8, 32, 16),
            (8, 64, 16),
        )
    ),
    "PEROP_MR_LADDER": tuple(f"perop_register_block_mr{n}" for n in (1, 2, 8, 16)),
    "PEROP_NR_LADDER": tuple(f"perop_register_block_nr{n}" for n in (2, 4, 8, 32)),
    "WHOLEMODEL_VF_BMMPAD_NAME": "accumulator_resident_wholemodel_vf_bmmpad",
}


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    if name in _COMPAT_CONSTANTS:
        return _COMPAT_CONSTANTS[name]
    for module_name in _PROVIDER_MODULES:
        try:
            module = _provider_module(module_name)
        except (OSError, ValueError):
            break
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(name)
