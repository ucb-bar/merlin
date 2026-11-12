"""Typed complete result rosters for ordinary captured operation programs."""

from __future__ import annotations


def _shape(value):
    if not isinstance(value, list):
        return []
    if not value:
        return [0]
    shape = _shape(value[0])
    if any(_shape(child) != shape for child in value):
        raise ValueError("captured result is not a rectangular tensor")
    return [len(value), *shape]


def publication(art, entry):
    """Require ABI, actual program signature and full golden for every result."""
    from merlin.common.mlir_query import forward_signature

    abi = (art.meta or {}).get("output_abi")
    if not isinstance(abi, list) or len(abi) < 2:
        raise ValueError("multiple-result operation requires a complete typed output ABI")
    _, actual = forward_signature(art.linalg_mlir)
    if len(actual) != len(abi):
        raise ValueError("captured result ABI differs from the actual program signature")
    if not isinstance(art.golden, list) or len(art.golden) != len(abi):
        raise ValueError("captured operation omitted a complete result golden")
    names = entry.get("outs") or ["Y" + str(index) for index in range(len(abi))]
    if (
        not isinstance(names, list)
        or len(names) != len(abi)
        or any(not isinstance(name, str) or not name or not name.isidentifier() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("captured operation requires one unique safe name per published result")
    for declared, typed, golden in zip(abi, actual, art.golden, strict=True):
        if not isinstance(declared, dict) or (declared.get("shape"), declared.get("dtype")) != typed:
            raise ValueError("captured result ABI differs from the actual typed program result")
        if _shape(golden) != declared["shape"]:
            raise ValueError("captured full result shape differs from its typed output ABI")
    return dict(zip(names, art.golden, strict=True))
