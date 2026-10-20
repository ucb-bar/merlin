"""Select a capsule writer without broadening its numeric regime."""

_PYTORCH_REGIMES = ("simt",)


def select_writer(entry: dict, *, source_for_op) -> str | None:
    """Which writer can materialize this entry -- ``builder`` / ``pytorch`` / ``None`` for neither.

    ``available_ops`` is the union of the direct-MLIR BUILDERS and the PyTorch bodies, so an op can be
    "materializable" in the abstract and still have no writer AT THIS DTYPE: the PyTorch path is float
    only. A cell in that corner is a real capability gap and must be reported as an uncovered cell to
    argue about, never emitted as an entry nothing can write.
    """
    from merlin.targetgen.corpus_spec import BUILDERS, regime_for_dtype
    from .capsule_builtin_source import native_storage_source_spec

    op = entry.get("op")
    if native_storage_source_spec(entry) is not None:
        return "pytorch" if source_for_op(op) == "pytorch" else None
    if op == "dtype_cast":
        return None
    if op in BUILDERS:
        return "builder"
    try:
        regime = regime_for_dtype(str(entry.get("operand_dtype") or ""))
    except Exception:  # noqa: BLE001 -- an unknown regime is not a float one
        regime = None
    return "pytorch" if regime in _PYTORCH_REGIMES else None



def mark_source(entry: dict) -> None:
    """Say which writer can materialize this entry's op, when it is not the direct-MLIR one.

    `available_ops` is the union of the direct-MLIR BUILDERS and the PyTorch bodies, so an op can be
    materializable and still have no builder -- and an entry that does not say so goes down the
    direct-MLIR path and dies with "no corpus builder for op". Measured: four synthesized capsules
    failed exactly that way on the first target whose elementwise family resolved to `gelu`.
    """
    from merlin.targetgen.corpus_spec import BUILDERS

    op = entry.get("op")
    if op and op not in BUILDERS:
        entry["source"] = "pytorch"

