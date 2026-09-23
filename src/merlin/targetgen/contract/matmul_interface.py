"""Pure single-matmul interface emission shared by corpus derivation and exporters."""

from __future__ import annotations

from typing import Any


def _header(target: str) -> str:
    return (
        'module attributes {merlin_iface.version = "0.1", '
        f'merlin_iface.target = "{target}", merlin_iface.abi_version = "0.1"}} {{'
    )


def _dt(odt: str) -> str:
    return odt


def emit_interface_mlir(
    *,
    lhs: str,
    weight: str,
    out: str,
    M: int,
    K: int,
    N: int,
    epilogue: list[str],
    output_dtype: str,
    acc_scale: float | None = None,
    comment: str = "",
    target: str,
    operand_dtype: str = "i8",
    acc_dtype: str = "i32",
    scale_block: int | None = None,
    scale_dtype: str = "i8",
    pool_attrs: dict[str, Any] | None = None,
    commit_rows: int | None = None,
    bias: str | None = None,
    bias_dtype: str | None = None,
    requant_shift: int | None = None,
) -> str:
    """Emit a single-matmul merlin_iface module (weight-stationary). The target must be explicit; pass the
    target's derived MLIR dtype spellings (e.g. ``f8E4M3FN``/``bf16`` for a float MXU) to emit its ISA.

    ``scale_block`` declares a BLOCK-SCALED operand format: one shared exponent per run of that many K
    elements. A microscaling operand is (elements, per-block scales) -- declaring only the elements, as
    this interface used to, hands a backend half a number and asks it for the product, which no compiler
    can supply. When set, the two scale streams are declared as first-class input tensors alongside the
    operands they scale. Omitted (None) leaves the module byte-identical for an unscaled target.

    ``bias`` names the per-column bias vector a ``bias_add`` epilogue consumes. It is a DECLARED operand
    for the same reason the scale streams are: the commit attribute ``epilogue = ["bias_add"]`` says a
    bias is added but not what to add, and a backend cannot reconstruct corpus data from the element
    bytes. ``bias_dtype`` defaults to ``acc_dtype`` because that is the domain the addition happens in --
    the bias lands on the accumulator, before any requant -- so declaring it in the operand dtype would
    describe a different computation from the one the golden performs. Omitted (None) leaves the module
    byte-identical for a capsule with no bias stage.

    ``requant_shift`` is the integer ``requant`` stage's round-half-up shift, and it is declared for the
    same reason the bias operand is: ``epilogue = ["requant"]`` says the accumulator is shifted but not
    by how much, and the three engines that grade the result each carry their own fallback -- so an
    undeclared shift makes the golden and the reference agree with each other while the backend is handed
    a stage with no parameter. The interface dialect's own verifier already refuses this pairing
    (``interface.commit epilogue has 'requant' but no 'requant_shift'``); it is checked here too so the
    module is refused where it is WRITTEN rather than wherever it is next parsed."""
    epi = ", ".join(f'"{e}"' for e in epilogue)
    commit_attrs = f'name = "{out}", epilogue = [{epi}], output_dtype = "{output_dtype}"'
    if acc_scale is not None:
        commit_attrs += f", acc_scale = {acc_scale} : f32"
    if bias is not None:
        # Names the operand the stage consumes. The golden engine reads exactly this attribute
        # (`capsule_golden._apply_epilogue` -> `attrs["bias"]`), so a module that declares the stage
        # without it and a golden that adds nothing would agree with each other and both be wrong.
        commit_attrs += f', bias = "{bias}"'
    # ⚠️ SAME PAIRING RULE AS THE POOL GEOMETRY BELOW, and the same two silent wrong answers when it is
    # broken: a shift with no stage is a parameter nothing reads, a stage with no shift is a parameter
    # every engine invents for itself.
    _requanted = "requant" in epilogue
    if requant_shift is not None and not _requanted:
        raise ValueError(
            f"requant_shift={requant_shift} was given but the epilogue {epilogue} declares no 'requant' "
            f"stage: the commit would carry a shift nothing applies"
        )
    if _requanted and requant_shift is None:
        raise ValueError(
            "a 'requant' epilogue needs its requant_shift declared; without it the golden, the reference "
            "and the simulator each fall back to their own shift and agree by coincidence while the "
            "backend is handed a stage whose one parameter nobody stated"
        )
    if _requanted:
        commit_attrs += f", requant_shift = {int(requant_shift)} : i64"
    # A POOLING epilogue is the one stage that changes the committed extent: the accumulator's rows
    # unflatten to a plane and pool down, so the commit result type is `commit_rows`, not M. Both the
    # geometry and the row count are passed in (computed once by the corpus builder) rather than
    # recomputed here -- two copies of the same formula is exactly how a golden and an emitted module
    # come to describe different tensors.
    #
    # ⚠️ THE STAGE AND ITS GEOMETRY MUST ARRIVE TOGETHER, and each half alone is a different silent
    # wrong answer. Geometry with no stage writes pool attributes onto a commit whose epilogue list
    # says nothing is pooled: a backend reading the epilogue commits the raw matrix while the module
    # looks pooled to anything reading the attributes. A stage with no geometry declares `maxpool` and
    # supplies neither window nor pooled extent, so the epilogue is silently skipped and the committed
    # type is M rows of unpooled accumulator. Neither raises anything downstream -- both just compute
    # the wrong tensor and agree with themselves. When this function gained `pool_attrs` the mismatched
    # call was a TypeError, which was at least loud; accepting the argument without checking the pair
    # is what turned it quiet.
    _pooled = "maxpool" in epilogue
    if pool_attrs and not _pooled:
        raise ValueError(
            f"pool geometry {sorted(pool_attrs)} was given but the epilogue {epilogue} declares no "
            f"'maxpool' stage: the commit would carry pool attributes while committing the raw matrix"
        )
    if _pooled and not (pool_attrs and commit_rows is not None):
        raise ValueError(
            "a 'maxpool' epilogue needs both pool_attrs (the window) and commit_rows (the pooled "
            f"extent); got pool_attrs={pool_attrs!r}, commit_rows={commit_rows!r}, so the stage would "
            "be silently skipped and M unpooled rows committed instead"
        )
    for k, v in (pool_attrs or {}).items():
        commit_attrs += (
            f", {k} = [{', '.join(str(int(x)) for x in v)}]" if isinstance(v, list) else f", {k} = {int(v)} : i64"
        )
    rows = M if commit_rows is None else int(commit_rows)
    lines = []
    if comment:
        lines.append(f"// {comment}")
    lines += [
        _header(target),
        f'  %{weight} = merlin_iface.tensor {{name = "{weight}", role = "weight"}} : tensor<{K}x{N}x{operand_dtype}>',
        f'  %{lhs} = merlin_iface.tensor {{name = "{lhs}", role = "input"}} : tensor<{M}x{K}x{operand_dtype}>',
    ]
    if bias is not None:
        lines.append(
            f'  %{bias} = merlin_iface.tensor {{name = "{bias}", role = "bias"}} : '
            f"tensor<{N}x{bias_dtype or acc_dtype}>"
        )
    if scale_block and K % scale_block == 0:
        # A block-scaled datapath consumes one E8M0 exponent per `scale_block` K elements, per row of the
        # lhs and per column of the weight. They are declared operands, not hidden state: the scales are
        # corpus data a backend cannot reconstruct from the element bytes.
        g = K // scale_block
        lines += [
            f'  %{lhs}_scale = merlin_iface.tensor {{name = "{lhs}_scale", role = "scale", '
            f'scale_of = "{lhs}", block = {scale_block} : i64}} : tensor<{g}x{M}x{scale_dtype}>',
            f'  %{weight}_scale = merlin_iface.tensor {{name = "{weight}_scale", role = "scale", '
            f'scale_of = "{weight}", block = {scale_block} : i64}} : tensor<{g}x{N}x{scale_dtype}>',
        ]
    lines += [
        f'  %{weight}_res = merlin_iface.resident_pack %{weight} {{layout = "packed_rhs"}} '
        f": (tensor<{K}x{N}x{operand_dtype}>) -> !merlin_iface.resident",
        f"  %acc0 = merlin_iface.matmul %{lhs}, %{weight}_res "
        f": (tensor<{M}x{K}x{operand_dtype}>, !merlin_iface.resident) -> !merlin_iface.acc<{acc_dtype}>",
        f"  %{out} = merlin_iface.commit %acc0 {{{commit_attrs}}} "
        f": (!merlin_iface.acc<{acc_dtype}>) -> tensor<{rows}x{N}x{_dt(output_dtype)}>",
        f"  merlin_iface.evict %{weight}_res : (!merlin_iface.resident) -> ()",
        "}",
    ]
    return "\n".join(lines) + "\n"
