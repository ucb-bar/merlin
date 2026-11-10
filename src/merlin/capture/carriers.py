"""Observe linalg carriers, operand formats and contraction geometry structurally."""

from __future__ import annotations

_OP_MARKER: dict[str, tuple[str, ...]] = {
    # A contraction may arrive tagged `matmul` (a float `linalg.matmul`) or `int_matmul` (the
    # `aten._int_mm` form a W8A8 capture emits, a linalg.generic accumulating in i32). Both ARE the
    # contraction; matching only the float spelling refused the very capture that carries the integer
    # arithmetic a systolic mesh runs -- the program was right and the check was reading for the wrong
    # word.
    "matmul": ("matmul", "int_matmul"),
    "linear": ("matmul", "int_matmul"),
    "attention_qk": ("matmul", "int_matmul"),
    "rmsnorm": ("rsqrt",),
}


def _tensor_types(s: str) -> list[tuple[list[int], str]]:
    """Structurally tokenize the ``tensor<...>`` types in a fragment (str.split, NO regex): each becomes
    ``([dims], dtype)``. Dynamic/non-integer dims are skipped (a shaped op capsule is always static)."""
    out: list[tuple[list[int], str]] = []
    for chunk in s.split("tensor<")[1:]:
        body = chunk.split(">", 1)[0]  # e.g. "16x16xf32"
        parts = body.split("x")
        if len(parts) < 2:
            continue
        *dims, dt = parts
        try:
            out.append(([int(d) for d in dims], dt))
        except ValueError:
            continue  # a dim that isn't an int -> not a static shape
    return out


def _carrying_ops(linalg_mlir: str) -> list[str]:
    """The MLIR operation that carries each ``prov.op`` tag, in tag order. ``""`` when unreadable.

    WHY THIS IS NEEDED. ``prov.op`` is provenance: m2m stamps the SOURCE operation a region came from
    onto every op the lowering produced for it. One captured convolution therefore tags its contraction
    AND the ``tensor.expand_shape`` / ``collapse_shape`` / ``splat`` / ``arith.constant`` ops around it,
    all with ``prov.op = "convolution_im2col_matmul"``. Counting the tag instead of the operation minted
    one routing demand per tag: measured on the ResNet-50 int8 capture, 1199 tags over 497 real compute
    operations, so 702 shape and metadata ops were asked whether the accelerator supports a contraction,
    refused, and counted onto the scalar lane. The headline "1,193 of 1,247 demands are scalar" was
    mostly that.

    Structural, no regex: the op name is the first dialect-qualified token on the tag's own line that is
    not an attribute key (an attribute key is followed by ``=``; an op name is not). Both spellings are
    read -- the pretty form ``%x = linalg.generic {...}`` and the quoted generic form
    ``%x = "tensor.insert_slice"(...)``, which is how 18 of this capture's ops are printed.
    """
    out: list[str] = []
    for segment in linalg_mlir.split('prov.op = "')[:-1]:
        line = segment.rsplit("\n", 1)[-1]
        tokens = line.replace("(", " ").replace(")", " ").replace(",", " ").replace("{", " { ").split()
        name = ""
        for index, token in enumerate(tokens):
            bare = token.strip('"')
            if "." not in bare or not bare.replace(".", "").replace("_", "").isalnum():
                continue
            following = tokens[index + 1] if index + 1 < len(tokens) else ""
            if following == "=":  # an attribute key, not the operation
                continue
            name = bare
            break
        out.append(name)
    return out


def _carrier_families(linalg_mlir: str) -> list[str]:
    """The ``prov.family`` of the operation carrying each ``prov.op`` tag, in tag order. ``""`` when
    that operation states no family.

    Read from the SAME LINE as the tag, like :func:`_carrying_ops`, because the two tags are NOT in
    one-to-one correspondence and splitting on them separately silently misaligns the lists. Measured
    on `M1_lstmnetvit_fp32`: 732 ``prov.op`` tags against 723 ``prov.family`` tags -- nine operations
    state an op and no family -- so a caller indexing the family list by the op list's position was
    reading the NEXT operation's family for everything from tag 662 onward, and the families of the
    model's tail were attributed to the wrong ops. The lengths differ, so the defect is detectable; the
    old code hid it behind ``fams[i] if i < len(fams) else ""``, which only ever notices at the very end.
    """
    out: list[str] = []
    segments = linalg_mlir.split('prov.op = "')
    for index in range(len(segments) - 1):
        line = segments[index].rsplit("\n", 1)[-1] + segments[index + 1].split("\n", 1)[0]
        _, sep, rest = line.partition('prov.family = "')
        out.append(rest.split('"', 1)[0] if sep else "")
    return out


def _carrier_reduces(linalg_mlir: str) -> list[bool | None]:
    """Whether the operation carrying each ``prov.op`` tag REDUCES, in tag order. ``None`` when the
    operation states no iteration space at all (a `tensor.*` or `arith.*` op has none).

    Read from the SAME LINE as the tag, exactly as :func:`_carrying_ops` reads the operation name, so
    the two lists are aligned by construction. That matters more than it sounds: an earlier census read
    the flag from the module's ``linalg.generic`` segments in text order and the family from the per-op
    lists, and indexed one by a counter over the other -- which silently assumes every generic in the
    text is a tagged carrier. Three capsules break it (one has 447 generics against 444 carriers), so
    from the first mismatch onward the flag came off the wrong operation. Checked over the corpus: all
    20,017 generic-carried tags state ``iterator_types`` on the tag's own line.

    WHY IT IS NEEDED. ``prov.family`` is stamped on every op of a captured region, so an im2col GATHER
    -- pure data movement, every iterator parallel -- carries ``contraction`` because the convolution it
    belongs to is one. Asking the mesh to support it is the same error :func:`_carrying_ops` was written
    against, one level in: that fix excluded the region's `tensor.*` ops by requiring a `linalg.` carrier,
    and a parallel-only `linalg.generic` passes that test while being just as much not a contraction.
    """
    out: list[bool | None] = []
    for segment in linalg_mlir.split('prov.op = "')[:-1]:
        line = segment.rsplit("\n", 1)[-1]
        head, sep, rest = line.partition("iterator_types")
        if not sep:
            out.append(None)
            continue
        # The iterator list is the bracketed group right after the key; nothing past it can contribute.
        iterators = rest.partition("[")[2].partition("]")[0]
        out.append("reduction" in iterators)
    return out


def _tag_lines(linalg_mlir: str) -> list[str]:
    """The whole source LINE carrying each ``prov.op`` tag, in tag order.

    Every per-op fact below is read from this one string, for the reason
    :func:`_carrier_families` and :func:`_carrier_reduces` document: ``prov.op`` is not in one-to-one
    correspondence with anything else in the module, so a list built by splitting on some OTHER token and
    indexed by the tag's position silently describes the wrong operation from the first mismatch onward.
    Measured on this corpus: 19 capsules state more ``prov.op = "matmul"`` tags than they contain
    ``linalg.matmul`` operations (``M1_lstmnetvit_gemmini``: 52 against 37), because a contraction may
    also be spelled as a ``linalg.generic`` -- so the extent reader that walked the named matmuls with its
    own counter handed 15 of that capsule's demands another layer's shape.

    A tag's operation is printed on one line in this IR (a ``linalg.generic``'s maps, iterators, operands
    and attributes are all on its header line, its region follows), and no line in the corpus carries two
    ``prov.op`` tags, so the line IS the operation.
    """
    segments = linalg_mlir.split('prov.op = "')
    return [segments[i].rsplit("\n", 1)[-1] + segments[i + 1].split("\n", 1)[0] for i in range(len(segments) - 1)]


def _ins_types(line: str) -> list[tuple[list[int], str]]:
    """The ``ins(...)`` operand tensor types of a tag line (``outs`` terminates the clause)."""
    if "ins(" not in line:
        return []
    return _tensor_types(line.split("ins(", 1)[1].split("outs", 1)[0])


def _outs_types(line: str) -> list[tuple[list[int], str]]:
    if "outs(" not in line:
        return []
    return _tensor_types(line.split("outs(", 1)[1])


def _ins_slot_formats(line: str) -> tuple[str | None, ...]:
    """Read EVERY pretty-printed ``ins`` slot, retaining unknown/dynamic slots in place."""
    from merlin.common import quant_formats as qf

    if "ins(" not in line:
        return ()
    clause = line.split("ins(", 1)[1].split("outs(", 1)[0]
    names, separator, types = clause.partition(":")
    slots = [name.strip() for name in names.strip().removesuffix(")").split(",") if name.strip()]
    if not separator or not slots:
        return tuple(None for _ in slots)
    chunks = types.split("tensor<")[1:]
    if len(chunks) != len(slots):
        return tuple(None for _ in slots)
    result: list[str | None] = []
    for chunk in chunks:
        body = chunk.split(">", 1)[0]
        parts = body.split("x")
        if len(parts) < 2 or any(not dim.isdigit() or int(dim) <= 0 for dim in parts[:-1]) or not qf.has(parts[-1]):
            result.append(None)
        else:
            result.append(qf.get(parts[-1]).name)
    return tuple(result)


def _element_format(line: str) -> str | None:
    """The canonical format name every ``ins`` operand of this line shares, or ``None`` (UNKNOWN).

    Resolved through the format registry (:mod:`merlin.common.quant_formats`), so the MLIR spelling ``i8``
    and a contract's ``int8`` are recognised as ONE format rather than two -- the same resolution
    ``routing._fmt_ok`` performs, deliberately, so the two sides cannot disagree about what a format is.

    ``None`` on every ambiguity, because an operand format that is guessed is worse than one that is
    absent: no ``ins`` clause, an element type the registry does not know (``i1``/``i32``/``i64``
    predicates, indices and accumulators are not operand formats), or operands that disagree with each
    other (a select over ``f32`` and ``i1`` has no single operand format and must not be given one).
    """
    formats = _ins_slot_formats(line)
    if not formats or any(fmt is None for fmt in formats):
        return None
    names = set(formats)
    return names.pop() if len(names) == 1 else None


def _affine_maps(line: str) -> list[tuple[list[str], list[str]]] | None:
    """``indexing_maps`` as ``[(domain dims, result dims)]``, or ``None`` when there are none/unreadable.

    Structural, no regex: the bracketed group after the key, split on ``affine_map<``. The arrow is
    protected before the closing ``>`` is sought, because ``->`` contains one -- reading
    ``affine_map<(d0, d1) -> (d0, d1)>`` by the first ``>`` yields ``(d0, d1) -`` and refuses every map in
    the corpus, which is a failure that fails CLOSED and therefore stays silent.
    """
    if "indexing_maps" not in line:
        return None
    body = line.split("indexing_maps", 1)[1].partition("[")[2].partition("]")[0]
    maps: list[tuple[list[str], list[str]]] = []
    for chunk in body.split("affine_map<")[1:]:
        inner = chunk.replace("->", "\x00").split(">", 1)[0].replace("\x00", "->")
        domain, arrow, results = inner.partition("->")
        domain, results = domain.strip(), results.strip()
        if not arrow or not domain.startswith("(") or not domain.endswith(")"):
            return None
        if not results.startswith("(") or not results.endswith(")"):
            return None
        dims = [t.strip() for t in domain[1:-1].split(",") if t.strip()]
        res = [t.strip() for t in results[1:-1].split(",")] if results[1:-1].strip() else []
        maps.append((dims, res))
    return maps or None


def _iterator_kinds(line: str) -> list[str] | None:
    if "iterator_types" not in line:
        return None
    body = line.split("iterator_types", 1)[1].partition("[")[2].partition("]")[0]
    return [t.strip().strip('"') for t in body.split(",") if t.strip()] or None


def _generic_contraction_extents(line: str) -> tuple[int, int, int, int] | None:
    """``(batch, M, K, N)`` for a ``linalg.generic`` that is UNAMBIGUOUSLY a contraction, else ``None``.

    An im2col'd convolution reaches the capture as a batched-matmul ``linalg.generic`` -- indexing maps
    ``(d0,d1,d3),(d0,d3,d2)->(d0,d1,d2)`` over ``[parallel, parallel, parallel, reduction]`` -- and the
    extent reader understood named ``linalg.matmul`` only, so the site carried no shape, no tile could be
    synthesized for it, and it counted ``n_unsynthesizable`` against the capsule at its cert tier.

    The shape is derived from the maps and the operand types rather than from the op's spelling, and every
    step that cannot be established REFUSES the whole op rather than filling in a plausible value:

    * exactly three maps over two ``ins`` operands and one ``outs``, all sharing one iteration domain;
    * every map result is a bare domain dimension, used once -- which is what rejects a real convolution
      (``(d2 * 2) + d4``) and a broadcast/diagonal access;
    * every iterator's bound is projected by some operand dimension and all its projections agree;
    * the output indexes exactly the parallel iterators, so the op accumulates rather than writing
      distinct cells under a reduction-labelled iterator;
    * at least one reduction iterator, and every reduction iterator is read by BOTH inputs and by neither
      output -- an "unshared" reduction is a row-wise reduce, not a contraction.

    The parallel iterators then partition themselves: read by both inputs -> batch, by the left only -> M,
    by the right only -> N. A parallel iterator read by neither, or absent from the output, refuses the op.
    """
    maps = _affine_maps(line)
    iterators = _iterator_kinds(line)
    if maps is None or iterators is None:
        return None
    ins, outs = _ins_types(line), _outs_types(line)
    if len(maps) != 3 or len(ins) != 2 or len(outs) != 1:
        return None
    domain = maps[0][0]
    if any(m[0] != domain for m in maps) or len(domain) != len(iterators):
        return None
    if any(kind not in ("parallel", "reduction") for kind in iterators):
        return None
    position = {d: i for i, d in enumerate(domain)}
    projections: list[list[int]] = []
    for _dims, results in maps:
        if any(r not in position for r in results):
            return None  # not a bare dimension: an affine expression, a symbol, a constant
        indices = [position[r] for r in results]
        if len(set(indices)) != len(indices):
            return None  # the same iterator twice in one access -- a diagonal, not a contraction
        projections.append(indices)
    bounds: list[int | None] = [None] * len(domain)
    for indices, (shape, _dt) in zip(projections, [ins[0], ins[1], outs[0]], strict=True):
        if len(indices) != len(shape):
            return None
        for p, extent in zip(indices, shape, strict=True):
            if bounds[p] is not None and bounds[p] != extent:
                return None  # two operands disagree about one iterator's trip count
            bounds[p] = extent
    if any(b is None for b in bounds):
        return None
    lhs, rhs, out = (set(p) for p in projections)
    reduction = {i for i, kind in enumerate(iterators) if kind == "reduction"}
    parallel = {i for i, kind in enumerate(iterators) if kind == "parallel"}
    if not reduction or out != parallel:
        return None
    if not (reduction <= lhs and reduction <= rhs):
        return None
    batch_dims, m_dims, n_dims = (lhs & rhs) & parallel, (parallel & lhs) - rhs, (parallel & rhs) - lhs
    if batch_dims | m_dims | n_dims != parallel:
        return None

    def _product(indices) -> int:
        total = 1
        for i in indices:
            total *= int(bounds[i])
        return total

    return _product(batch_dims), _product(m_dims), _product(reduction), _product(n_dims)


def _named_matmul_extents(line: str) -> tuple[int, int, int, int] | None:
    """``(batch, M, K, N)`` for a named ``linalg.matmul`` tag line, or ``None``.

    The plain case is ``ins(tensor<MxK>, tensor<KxN>)``. The shared reduction extent is CHECKED rather
    than assumed, which is also what makes ``prov.transposed_b`` harmless: that tag is provenance about
    the source ``aten.linear`` (whose weight is stored ``[N, K]``), not a statement about the operand
    order of the emitted op -- and were it one, the only shapes that pass ``K == K2`` are the ones where
    both readings give the same ``(M, K, N)``.

    A capture also emits a rank-3 left operand against a rank-2 right one -- a batched contraction over a
    SHARED weight, which two of these capsules spell as ``linalg.matmul``. Its leading dims are batch and
    its trailing pair is ``(M, K)``. The old reader took the first two dims of whatever it found, so a
    ``1x345x32 @ 32x32`` layer was recorded as ``M=1, K=345, N=32``: the right three numbers in the wrong
    roles, on 40 sites. The output shape is cross-checked against the batch+M+N the operands imply, so a
    shape that does not compose refuses instead of being reinterpreted.
    """
    types = _ins_types(line)
    if len(types) < 2:
        return None
    lhs, rhs = types[0][0], types[1][0]
    if len(rhs) != 2 or len(lhs) < 2 or lhs[-1] != rhs[0]:
        return None
    batch_dims, m, k, n = lhs[:-2], lhs[-2], lhs[-1], rhs[1]
    outs = _outs_types(line)
    if outs and outs[0][0] != [*batch_dims, m, n]:
        return None
    total = 1
    for d in batch_dims:
        total *= d
    return total, m, k, n


def _carrier_extents(linalg_mlir: str) -> list[tuple[int, int, int, int] | None]:
    """``(batch, M, K, N)`` for the operation carrying each ``prov.op`` tag, in tag order; ``None`` when
    the operation states no contraction shape this reader can establish.

    Both spellings a capture uses, read from the tag's OWN line (see :func:`_tag_lines`): a named
    ``linalg.matmul`` over two rank-2 operands, and a ``linalg.generic`` whose maps make it a contraction
    (:func:`_generic_contraction_extents`). ``linalg.matmul_transpose_b`` and friends are NOT read here --
    their ``ins`` types describe a transposed operand pair, so reading them as ``MxK, KxN`` records the
    wrong shape. Fail closed: an operation this cannot read yields ``None``, exactly as if no extent had
    ever been derivable for it.
    """
    out: list[tuple[int, int, int, int] | None] = []
    for line, carrier in zip(_tag_lines(linalg_mlir), _carrying_ops(linalg_mlir), strict=True):
        if carrier == "linalg.matmul":
            out.append(_named_matmul_extents(line))
        elif carrier == "linalg.generic":
            out.append(_generic_contraction_extents(line))
        else:
            out.append(None)
    return out


def _carrier_formats(linalg_mlir: str) -> list[str | None]:
    """The canonical element format of the operation carrying each ``prov.op`` tag, in tag order."""
    return [_element_format(line) for line in _tag_lines(linalg_mlir)]


def _carrier_input_formats(linalg_mlir: str) -> list[tuple[str | None, ...]]:
    """Per-tag pretty ``ins`` slots, with no dropped unknown operand."""
    return [_ins_slot_formats(line) for line in _tag_lines(linalg_mlir)]


def _generic_form_linalg_summary(linalg_mlir: str) -> dict:
    """Read a generic-printed capture from operations, not the tag's physical line.

    MLIR prints a generic op's attributes *after* its nested region.  The old
    line reader therefore saw no carrier, iterator types, shape or operand
    format for every generic-printed contraction.  This is the same parsed
    program used by the independent completeness check; the two observations
    remain separate because this reader uses provenance and the other checks
    the computation body.
    """
    from math import prod

    from merlin.common import mlir_query as mq
    from merlin.common import quant_formats as qf
    from merlin.compiler.ir import contraction_shapes as ks

    module = mq.parse(linalg_mlir)
    shapes = {id(op): shape for op, shape in ks.observe_contractions(module)}
    rows = {
        "prov_ops": [],
        "prov_families": [],
        "carrier_region_ids": [],
        "carrying_ops": [],
        "carrier_reduces": [],
        "carrier_extents": [],
        "carrier_formats": [],
        "carrier_input_formats": [],
    }
    for op in mq.walk(module):
        tag = mq.attr_str(op, "prov.op")
        if tag is None:
            continue
        name = mq.op_name(op)
        rows["prov_ops"].append(tag)
        rows["prov_families"].append(mq.attr_str(op, "prov.family") or "")
        rows["carrier_region_ids"].append(mq.attr_str(op, "prov.region_id"))
        rows["carrying_ops"].append(name)
        iterators = (op.properties or {}).get("iterator_types") or op.attributes.get("iterator_types")
        rows["carrier_reduces"].append("reduction" in str(iterators) if iterators is not None else None)
        shape = shapes.get(id(op))
        if shape is not None and len(shape.parallel) >= 2 and shape.reduction:
            parallel = tuple(int(x) for x in shape.parallel)
            rows["carrier_extents"].append((prod(parallel[:-2]), parallel[-2], prod(shape.reduction), parallel[-1]))
        else:
            rows["carrier_extents"].append(None)
        fmt = None
        input_formats: tuple[str | None, ...] = ()
        if name.startswith("linalg."):
            # Linalg's trailing operands initialize its results; only the
            # preceding operands specify the computation's input format.
            inputs = op.operands[: len(op.operands) - len(op.results)]
            input_formats = tuple(
                qf.get(spelling).name
                if qf.has(spelling) and shape is not None and all(type(dim) is int and dim > 0 for dim in shape)
                else None
                for value in inputs
                for spelling in (str(value.type.element_type) if hasattr(value.type, "element_type") else "",)
                for shape in (value.type.get_shape() if hasattr(value.type, "get_shape") else None,)
            )
            if input_formats and all(input_formats):
                names = set(input_formats)
                if len(names) == 1:
                    fmt = names.pop()
        rows["carrier_formats"].append(fmt)
        rows["carrier_input_formats"].append(input_formats)
    try:
        inputs, outputs = mq.forward_signature(module)
    except ValueError:
        inputs, outputs = [], []
    return {**rows, "inputs": inputs, "output": outputs[0] if outputs else None}
