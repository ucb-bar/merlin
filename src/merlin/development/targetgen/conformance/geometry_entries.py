"""Convolution, capacity and shape-geometry obligations for derived profiles."""

from __future__ import annotations

import copy
from math import isqrt
from typing import Any

from merlin.development.targetgen.corpus.selection import (
    _tile_int,
    cap_to_affordable,
    extents_for,
    narrowest_admitted,
)


def add_geometry_entries(
    *,
    SOURCE_ROLE,
    SYNTH_PREFIX,
    _CONV_OPS_BY_RANK,
    _activation_contraction_entries,
    _image_for_window,
    _quanta,
    _signed_stimulus_range,
    _tail_isolation_entries,
    _writer_for,
    admitted_dtypes,
    declare_pool_window,
    declare_residual_add_params,
    entries,
    kept,
    mem_block,
    op_for_family,
    ops_gradeable_at,
    pass_requirements_for,
    pool,
    probes,
    regime_dtype,
    spec_doc,
    unexpressable,
):
    conv_dtype = regime_dtype or (kept[0] if kept else narrowest_admitted(admitted_dtypes))
    for _cw in (spec_doc.get("conv_geometry") or {}).get("required") or ():
        sig = str(_cw.get("signature") or "")
        if not sig:
            continue
        kern = [int(v) for v in (_cw.get("kernel") or ())]
        if not conv_dtype:
            unexpressable.append(f"conv-window axis: no admitted dtype to build {sig!r} at")
            continue
        _op = next((o for o in _CONV_OPS_BY_RANK.get(len(kern), ()) if o in pool), None)
        if _op is None:
            # An absent OP, not an absent obligation -- and the two must not read the same. A target
            # whose pool has no rank-N convolution cannot be asked for this window; a target that has
            # one and lacks the member is uncovered.
            unexpressable.append(
                f"conv-window axis: this target's op pool materializes no rank-{len(kern)} "
                f"convolution, so window {sig} has no member here"
            )
            continue
        pad_known = bool(_cw.get("pad_known"))
        pb = [int(v) for v in (_cw.get("pad_before") or ())] if pad_known else [0] * len(kern)
        pa = [int(v) for v in (_cw.get("pad_after") or ())] if pad_known else [0] * len(kern)
        entry = {
            "cat": "layers",
            "kind": "layer",
            "name": f"{SYNTH_PREFIX}_conv_{sig.replace('/', '_').replace('-', '_')}",
            "op": _op,
            "operand_dtype": conv_dtype,
            "lhs": "A0",
            "weight": "W",
            "out": "Y0",
            "kh": kern[0],
            "kw": kern[-1],
            "stride": [int(v) for v in (_cw.get("stride") or ())] or [1] * len(kern),
            "dilation": [int(v) for v in (_cw.get("dilation") or ())] or [1] * len(kern),
            "padding": list(pb) + list(pa),
            # ⚠️ THE IMAGE IS SIZED TO THE WINDOW, NEVER LEFT AT THE DEFAULT. `conv2d` defaults to an
            # 8x8 image, and a large window walks off it: measured, k16x16/s16x16 gives an output of
            # 0x0 rows and k8x8/s8x8 gives 1x1. A zero-row capsule is not a small test, it is a test
            # of nothing that still builds, still runs and still passes. So H and W are solved from
            # the window for a fixed small output -- the inverse of `conv_out_dims` -- which keeps
            # every window's member the same size in the thing certification is actually priced on
            # (written output) regardless of how large the window is.
            **dict(
                zip(
                    ("Himg", "Wimg"),
                    _image_for_window(
                        kern,
                        [int(v) for v in (_cw.get("stride") or ())] or [1] * len(kern),
                        [int(v) for v in (_cw.get("dilation") or ())] or [1] * len(kern),
                        pb,
                        pa,
                    ),
                )
            ),
            "source_role": SOURCE_ROLE,
            "source_reference": (
                f"synthesized for the convolution-window axis: window {sig}, recovered structurally "
                f"from {_cw.get('n_regions')} region(s) in selected captures. A captured convolution "
                f"carries no padding/stride/dilation attribute, so the geometry comes from the "
                f"indexing of its windows and the producer of its padding"
                + (
                    ""
                    if pad_known
                    else ". THE PADDING WAS NOT READABLE in the capture -- it is applied by an index gather, "
                    "i.e. a reflection pad -- so this member tests the WINDOW and its stepping at zero "
                    "padding and asserts nothing about a padding identity that is not zero"
                )
            ),
            "label": "public",
            # THE OBLIGATION THIS MEMBER DISCHARGES, named. A `padUNKNOWN` window cannot be matched by
            # re-deriving a signature from the member's attributes: the member declares zero padding
            # (it is the only padding we can spell) and would re-derive as `pad0x0`, so the obligation
            # it exists for could never be satisfied by the member built for it. The declared
            # signature closes that, and the coverage reader still VERIFIES the window itself against
            # the attributes -- only the unspellable padding component is taken on the declaration.
            "generalization": {"generalization_axis": "conv_window", "conv_window": sig},
            **extents_for("aligned", probes, quantum=_quanta.get(conv_dtype)),
        }
        declare_pool_window(entry)
        declare_residual_add_params(entry)
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        entries.append(entry)

    # ---- the RESIDUAL-STATE axis ----------------------------------------------------------------------
    # What the device's stores held BEFORE the operation. A padded convolution that reads rows it never
    # wrote, or a first reduction step that accumulates onto rows it never cleared, is right on a device
    # whose stores start empty and wrong after any earlier command -- and the tensor-level tiers, which
    # model no store, cannot tell. Each padded window member and one partial contraction get a sibling
    # that runs an independent, signed, non-zero contraction first (a `prelude`), sized in tiles to
    # cover at least the rows the operation itself occupies, so stale state is non-zero when it runs.
    state_tile = int((spec_doc.get("boundaries") or {}).get("tile_edge") or 0)

    def _tiles(extent: int) -> str:
        return f"{max(1, -(-int(extent) // state_tile))}*tile"

    if state_tile > 0:
        for source in [
            e for e in entries if (e.get("generalization") or {}).get("generalization_axis") == "conv_window"
        ]:
            if source.get("op") != "conv2d" or not any(int(v) for v in source.get("padding") or ()):
                continue
            ci = int(source.get("ci", 4))
            stateful = {
                **copy.deepcopy(source),
                "name": f"{SYNTH_PREFIX}_residual_state_{source['name'].removeprefix(SYNTH_PREFIX + '_')}",
                "prelude": {
                    "M": _tiles(int(source["Himg"]) * int(source["Wimg"])),
                    "K": _tiles(int(source["kh"]) * int(source["kw"]) * ci),
                    "N": "tile",
                },
                "stimulus_range": list(_signed_stimulus_range()),
                "source_reference": (
                    f"synthesized for the residual-state axis: {source['name']} run after an independent "
                    "contraction leaves the operand and accumulator stores non-zero, so a padded border "
                    "read from rows the program never wrote is wrong"
                ),
                "generalization": {"generalization_axis": "residual_state"},
            }
            stateful["pass_requirements"] = pass_requirements_for(stateful, spec_doc)
            entries.append(stateful)
        partial = next(
            (e for e in entries if e.get("name") == f"{SYNTH_PREFIX}_contraction_{conv_dtype}_partial"), None
        )
        if partial is not None:
            stateful = {
                **copy.deepcopy(partial),
                "name": f"{SYNTH_PREFIX}_residual_state_contraction_partial",
                "cat": "layers",
                "kind": "layer",
                "prelude": {"M": "2*tile", "K": "2*tile", "N": "2*tile"},
                "stimulus_range": list(_signed_stimulus_range()),
                "source_reference": (
                    "synthesized for the residual-state axis: a partial-tile contraction run after an "
                    "independent contraction leaves the accumulator non-zero, so a first reduction step "
                    "that accumulates instead of overwriting is wrong"
                ),
                "generalization": {"generalization_axis": "residual_state"},
            }
            stateful["pass_requirements"] = pass_requirements_for(stateful, spec_doc)
            entries.append(stateful)

    # A small window capsule establishes convolution semantics, but cannot expose
    # a scheduler that keeps the *entire* im2col image in the operand store. Ask
    # for the first square output beyond that store's RTL-derived capacity, using
    # the most compact materializable 2-D window. The shape comes from facts and
    # the non-validation capture's window, never from a held-out model.
    conv_capacity_boundary: dict[str, Any] | None = None
    capacity_rows = int(mem_block.get("capacity_rows") or 0)
    tile_edge = int((spec_doc.get("boundaries") or {}).get("tile_edge") or 0)
    if capacity_rows > 0 and tile_edge > 0:
        candidates = []
        for source in entries:
            if (source.get("generalization") or {}).get("generalization_axis") != "conv_window":
                continue
            if source.get("op") != "conv2d":
                continue
            kh, kw, ci = int(source["kh"]), int(source["kw"]), int(source.get("ci", 4))
            if kh * kw <= 1 or ci <= 0:
                continue
            rows_per_window = (kh * kw * ci + tile_edge - 1) // tile_edge
            side = isqrt(capacity_rows // rows_per_window) + 1
            padding = [int(v) for v in source["padding"]]
            h, w = _image_for_window(
                [kh, kw], source["stride"], source["dilation"], padding[:2], padding[2:], output_extent=side
            )
            width = _tile_int(source.get("N"), tile_edge) or tile_edge
            unknown_pad = "UNKNOWN" in str((source.get("generalization") or {}).get("conv_window") or "")
            candidates.append(
                (
                    side * side * kh * kw * ci * width,
                    unknown_pad,
                    sum(abs(int(d) - 1) for d in source["dilation"]),
                    sum(abs(int(s) - 1) for s in source["stride"]),
                    sum(abs(padding[i] - padding[i + 2]) for i in range(2)),
                    h * w * ci,
                    source["name"],
                    side,
                    rows_per_window,
                    source,
                )
            )
        if candidates:
            _, _, _, _, _, _, _, side, rows_per_window, source = min(candidates)
            entry = copy.deepcopy(source)
            kern = [int(entry["kh"]), int(entry["kw"])]
            stride = [int(v) for v in entry["stride"]]
            dilation = [int(v) for v in entry["dilation"]]
            padding = [int(v) for v in entry["padding"]]
            h, w = _image_for_window(kern, stride, dilation, padding[:2], padding[2:], output_extent=side)
            entry.update(
                name=f"{SYNTH_PREFIX}_conv_operand_capacity_boundary",
                Himg=h,
                Wimg=w,
                generalization={"generalization_axis": "shape"},
                source_reference=(
                    "synthesized for the convolution operand-capacity boundary: "
                    f"the RTL-derived operand store holds {capacity_rows} rows, while "
                    f"{side * side} output windows need at least {side * side * rows_per_window} "
                    f"rows if materialized together; window geometry comes from {source['name']}"
                ),
            )
            why = cap_to_affordable(entry, spec_doc, extends=source["name"])
            if why:
                entry["source_reference"] += f". {why}"
            entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
            entries.append(entry)
            conv_capacity_boundary = {
                "source": source["name"],
                "capacity_rows": capacity_rows,
                "rows_per_window": rows_per_window,
                "output_windows": side * side,
                "image": [h, w],
            }

    # ---- the GEOMETRY axis --------------------------------------------------------------------------
    # ASPECT RATIO, which no cell can state. A `(family, dtype, alignment)` cell puts a 448:1
    # tall-skinny convolution and a square projection in the same box, and every capsule the other axes
    # emit is square by construction (`M = tile`, `N = tile`). Measured on a real vision capture: five
    # geometry classes carry its contraction work, the corpus covered ONE, and the heaviest block of it
    # -- `M=3584, K=14, N=8`, repeated 48 times -- has an aspect ratio nothing in the corpus reaches
    # within two orders of magnitude. A compiler that tiles that shape badly passes every existing
    # capsule.
    #
    # The extents are the REAL ones, so the capsule tests the geometry at the size the work occurs at.
    # That is affordable only because the tier follows the size: a class whose representative writes
    # more than the certification budget affords becomes a loop-tier capsule that names the certified
    # sibling it rests on, which is the same rule the memory-regime and accumulation-depth axes use.
    geom_block = spec_doc.get("shape_geometry") or {}
    unsized_geometry: list[str] = []
    # THE SAME DTYPE THE MEMORY-REGIME AXIS USES, and for the same reason: these capsules rest on a
    # certified `SY_contraction_<dtype>_aligned` sibling, so the dtype has to be one the CELL axis
    # actually emitted. Taking the head of the precision preference instead named a sibling that does
    # not exist on a target whose cells are float -- an L2-only capsule pointing at nothing, which is
    # precisely the orphan the `extends` field exists to prevent.
    geom_dtype = regime_dtype or (kept[0] if kept else narrowest_admitted(admitted_dtypes))
    geom_op = (
        op_for_family("contraction", admitted_ops=ops_gradeable_at(geom_dtype, pool), dtype=geom_dtype)
        if geom_dtype
        else None
    )
    _emitted = {str(e.get("name")) for e in entries}
    for _req in geom_block.get("required") or ():
        klass = str(_req.get("class") or "")
        # A SKIP THAT SAYS NOTHING READS AS COVERAGE. Both of these are refusals to emit, and both
        # used to be a bare `continue` -- so the class left the requirement without leaving a trace,
        # and `geometry_classes_unsynthesizable` (whose whole contract is "a class silently absent
        # here reads downstream as a geometry the corpus covers") did not name it. Measured on this
        # repo's own gemmini census: the residual `unknown` class carries 36 regions and 4.6% of all
        # contraction MAC work -- more than two of the classes that DID get a capsule -- and nothing
        # anywhere recorded that it had been dropped.
        if not klass:
            unsized_geometry.append(
                "<unnamed> (a requirement row carries no class, so there is no aspect ratio to "
                "synthesize and no name to report it under)"
            )
            continue
        if klass == "unknown":
            # Deliberately not emitted, and that is a judgement rather than a wall: `unknown` is the
            # taxonomy's RESIDUAL label, not an aspect ratio, so the regions behind it share no ratio
            # and one capsule cannot represent them. Recorded because the mass is real.
            unsized_geometry.append(
                f"{klass} (the taxonomy's residual label, not an aspect ratio: {_req.get('n_regions')} "
                f"region(s) carrying {_req.get('mac_fraction')} of all contraction MAC work match no "
                f"named class, so they share no geometry one capsule could represent -- a member named "
                f"after the fall-through would claim a coverage the classifier never asserted)"
            )
            continue
        if not geom_op or not geom_dtype:
            unsized_geometry.append(f"{klass} (no contraction op gradeable at any admitted dtype)")
            continue
        m, k, n = _req.get("M"), _req.get("K"), _req.get("N")
        if not (isinstance(m, int) and isinstance(k, int) and isinstance(n, int)):
            unsized_geometry.append(
                f"{klass} ({_req.get('unreachable') or 'the requirement carries no resolved M/K/N'})"
            )
            continue
        _sib = f"{SYNTH_PREFIX}_contraction_{geom_dtype}_aligned"
        if _sib not in _emitted:
            # An L2-only capsule that names a sibling nothing emitted is an orphan: it rests on a
            # functional guarantee that does not exist. Reported, never emitted.
            unsized_geometry.append(
                f"{klass} (no certified sibling {_sib!r} was emitted for this dtype, so a loop-tier "
                f"capsule of this geometry would rest on nothing)"
            )
            continue
        # The dtype's own datapath granularity applies to a REAL shape exactly as it does to a
        # tile-relative one: a block-scaled format cannot execute a K that is not a whole group.
        q = _quanta.get(geom_dtype) or {}
        row_q, red_q = int(q.get("row") or 1), int(q.get("reduction") or 1)
        up = lambda v, g: -(-int(v) // int(g)) * int(g)  # noqa: E731 -- local rounding helper
        m, n, k = up(m, row_q), up(n, row_q), up(k, red_q)
        entry = {
            "cat": "isa",
            "kind": "isa",
            "name": f"{SYNTH_PREFIX}_geometry_{klass}".replace("-", "_"),
            "op": geom_op,
            "operand_dtype": geom_dtype,
            "out": "Y0",
            "lhs": "A0",
            "weight": "W",
            "source_role": SOURCE_ROLE,
            "source_reference": (
                f"synthesized for geometry class {klass!r}: real captures present "
                f"{_req.get('n_regions')} contraction region(s) of this aspect ratio carrying "
                f"{_req.get('mac_fraction')} of all contraction MAC work, and the heaviest of them is "
                f"M={m} K={k} N={n}. Every other synthesized capsule is square, so without this the "
                f"corpus cannot tell a compiler that tiles this ratio well from one that does not"
            ),
            "label": "public",
            "generalization": {"generalization_axis": "shape_geometry"},
            "M": m,
            "K": k,
            "N": n,
        }
        _why = cap_to_affordable(entry, spec_doc, extends=_sib)
        if _why:
            entry["source_reference"] += f". {_why}"
        if _writer_for(entry) is None:
            unsized_geometry.append(f"{klass} (op {geom_op!r} has no writer at dtype {geom_dtype!r})")
            continue
        if _writer_for(entry) == "pytorch":
            entry["source"] = "pytorch"
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        entries.append(entry)

    entries.extend(
        _tail_isolation_entries(geom_op, geom_dtype, spec_doc, emitted={str(e.get("name")) for e in entries})
    )
    entries.extend(_activation_contraction_entries(geom_dtype, spec_doc, emitted={str(e.get("name")) for e in entries}))

    return conv_capacity_boundary, unsized_geometry
