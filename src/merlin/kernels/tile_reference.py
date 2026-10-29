"""Recover complete contractions from target-projected tile data movement.

The selected target observer supplies tile semantics and storage coordinates.
This algebra checks operand identities, exact ordered reductions, output
coverage, and packed-view correspondence without expanding scalar products.
"""

from __future__ import annotations

from collections import Counter

from merlin.kernels.kernel_evidence import EvidenceFact, SemanticRegion


def recover_tile_region(events, declarations, *, evidence_id):
    tiles, accumulators, output_tiles = {}, {}, {}
    lhs_bytes, lhs_unique, loaded = Counter(), {}, Counter()
    stores, groups = [], {}
    weight, destination = None, None
    compute_count = 0
    widths = dict(i8=1, i16=2, i32=4, f32=4)

    def view(tensor, offset, stride, rows, cols):
        declaration = declarations[tensor]
        width = widths[declaration["dtype"]]
        shape = declaration["shape"]
        if offset % width or stride % width or len(shape) not in (2, 3):
            raise ValueError("tile origin requires an exact byte-addressable matrix or panel view")
        if len(shape) == 2:
            leading = shape[1]
            row, col = divmod(offset // width, leading)
            panel = None
        else:
            panels, depth, leading = shape
            panel, remainder = divmod(offset // width, depth * leading)
            row, col = divmod(remainder, leading)
            if panel >= panels:
                raise ValueError("packed tile origin escapes its declared panels")
        if stride != leading * width or row + rows > shape[-2] or col + cols > leading:
            raise ValueError("tile transfer escapes a source view or uses unproved row strides")
        return dict(tensor=tensor, row=row, col=col, rows=rows, cols=cols, panel=panel)

    for event_index, event in enumerate(events):
        kind = event["kind"]
        if kind == "load":
            tensor = event["tensor"]
            width = widths[declarations[tensor]["dtype"]]
            block = event["tile_cols"]
            if event["cols"] % block:
                raise ValueError("tile replay cannot silently drop a ragged load")
            for column in range(0, event["cols"], block):
                tile = view(tensor, event["offset"] + column * width, event["stride"], event["rows"], block)
                tiles[event["address"] + (column // block) * event["block_stride"]] = tile
            loaded[tensor] += event["rows"] * event["cols"] * width
        elif kind == "bind":
            if event["weight"] is not None:
                if event["weight"] not in tiles:
                    raise ValueError("preload reads a tile whose source was not captured")
                weight = dict(tiles[event["weight"]])
            destination = event
        elif kind == "compute":
            if destination is None or weight is None or event["lhs"] not in tiles:
                raise ValueError("compute requires observed LHS/RHS origins and destination")
            lhs = tiles[event["lhs"]]
            if (lhs["cols"] != weight["rows"] or lhs["rows"] != event["rows"]
                    or lhs["cols"] != event["cols"] or lhs["panel"] is not None):
                raise ValueError("compute dimensions do not establish an ordinary matrix product")
            key = destination["address"]
            if not destination["accumulate"]:
                accumulators[key] = []
            if key not in accumulators:
                raise ValueError("accumulation has no observed initializer")
            accumulators[key].append((dict(lhs), dict(weight), event_index))
            lhs_bytes[lhs["tensor"]] += lhs["rows"] * lhs["cols"] * widths[declarations[lhs["tensor"]]["dtype"]]
            lhs_unique.setdefault(lhs["tensor"], set()).add((lhs["row"], lhs["col"], lhs["rows"], lhs["cols"]))
            compute_count += 1
        elif kind == "store":
            tensor = event["tensor"]
            width = widths[declarations[tensor]["dtype"]]
            block = event["tile_cols"]
            if event["cols"] % block:
                raise ValueError("tile replay cannot silently drop a ragged readout")
            for column in range(0, event["cols"], block):
                output = view(tensor, event["offset"] + column * width, event["stride"], event["rows"], block)
                key = event["address"] + (column // block) * event["block_stride"]
                contributions = accumulators.get(key)
                if not contributions:
                    raise ValueError("readout has no complete observed reduction")
                cursor, lhs_name, rhs_name, rhs_offset = 0, None, None, None
                for lhs, rhs, index in contributions:
                    shape = declarations[rhs["tensor"]]["shape"]
                    rhs_col = rhs["col"] + (0 if rhs["panel"] is None else rhs["panel"] * shape[-1])
                    if (lhs["row"] != output["row"] or lhs["col"] != cursor or rhs["row"] != cursor
                            or rhs["cols"] != output["cols"] or lhs["rows"] != output["rows"]):
                        raise ValueError("captured reduction does not match the output's ordered scalar product")
                    offset = rhs_col - output["col"]
                    if lhs_name is None:
                        lhs_name, rhs_name, rhs_offset = lhs["tensor"], rhs["tensor"], offset
                    if (lhs_name, rhs_name, rhs_offset) != (lhs["tensor"], rhs["tensor"], offset):
                        raise ValueError("one output mixes incompatible projection operands")
                    cursor += lhs["cols"]
                lhs_shape = declarations[lhs_name]["shape"]
                if cursor != lhs_shape[1]:
                    raise ValueError("readout omits part of the complete source reduction")
                numeric = dict(event["numeric"])
                group = dict(lhs=lhs_name, rhs=rhs_name, rhs_column_offset=rhs_offset, m=lhs_shape[0], k=cursor,
                             n=declarations[tensor]["shape"][1], numeric=numeric)
                if tensor in groups and groups[tensor] != group:
                    raise ValueError("output tiles disagree on mathematics or numerical boundaries")
                groups[tensor] = group
                tile_key = tensor, output["row"], output["col"]
                if tile_key in output_tiles:
                    raise ValueError("output tile is overwritten or multiply observed")
                output_tiles[tile_key] = output
                stores.append(dict(tensor=tensor, row=output["row"], col=output["col"],
                                   reduction_segments=len(contributions), k=cursor, first_compute=contributions[0][2],
                                   last_compute=contributions[-1][2], readout=event_index))
        else:
            raise ValueError("tile semantic replay encountered an unsupported selected effect")
    if not groups:
        raise ValueError("captured region has no observed output")
    declared_arrays = {name for name, value in declarations.items() if value["shape"]}
    if set(loaded) | set(groups) != declared_arrays:
        raise ValueError("captured region leaves a declared tensor's role or output behavior unresolved")
    if set(loaded) & set(groups):
        raise ValueError("source read/write intermediates require a versioned connected-region dataflow observer")
    for tensor, group in groups.items():
        rhs_shape = declarations[group["rhs"]]["shape"]
        rhs_columns = rhs_shape[-1] * (rhs_shape[0] if len(rhs_shape) == 3 else 1)
        if group["rhs_column_offset"] < 0 or group["rhs_column_offset"] + group["n"] > rhs_columns:
            raise ValueError("projection RHS view escapes its complete physical backing")
        coverage = sum(tile["rows"] * tile["cols"] for key, tile in output_tiles.items() if key[0] == tensor)
        if coverage != group["m"] * group["n"]:
            raise ValueError("captured region does not cover all source output elements")
        tiles_for_output = [tile for key, tile in output_tiles.items() if key[0] == tensor]
        if any(max(a["row"], b["row"]) < min(a["row"] + a["rows"], b["row"] + b["rows"])
               and max(a["col"], b["col"]) < min(a["col"] + a["cols"], b["col"] + b["cols"])
               for index, a in enumerate(tiles_for_output) for b in tiles_for_output[index + 1:]):
            raise ValueError("output tile coverage overlaps")
    tensors, inputs, operations, outputs, problems = [], [], [], [], []
    for index, (output, group) in enumerate(groups.items()):
        lhs, rhs = group["lhs"], f"rhs_view:{index}"
        if lhs not in inputs:
            inputs.append(lhs)
            tensors.append(dict(key=lhs, shape=[group["m"], group["k"]],
                                dtype=declarations[lhs]["dtype"], layout="logical"))
        inputs.append(rhs)
        tensors.append(dict(key=rhs, shape=[group["k"], group["n"]],
                            dtype=declarations[group["rhs"]]["dtype"], layout="logical"))
        tensors.append(dict(key=output, shape=[group["m"], group["n"]],
                            dtype=declarations[output]["dtype"], layout="logical"))
        attrs = {key: group[key] for key in ("m", "k", "n")}
        attrs.update(reduction_order="ordered_k", initialization="zero")
        operations.append(dict(key=f"matmul:{index}", op="matmul", inputs=[lhs, rhs], outputs=[output],
                               attributes=attrs, numeric=group["numeric"], numeric_complete=True))
        outputs.append(output)
        physical_shape = declarations[group["rhs"]]["shape"]
        rhs_view = dict(kind="column_panels" if len(physical_shape) == 3 else "identity",
                        physical_shape=physical_shape, logical_shape=[group["k"], group["n"]],
                        panel=physical_shape[-1], column_offset=group["rhs_column_offset"],
                        source_tensor=group["rhs"], source_evidence=[f"{evidence_id}#tile_dataflow"],
                        equation="physical[(n+column_offset)//panel,k,(n+column_offset)%panel]=logical[k,n]")
        problems.append(dict(**attrs, dtypes=[declarations[lhs]["dtype"], declarations[group["rhs"]]["dtype"],
                                             declarations[output]["dtype"]],
                             numeric=group["numeric"], rhs_view=rhs_view))
    region = SemanticRegion.from_record(dict(region_id=evidence_id, tensors=tensors, operations=operations,
                                            inputs=inputs, outputs=outputs,
                                            external_uses={output: 1 for output in outputs}, complete=True))
    lhs_total = {name: loaded[name] for name in lhs_bytes}
    lhs_extent = {name: declarations[name]["shape"][0] * declarations[name]["shape"][1]
                       * widths[declarations[name]["dtype"]] for name in lhs_bytes}
    retained = all(lhs_total[name] == lhs_extent[name] for name in lhs_bytes)
    evidence = (f"{evidence_id}#tile_dataflow",)
    facts = [EvidenceFact("memory.panel_reuse", any(lhs_bytes[name] > loaded[name] for name in lhs_bytes),
                          "inferred", evidence, "exact captured tile source identities and compute uses")]
    if retained:
        facts.append(EvidenceFact("memory.onchip_resident", "a", "inferred", evidence,
                                  "complete invocation LHS loaded exactly once and reused across all output tiles"))
    return dict(semantic_region=region, problem=problems[0] if len(problems) == 1 else None,
                problems=problems, rhs_view=problems[0]["rhs_view"] if len(problems) == 1 else None,
                facts=facts, tile_dataflow=dict(compute_tiles=compute_count, output_tiles=stores,
                    loaded_bytes=dict(loaded), lhs_requested_compute_bytes=dict(lhs_bytes),
                    lhs_loaded_bytes=lhs_total, lhs_logical_bytes=lhs_extent,
                    lhs_retained_complete_invocation=retained, reduction_order="observed_complete_ordered_k",
                    completion="not_established", timing="not_established"))
