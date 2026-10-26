"""Ordered, source-bound composition observations over contract-decoded operands.

No instruction operand is unpacked here. Decoders own field layouts and flag
meanings; selected RTL facts own dataflow codes, load states and tile geometry.
"""

import copy
import math
from collections import Counter

from .frozen_placement import require_same_region

MODES = frozenset(
    {
        "readout_multipliers",
        "accumulator_preload",
        "pooled_store",
        "scratchpad_source_pooling_fenced",
        "scaled_accumulate_load",
        "output_stationary",
        "i32_exit",
        "k_extension",
        "shared_input",
        "column_block",
    }
)
_LOADS = {"MVIN", "MVIN2", "MVIN3"}
_COMPUTES = {"COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE"}


def _known(decoded, field):
    value = decoded.get(field)
    if value is None or value == "UNKNOWN":
        raise ValueError(f"UNKNOWN: unresolved decoded field {field}")
    return value


def _integers(parameters, key):
    values = parameters.get(key)
    if not isinstance(values, list) or not values or any(type(value) is not int for value in values):
        raise ValueError(f"UNKNOWN: {key} needs declared integer values")
    return values


def _identity(value):
    from .trace_check import _operand_identity

    identity = _operand_identity(value)
    if identity is None:
        raise ValueError("UNKNOWN: unresolved operand identity")
    return identity


def _buffer_identity(value):
    identity = _identity(value)
    return identity[:2] if identity[0] in {"argbase", "global"} else identity


def _dataflow_code(facts, target, mode):
    from merlin.targetgen.rtl.firrtl_dataflow import select_code
    from merlin.targetgen.target_registry import load_contract

    # A complete structural record needs no source fallback or target lookup.
    dataflow = facts.get("facts", facts).get("timing_structure", {}).get("dataflow")
    return select_code(
        facts, {} if dataflow is not None and dataflow.get("status") == "derived" else load_contract(target), mode
    )


def _selected_hardware(target, needs):
    """Only load facts needed by the requested modes; absent is never a default."""
    hardware = {}
    if "output_stationary" in needs:
        from merlin.targetgen.rtl.facts import load_facts

        facts = load_facts(target)
        hardware["output_stationary"] = _dataflow_code(facts, target, "output_stationary")
    if needs & {"scaled_accumulate_load", "k_extension", "contraction_lineage", "accumulator_preload", "runtime_dma"}:
        from merlin.targetgen.rocc.decode import _semantics

        semantics = _semantics(target)
        observer = getattr(semantics, "operand_observer", None) or semantics
        if needs & {"scaled_accumulate_load", "runtime_dma"}:
            mapping = getattr(observer, "load_states", None)
            if mapping is None:
                from merlin.targetgen.composition_facts import load_state_mapping
                from merlin.targetgen.rtl.facts import load_facts
                from merlin.targetgen.target_registry import load_contract

                mapping = load_state_mapping(
                    facts=load_facts(target), contract=load_contract(target), isa=observer.isa_constants(target)
                )
            hardware["load_states"] = dict(mapping.ports)
        if "runtime_dma" in needs:
            from merlin.common.quant_formats import storage_bits
            from merlin.targetgen.rtl.facts import load_facts

            facts = load_facts(target)
            body = facts.get("facts", facts)
            dtypes = [row.get("dtype") for row in body.get("datapaths", []) if row.get("name") == "input"]
            if len(dtypes) != 1 or not dtypes[0]:
                raise ValueError("UNKNOWN: DMA input element width is underived")
            bits = storage_bits(dtypes[0])
            if bits <= 0 or bits % 8:
                raise ValueError("UNKNOWN: DMA input element width is not byte-addressable")
            hardware["input_bytes"] = bits // 8
            hardware["max_load_bytes"] = getattr(mapping, "max_element_bytes", None)
        if needs & {"k_extension", "contraction_lineage", "accumulator_preload"}:
            isa = observer.isa_constants(target)
        if "k_extension" in needs:
            dimension = isa.get("DIM")
            if type(dimension) is not int or dimension <= 0:
                raise ValueError("UNKNOWN: contraction tile geometry is underived")
            hardware["tile_k"] = dimension
        if needs & {"k_extension", "contraction_lineage", "accumulator_preload"}:
            masks = [
                step.get("mask")
                for program in isa.get("OPERAND_FIELDS", {}).values()
                for step in program.get("fields", [])
                if step.get("as") in {"flag", "flag_choice"}
            ]
            if not masks or any(type(mask) is not int or mask <= 0 for mask in masks):
                raise ValueError("UNKNOWN: accumulator address flags are underived")
            hardware["address_flags"] = 0
            for mask in masks:
                hardware["address_flags"] |= mask
    return hardware


def _events(instructions, hardware):
    """Snapshot effective configuration at each use, rather than count setup."""
    store_config, execution_config, load_configs = {}, {}, {}
    config_index = None
    commits, computes, loads = [], [], []
    pending = []
    preload = None
    for index, row in enumerate(instructions):
        kind, decoded = row.get("class"), row.get("decoded") or {}
        if kind == "UNKNOWN":
            raise ValueError("UNKNOWN: undecoded instruction in composition region")
        if kind == "CONFIG_ST":
            store_config, config_index = decoded, index
        elif kind == "CONFIG_EX":
            execution_config = decoded
        elif kind == "CONFIG_LD" and "load_states" in hardware:
            load_configs[_known(decoded, "state_id")] = decoded
        elif kind in _LOADS:
            state = hardware.get("load_states", {}).get(kind)
            loads.append((decoded, load_configs.get(state, {})))
        elif kind == "PRELOAD":
            preload = decoded
        elif kind in _COMPUTES:
            computes.append((decoded, execution_config))
            pending.append((decoded, preload))
            preload = None
        elif kind == "MVOUT":
            commits.append((decoded, store_config, config_index, pending))
            pending = []
    return commits, computes, loads


def _obligation_values(check, instructions, *, target):
    """Observe effective state at use; unused setup cannot discharge a duty."""
    needs = (
        {"scaled_accumulate_load"}
        if (check in {"unit_mvin_scale", "nonnegative_stride"} and any(r.get("class") in _LOADS for r in instructions))
        else set()
    )
    hardware = _selected_hardware(target, needs)
    commits, computes, loads = _events(instructions, hardware)
    transfers = [r["decoded"] for r in instructions if r.get("class") in _LOADS | {"MVOUT"}]
    if check == "unit_mvin_scale":
        return [_known(config, "scale") for _, config in loads]
    if check == "weight_stationary":
        from merlin.targetgen.rtl.facts import load_facts

        facts = load_facts(target)
        code = _dataflow_code(facts, target, "weight_stationary")
        return ["WS" if _known(config, "dataflow") == code else "OS" for _, config in computes]
    if check == "nonzero_dram":
        values = [_known(row, "dram") for row in transfers]
        if any(not isinstance(v, dict) or v.get("kind") != "const" or type(v.get("raw")) is not int for v in values):
            raise ValueError("UNKNOWN: DRAM address is not observed in retirement")
        return [v["raw"] for v in values]
    if check == "transfer_rows":
        return [_known(row, "rows") for row in transfers]
    if check == "movement_bytes":
        # F1 transfers use byte storage; columns are bytes per transferred row.
        return [_known(row, "cols") for row in transfers]
    if check == "nonnegative_stride":
        return [_known(config, "stride") for _, config in loads] + [
            _known(config, "out_stride_bytes") for _, config, _, _ in commits
        ]
    if check == "max_columns":
        return [_known(store, "cols") for store, _, _, _ in commits]
    if check == "full_i32":
        return [
            "full_i32" if _known(store, "readout") == "i32" else _known(store, "readout") for store, _, _, _ in commits
        ]
    if check == "primitive_k_bound":
        return [_known(compute, "a_cols") * 128 * 128 for compute, _ in computes]
    if check in {"mvout_pool_stride", "mvout_pool_size"}:
        return [_known(config, check.removeprefix("mvout_")) for _, config, _, _ in commits]
    if check in {"mvout_scale", "mvout_relu"}:
        field, identity = ("acc_scale", 1.0) if check == "mvout_scale" else ("acc_act", 0)
        return [_known(config, field) != identity for _, config, _, _ in commits]
    if check == "pool_off":
        return [_known(config, "pool_stride") == 0 and _known(config, "pool_size") == 0 for _, config, _, _ in commits]
    if check == "raw_i32_exit":
        return [
            _known(store, "readout") == "i8"
            or (
                _known(store, "readout") == "i32"
                and (
                    _known(config, "pool_stride") == 0
                    and _known(config, "pool_size") == 0
                    and _known(config, "acc_scale") == 1.0
                    and _known(config, "acc_act") == 0
                    and 1 <= _known(store, "cols") <= 16
                )
            )
            for store, config, _, _ in commits
        ]
    if check == "pool_window":
        # _pool already binds the complete window to the declared mode. These
        # are the row's hardware limits, independently of that mode's values.
        return [
            _known(store, "readout") == "i8"
            and 1 <= _known(store, "cols") <= 16
            and all(1 <= _known(config, field) <= 3 for field in ("pool_size", "pool_stride"))
            and all(
                1 <= _known(config, field) <= 255 for field in ("pool_out_dim", "porows", "pocols", "orows", "ocols")
            )
            for store, config, _, _ in commits
        ]
    raise ValueError(f"UNKNOWN: no retired observation for {check}")


def check_emission_obligations(instructions, obligations, *, target):
    """Enforce declared command duties and retain every unobserved compiler duty."""
    from merlin.targetgen.admission_conditions import (
        obligation_condition,
        screen_conditions,
        validate_emission_obligations,
    )

    validate_emission_obligations(obligations)
    results = {}
    for name, duty in obligations.items():
        if duty["status"] == "compiler_obligation":
            results[name] = copy.deepcopy(duty)
            continue
        values = _obligation_values(duty["check"], instructions, target=target)
        # A transfer-only component may have no mesh work. It must still have
        # observed DMA; all other checks require their own observation.
        if (
            not values
            and duty["check"] in {"weight_stationary", "primitive_k_bound"}
            and any(r.get("class") in _LOADS for r in instructions)
        ):
            results[name] = {**duty, "observation_status": "not_applicable_no_mesh_work"}
            continue
        if not values:
            raise ValueError(f"UNKNOWN: emission obligation {name} has no retired observation")
        condition = obligation_condition(name, duty)
        if any(screen_conditions(condition, {name: value})[0] != "matched" for value in values):
            raise ValueError(f"emission obligation {name} failed {duty['check']}")
        results[name] = {**duty, "observation_status": "pass", "observed_uses": len(values)}
    return results


def _output_commits(commits, parameters):
    """Select explicit logical output bindings for independent sibling exits."""
    outputs = parameters.get("outputs")
    if outputs is None:
        return commits
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("UNKNOWN: output-scoped mode needs explicit operand identities")
    wanted = {_buffer_identity(value) for value in outputs}
    selected = [event for event in commits if _buffer_identity(_known(event[0], "dram")) in wanted]
    observed = {_buffer_identity(_known(event[0], "dram")) for event in selected}
    if observed != wanted:
        raise ValueError("UNKNOWN: output-scoped mode has no matching store for every output")
    return selected


def _accumulator_span(decoded, address_field, rows_field, cols_field, hardware):
    if _known(decoded, "is_acc_addr") is not True:
        raise ValueError("contraction/readout address does not select the accumulator")
    address = _known(decoded, address_field)
    rows, cols = _known(decoded, rows_field), _known(decoded, cols_field)
    if any(type(value) is not int or value < 0 for value in (address, rows, cols)) or min(rows, cols) <= 0:
        raise ValueError("UNKNOWN: contraction/readout accumulator extent is unresolved")
    start = address & ~hardware["address_flags"]
    return start, start + rows, cols


def _without_rows(resident, start, end):
    """Subtract overwritten rows while retaining both unaffected fragments."""
    remaining = []
    for left, right, width in resident:
        if right <= start or end <= left:
            remaining.append((left, right, width))
            continue
        if left < start:
            remaining.append((left, start, width))
        if end < right:
            remaining.append((end, right, width))
    return remaining


def _contraction_lineage(instructions, hardware):
    """Require every readout element to belong to still-resident contraction work."""
    resident, preload = [], None
    for row in instructions:
        kind, decoded = row.get("class"), row.get("decoded") or {}
        if kind == "PRELOAD":
            preload = decoded
        elif kind in _COMPUTES:
            resident.append(_accumulator_span(preload or {}, "c_addr", "c_rows", "c_cols", hardware))
        elif kind in _LOADS and decoded.get("is_acc_addr") is True:
            start, end, _ = _accumulator_span(decoded, "spad_addr", "rows", "cols", hardware)
            resident = _without_rows(resident, start, end)
        elif kind == "MVOUT":
            start, end, cols = _accumulator_span(decoded, "acc_addr", "rows", "cols", hardware)
            covered = start
            for left, right, width in sorted(resident):
                if width >= cols and left <= covered < right:
                    covered = right
            if covered < end:
                raise ValueError("readout accumulator rows/columns are not written by the contraction")


def _covered(span, resident):
    start, end, cols = span
    for left, right, width in sorted(resident):
        if width >= cols and left <= start < right:
            start = max(start, right)
    return start >= end


def _accumulator_seed(instructions, parameters, hardware):
    """Follow the declared seed through accumulator writes, computes and exits."""
    source = _buffer_identity(parameters.get("input"))
    seeds, results, preload = [], [], None
    computes = 0
    for row in instructions:
        kind, decoded = row.get("class"), row.get("decoded") or {}
        if kind in _LOADS and _known(decoded, "is_acc_addr") is True:
            span = _accumulator_span(decoded, "spad_addr", "rows", "cols", hardware)
            start, end, _ = span
            seeds = _without_rows(seeds, start, end)
            results = _without_rows(results, start, end)
            if _buffer_identity(_known(decoded, "dram")) == source and _known(decoded, "accumulate") is False:
                seeds.append(span)
        elif kind == "PRELOAD":
            preload = decoded
        elif kind in _COMPUTES:
            span = _accumulator_span(preload or {}, "c_addr", "c_rows", "c_cols", hardware)
            if _known(preload, "accumulate") is not True or not _covered(span, seeds):
                raise ValueError("contraction does not consume its source-bound accumulator seed")
            results.append(span)
            computes += 1
        elif kind == "MVOUT":
            span = _accumulator_span(decoded, "acc_addr", "rows", "cols", hardware)
            if not _covered(span, results):
                raise ValueError("readout does not consume the seeded contraction rows")
    if not computes:
        raise ValueError("UNKNOWN: accumulator seed has no consuming contraction")


def _element_count(shape):
    if not isinstance(shape, list) or not shape or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError("UNKNOWN: source tensor shape is unresolved")
    return math.prod(shape)


def _store_cells(store, config):
    identity = _identity(_known(store, "dram"))
    if identity[0] not in {"argbase", "global"} or type(identity[-1]) is not int:
        raise ValueError("UNKNOWN: readout has no source-relative output address")
    offset, rows, cols = identity[-1], _known(store, "rows"), _known(store, "cols")
    stride = _known(config, "out_stride_bytes")
    if any(type(v) is not int or v < 0 for v in (offset, rows, cols, stride)) or min(rows, cols) <= 0:
        raise ValueError("UNKNOWN: readout extent/stride is unresolved")
    if _known(config, "pool_stride") == 0:
        return [offset + r * stride + c for r in range(rows) for c in range(cols)]
    height, width, pitch = (_known(config, key) for key in ("porows", "pocols", "pool_out_dim"))
    if rows != 1 or any(type(v) is not int or v <= 0 for v in (height, width, pitch)) or width > pitch:
        raise ValueError("UNKNOWN: pooled readout extent is unresolved")
    return [offset + (r * pitch + x) * stride + c for r in range(height) for x in range(width) for c in range(cols)]


def _partitioned_readout(commits, partitions, *, relu, scaled, scratchpad, rows=None, scale_loads=None):
    if not isinstance(partitions, list) or not partitions:
        raise ValueError("UNKNOWN: readout needs source output partitions")
    wanted = [_buffer_identity(part["output"]) for part in partitions]
    if len(set(wanted)) != len(wanted) or {_buffer_identity(_known(s, "dram")) for s, _, _, _ in commits} != set(
        wanted
    ):
        raise ValueError("readout stores differ from the declared output allocations")
    for part in partitions:
        scoped = _output_commits(commits, dict(outputs=[part["output"]]))
        _readout(scoped, part, relu=relu, scaled=scaled, scratchpad=scratchpad, rows=rows, scale_loads=scale_loads)


_IDENTITY_BITS = 0x3F800000


def _runtime_scale_input(parameters):
    """Authenticate a runtime partition record before any store is read."""
    if "column_scale_bits" in parameters or "acc_scale_bits" in parameters:
        raise ValueError("runtime scale_input and column_scale_bits are mutually exclusive")
    scale, bits = parameters.get("scale_input"), parameters.get("expected_input_bits")
    evidence = parameters.get("scale_load_evidence_id")
    if (
        not isinstance(scale, dict)
        or set(scale) != {"kind", "arg_index", "offset", "dtype", "shape"}
        or scale["kind"] != "argbase"
        or type(scale["arg_index"]) is not int
        or scale["arg_index"] < 0
        or scale["offset"] != 0
        or scale["dtype"] != "f32"
        or scale["shape"] != [1]
    ):
        raise ValueError("UNKNOWN: runtime readout scale_input is not an f32[1] argument at offset 0")
    if type(bits) is not int or not 0 <= bits < 1 << 32 or bits >> 23 & 0xFF == 0xFF:
        raise ValueError("UNKNOWN: runtime readout has no frozen finite f32 input bits")
    if not isinstance(evidence, str) or not evidence:
        raise ValueError("UNKNOWN: runtime readout has no scale_load_evidence_id")
    return scale["arg_index"], bits, evidence


def _runtime_readout(commits, parameters, *, relu, rows, scale_loads):
    """Every dependent store observes the frozen input bits through a retired load chain.

    The partition names the executed input bits, not a compile-time multiplier:
    0 and 1 are as valid as any other finite scale. Each effective CONFIG_ST
    must be one of the authenticated report's load-to-rs2 chains for exactly
    this scale argument; equal bits from another source do not qualify.
    """
    from .runtime_scale_trace import authenticate_report

    arg, bits, evidence = _runtime_scale_input(parameters)
    if scale_loads is None or rows is None:
        raise ValueError("UNKNOWN: runtime readout has no retired scale-load evidence")
    report = authenticate_report(scale_loads)
    keys = [key for key, value in report["evidence"].items() if value == evidence]
    if len(keys) != 1 or keys[0].rpartition(":")[2] != str(arg):
        raise ValueError("runtime scale_load_evidence_id is not this scale input's observed chain")
    kernel = keys[0].rpartition(":")[0]
    uses = {u["command_ordinal"]: u for u in report["config_uses"] if u["kernel"] == kernel and u["arg_index"] == arg}
    commits = _output_commits(commits, parameters)
    if not commits:
        raise ValueError("UNKNOWN: runtime readout has no dependent store")
    count, covered = _element_count(parameters.get("shape")), set()
    for store, config, index, _computes in commits:
        if _known(store, "readout") != "i8" or _known(store, "is_acc_addr") is not True:
            raise ValueError("scaled/activated readout requires a narrow i8 store")
        if index is None:
            raise ValueError("runtime readout store has no effective CONFIG_ST")
        if _known(config, "acc_scale_bits") != bits:
            raise ValueError("effective retired CONFIG_ST scale bits differ from the frozen runtime input")
        row = rows[index]
        use = uses.get(row.get("command_ordinal"))
        if (
            use is None
            or row.get("kernel") != kernel
            or use["pc"] != row.get("pc")
            or use["rs2"] != (row.get("rs2") or {}).get("raw")
            or use["scale_bits"] != bits
        ):
            raise ValueError("effective CONFIG_ST rs2 has no retired load provenance from its scale input")
        if relu and _known(config, "relu") is not True:
            raise ValueError("readout lacks simultaneous ReLU activation")
        if not relu and _known(config, "acc_act") != 0:
            raise ValueError("runtime readout applies an undeclared activation")
        cells = _store_cells(store, config)
        if any(c < 0 or c >= count for c in cells):
            raise ValueError("readout store exceeds its source output")
        if len(set(cells)) != len(cells) or covered.intersection(cells):
            raise ValueError("readout stores overlap source output elements")
        covered.update(cells)
    if len(covered) != count:
        raise ValueError("readout stores do not cover every source output element")


def _readout(
    commits, parameters, *, relu=False, scaled=False, per_block=False, scratchpad=False, rows=None, scale_loads=None
):
    if "partitions" in parameters:
        if per_block:
            raise ValueError("UNKNOWN: column_block requires explicit per-block CONFIG_ST re-issue evidence")
        return _partitioned_readout(
            commits,
            parameters["partitions"],
            relu=relu,
            scaled=scaled,
            scratchpad=scratchpad,
            rows=rows,
            scale_loads=scale_loads,
        )
    if "scale_input" in parameters:
        if per_block or scratchpad:
            raise ValueError("UNKNOWN: runtime scale readout supports only tensor-wide accumulator stores")
        return _runtime_readout(commits, parameters, relu=relu, rows=rows, scale_loads=scale_loads)
    commits = _output_commits(commits, parameters)
    columns = parameters.get("column_scale_bits")
    count = _element_count(parameters["shape"]) if "shape" in parameters else None
    covered = set()
    expected = [] if columns is not None else _integers(parameters, "acc_scale_bits")
    stores_per_block = _integers(parameters, "stores_per_block") if per_block else None
    if stores_per_block is not None:
        if len(stores_per_block) != len(expected) or any(count <= 0 for count in stores_per_block):
            raise ValueError("UNKNOWN: column_block needs source-derived store counts for each block")
        expected = [bits for bits, count in zip(expected, stores_per_block) for _ in range(count)]
    observed, configs = [], []
    for store, config, index, _computes in commits:
        if _known(store, "readout") != "i8" or _known(store, "is_acc_addr") is not (not scratchpad):
            raise ValueError("scaled/activated readout requires a narrow i8 store")
        if columns is not None:
            if not isinstance(columns, list) or not columns or any(type(v) is not int for v in columns):
                raise ValueError("UNKNOWN: readout needs source-bound column multipliers")
            offset = _identity(_known(store, "dram"))[-1]
            width = _known(store, "cols")
            if type(offset) is not int or type(width) is not int or width <= 0:
                raise ValueError("UNKNOWN: pooled readout column extent is unresolved")
            first = offset % len(columns)
            gains = columns[first : first + width]
            if len(gains) != width or len(set(gains)) != 1 or _known(config, "acc_scale_bits") != gains[0]:
                raise ValueError("pooled readout multiplier differs from its source columns")
            expected.append(gains[0])
            if count is not None:
                if len(columns) != parameters["shape"][-1]:
                    raise ValueError("readout multiplier columns differ from source shape")
                cells = _store_cells(store, config)
                if any(c < 0 or c >= count or columns[c % len(columns)] != gains[0] for c in cells):
                    raise ValueError("readout store exceeds its source column partition")
                if len(set(cells)) != len(cells) or covered.intersection(cells):
                    raise ValueError("readout stores overlap source output elements")
                covered.update(cells)
        observed.append(_known(config, "acc_scale_bits"))
        configs.append(index)
        if relu and _known(config, "relu") is not True:
            raise ValueError("readout lacks simultaneous ReLU activation")
        if scaled:
            scale = _known(config, "acc_scale")
            if type(scale) not in (int, float) or not math.isfinite(scale) or scale == 1:
                raise ValueError("readout lacks a finite non-unit accumulator scale")
    if Counter(observed) != Counter(expected):
        raise ValueError("readout multiplier multiset differs from declared stores")
    if count is not None and len(covered) != count:
        raise ValueError("readout stores do not cover every source output element")
    if per_block:
        boundary = 0
        for count in stores_per_block:
            if boundary and configs[boundary] == configs[boundary - 1]:
                raise ValueError("column_block requires per-block CONFIG_ST re-issue")
            boundary += count
        if observed != expected:
            raise ValueError("column_block requires ordered per-block multiplier values")


def _pool(commits, parameters):
    if "geometry" in parameters:
        return _pool_geometry(commits, parameters["geometry"])
    fields = parameters.get("fields")
    required = {"pool_stride", "pool_size", "pool_out_dim", "porows", "pocols", "orows", "ocols", "upad", "lpad"}
    if not isinstance(fields, dict) or set(fields) != required or any(type(v) is not int for v in fields.values()):
        raise ValueError("UNKNOWN: pooled_store needs the complete declared pool window")
    if fields["pool_stride"] <= 0:
        raise ValueError("pooled_store requires an enabled pooling stride")
    for _store, config, _index, _computes in commits:
        if any(_known(config, key) != value for key, value in fields.items()):
            raise ValueError("active pooled-store window differs from declaration")


def _pool_geometry(commits, geometry):
    """Check arbitrary row bands against source windows and complete output coverage."""
    h, w, channels = geometry["image"]
    size, stride = geometry["window"], geometry["stride"]
    oh, ow = (h - size) // stride + 1, (w - size) // stride + 1
    if min(h, w, channels, size, stride, oh, ow) <= 0:
        raise ValueError("UNKNOWN: invalid source pool geometry")
    covered, buffer = set(), None
    for store, config, _index, _computes in commits:
        identity = _identity(_known(store, "dram"))
        if identity[0] not in {"argbase", "global"} or type(identity[-1]) is not int:
            raise ValueError("UNKNOWN: pooled output has no source-relative address")
        if buffer is None:
            buffer = identity[:-1]
        if identity[:-1] != buffer:
            raise ValueError("pooled stores target different output allocations")
        pixel, column = divmod(identity[-1], channels)
        row, x = divmod(pixel, ow)
        rows, width = _known(config, "porows"), _known(store, "cols")
        expected = dict(
            pool_stride=stride,
            pool_size=size,
            pool_out_dim=ow,
            pocols=ow,
            ocols=w,
            orows=(rows - 1) * stride + size,
            upad=0,
            lpad=0,
        )
        if (
            x
            or row < 0
            or rows <= 0
            or row + rows > oh
            or width <= 0
            or column + width > channels
            or _known(store, "rows") != 1
            or _known(config, "out_stride_bytes") != channels
            or any(_known(config, k) != v for k, v in expected.items())
        ):
            raise ValueError("pooled store band differs from source window geometry")
        cells = {(r, c, k) for r in range(row, row + rows) for c in range(ow) for k in range(column, column + width)}
        if cells & covered:
            raise ValueError("pooled stores overlap output elements")
        covered.update(cells)
    if len(covered) != oh * ow * channels:
        raise ValueError("pooled stores do not cover every source output element")


def _scratchpad_pool_fence(instructions):
    """Require a still-resident, completed scratchpad source at each pooled exit."""
    resident, completed = [], []
    for row in instructions:
        kind, decoded = row.get("class"), row.get("decoded") or {}
        if kind in _LOADS and _known(decoded, "is_acc_addr") is False:
            start, rows, cols = (_known(decoded, key) for key in ("spad_addr", "rows", "cols"))
            if any(type(value) is not int or value < 0 for value in (start, rows, cols)) or min(rows, cols) <= 0:
                raise ValueError("UNKNOWN: pooling source extent is unresolved")
            end = start + rows
            resident = _without_rows(resident, start, end)
            completed = _without_rows(completed, start, end)
            resident.append((start, end, cols))
        elif kind == "FENCE" and resident:
            # Opaque accelerator DMA needs both memory and device-I/O ordering.
            # The standard RISC-V default fence is iorw,iorw with fm=0.
            pred, succ, mode = (_known(decoded, key) for key in ("pred", "succ", "fm"))
            if any(type(value) is not int for value in (pred, succ, mode)) or mode != 0:
                raise ValueError("UNKNOWN: pooling completion fence semantics are unresolved")
            if pred == 15 and succ == 15:
                completed = list(resident)
        elif kind == "MVOUT":
            if _known(decoded, "is_acc_addr") is not False:
                raise ValueError("scratchpad pooling store does not select scratchpad rows")
            start, rows, cols = (_known(decoded, key) for key in ("acc_addr", "rows", "cols"))
            if any(type(value) is not int or value < 0 for value in (start, rows, cols)) or min(rows, cols) <= 0:
                raise ValueError("UNKNOWN: pooling store extent is unresolved")
            if not _covered((start, start + rows, cols), completed):
                raise ValueError("scratchpad pooling source lacks a fence after its resident load")


def _scaled_load(loads, parameters):
    if "operands" in parameters:
        return _bound_scaled_loads(loads, parameters["operands"])
    expected = _integers(parameters, "scale_bits")
    observed = []
    for load, config in loads:
        if _known(load, "is_acc_addr") is True and _known(load, "accumulate") is True:
            observed.append(_known(config, "scale_bits"))
    if Counter(observed) != Counter(expected):
        raise ValueError("scaled accumulator loads differ from declared operand multipliers")


def _bound_scaled_loads(loads, operands):
    if not isinstance(operands, list) or len(operands) != 2:
        raise ValueError("UNKNOWN: scaled operand sum needs both source operands")
    wanted = {_buffer_identity(part["input"]): part for part in operands}
    if len(wanted) != len(operands):
        raise ValueError("UNKNOWN: scaled operands have ambiguous source bindings")
    covered = {key: set() for key in wanted}
    for load, config in loads:
        if _known(load, "is_acc_addr") is not True:
            raise ValueError("scaled operand load does not target the accumulator")
        identity = _identity(_known(load, "dram"))
        key = _buffer_identity(_known(load, "dram"))
        if key not in wanted or identity[0] not in {"argbase", "global"}:
            raise ValueError("scaled load is not bound to a source operand")
        part = wanted[key]
        count = _element_count(part["shape"])
        if (
            _known(config, "scale_bits") != part["scale_bits"]
            or _known(config, "shrink") != 1
            or _known(load, "accumulate") is not part["accumulate"]
        ):
            raise ValueError("scaled load differs from its proved source operand configuration")
        offset, rows, cols, stride = identity[-1], _known(load, "rows"), _known(load, "cols"), _known(config, "stride")
        if any(type(v) is not int or v < 0 for v in (offset, rows, cols, stride)) or min(rows, cols) <= 0:
            raise ValueError("UNKNOWN: scaled operand load extent is unresolved")
        cells = [offset + r * stride + c for r in range(rows) for c in range(cols)]
        if any(c < 0 or c >= count for c in cells) or len(set(cells)) != len(cells) or covered[key].intersection(cells):
            raise ValueError("scaled operand loads overlap or exceed the source tensor")
        covered[key].update(cells)
    if any(len(covered[key]) != _element_count(part["shape"]) for key, part in wanted.items()):
        raise ValueError("scaled loads do not cover every source operand element")


def _k_extension(commits, parameters, hardware):
    source, extension = parameters.get("source_k"), parameters.get("extension_k")
    if type(source) is not int or source <= 0 or type(extension) is not int or extension <= 0:
        raise ValueError("UNKNOWN: k_extension needs source and coefficient-digit K extents")
    tile = hardware["tile_k"]
    original = (source + tile - 1) // tile
    extended = (source + extension + tile - 1) // tile
    if extended <= original:
        raise ValueError("k_extension lacks the declared extra K tiles per output")
    mask = hardware["address_flags"]
    pending = Counter()
    for store, config, _, work in commits:
        # Blocked schedules compute several accumulator outputs before draining
        # them. Preserve each destination's work across intervening stores.
        for _, preload in work:
            address = _known(preload or {}, "c_addr")
            if type(address) is not int:
                raise ValueError("UNKNOWN: K-extension compute destination is unresolved")
            pending[address & ~mask] += 1
        destination = _known(store, "acc_addr")
        if type(destination) is not int:
            raise ValueError("UNKNOWN: output accumulator address is unresolved")
        base = destination & ~mask
        destinations = [base]
        if config and config.get("pool_stride"):
            # A pooled MVOUT drains a whole accumulator image. Authenticate
            # every computed tile, including the final partial row tile.
            rows, cols = _known(config, "orows"), _known(config, "ocols")
            if any(type(value) is not int or value <= 0 for value in (rows, cols)):
                raise ValueError("UNKNOWN: pooled K-extension image extent is unresolved")
            destinations = [base + offset for offset in range(0, rows * cols, tile)]
        for address in destinations:
            if pending.pop(address, 0) != extended:
                raise ValueError("k_extension lacks the declared extra K tiles per output")
    if pending:
        raise ValueError("K-extension compute work targets a different accumulator output")


def _shared_input(instructions, parameters):
    expected = _buffer_identity(parameters.get("input"))
    live, loaded, outputs = {}, Counter(), {}
    used = set()
    for row in instructions:
        kind, decoded = row.get("class"), row.get("decoded") or {}
        if kind in _LOADS:
            destination = _known(decoded, "spad_addr")
            source = _buffer_identity(_known(decoded, "dram"))
            extent = (_known(decoded, "rows"), _known(decoded, "cols"))
            if type(destination) is not int or any(type(size) is not int or size <= 0 for size in extent):
                raise ValueError("UNKNOWN: shared-input load has unresolved extent/address")
            for address, (_, resident_extent) in list(live.items()):
                if address < destination + extent[0] and destination < address + resident_extent[0]:
                    del live[address]
            if source == expected:
                loaded[(destination, extent)] += 1
            live[destination] = (source, extent)
        elif kind in _COMPUTES:
            address = _known(decoded, "a_spad")
            if address not in live:
                raise ValueError("UNKNOWN: compute input has no resident load")
            source, extent = live[address]
            if source == expected:
                used.add((address, extent))
        elif kind == "MVOUT":
            output = _buffer_identity(_known(decoded, "dram"))
            outputs.setdefault(output, set()).update(used)
            used = set()
    if (
        len(outputs) < 2
        or not loaded
        or any(count != 1 for count in loaded.values())
        or any(tiles != set(loaded) for tiles in outputs.values())
    ):
        raise ValueError("shared_input needs one resident input load set consumed by multiple output tensors")


def check_modes(instructions, modes, parameters, *, target, runtime_scale_loads=None):
    """Check a single already-bound region. Values are decoded, never bit-sliced."""
    needs = {name for name, required in modes.items() if required}
    if any(row.get("class") in _COMPUTES for row in instructions):
        needs.add("contraction_lineage")
    hardware = _selected_hardware(target, needs)
    if "contraction_lineage" in needs:
        _contraction_lineage(instructions, hardware)
    if "accumulator_preload" in needs:
        _accumulator_seed(instructions, parameters.get("accumulator_preload", {}), hardware)
    commits, computes, loads = _events(instructions, hardware)
    if not commits:
        raise ValueError("UNKNOWN: composition has no output store")
    if needs & {"readout_multipliers", "column_block"}:
        name = "column_block" if "column_block" in needs else "readout_multipliers"
        _readout(
            commits,
            parameters.get(name, {}),
            relu=bool(modes.get("relu")),
            scaled=bool(modes.get("acc_scale")),
            per_block=name == "column_block",
            scratchpad="scratchpad_source_pooling_fenced" in needs,
            rows=instructions,
            scale_loads=runtime_scale_loads,
        )
    elif needs & {"relu", "acc_scale"}:
        raise ValueError("UNKNOWN: fused activation/scale requires declared readout_multipliers")
    if "pooled_store" in needs:
        _pool(commits, parameters.get("pooled_store", {}))
    if "scratchpad_source_pooling_fenced" in needs:
        _scratchpad_pool_fence(instructions)
    if "scaled_accumulate_load" in needs:
        _scaled_load(loads, parameters.get("scaled_accumulate_load", {}))
    if "output_stationary" in needs:
        if not computes or any(_known(config, "dataflow") != hardware["output_stationary"] for _, config in computes):
            raise ValueError("contraction does not execute with output-stationary dataflow")
    if "i32_exit" in needs:
        declared = parameters.get("i32_exit", {})
        exits = _output_commits(commits, declared)
        if any(_known(store, "readout") != "i32" for store, _, _, _ in exits):
            raise ValueError("i32_exit requires full-width stores")
        if declared.get("identity_config") is True and any(
            not config
            or _known(config, "acc_scale_bits") != _IDENTITY_BITS
            or _known(config, "acc_act") != 0
            or _known(config, "pool_stride") != 0
            for _, config, _, _ in exits
        ):
            # A raw sibling beside a runtime readout carries explicit identity
            # state, so leaked scale/activation configuration is observable.
            raise ValueError("raw i32 exit requires identity scale and no activation")
    if "k_extension" in needs:
        _k_extension(commits, parameters.get("k_extension", {}), hardware)
    if "shared_input" in needs:
        _shared_input(instructions, parameters.get("shared_input", {}))


def _dma_footprint(kind, decoded, config, hardware):
    """Bound the final byte touched by strided rows, including broadcast stride."""
    rows, cols = (_known(decoded, key) for key in ("rows", "cols"))
    if any(type(size) is not int or size <= 0 for size in (rows, cols)):
        raise ValueError("UNKNOWN: DMA row/column extent is unresolved")
    if kind in _LOADS:
        accumulator = _known(decoded, "is_acc_addr")
        if type(accumulator) is not bool:
            raise ValueError("UNKNOWN: DMA load destination format is unresolved")
        shrink = _known(config, "shrink") if accumulator else 0
        if type(shrink) is not int or shrink not in (0, 1):
            raise ValueError("UNKNOWN: accumulator load shrink format is unresolved")
        width = hardware["input_bytes"] if not accumulator or shrink else hardware["max_load_bytes"]
        stride = _known(config, "stride")
    else:
        from merlin.common.quant_formats import storage_bits

        bits = storage_bits(_known(decoded, "readout"))
        if bits <= 0 or bits % 8:
            raise ValueError("UNKNOWN: DMA store element width is unresolved")
        width = bits // 8
        stride = _known(config, "out_stride_bytes")
        if _known(config, "pool_stride") != 0:
            porows, pocols, pitch = (_known(config, key) for key in ("porows", "pocols", "pool_out_dim"))
            if any(type(v) is not int or v <= 0 for v in (porows, pocols, pitch)) or pocols > pitch or rows != 1:
                raise ValueError("UNKNOWN: pooled DMA footprint is not established")
            rows = (porows - 1) * pitch + pocols
    if type(width) is not int or width <= 0 or type(stride) is not int or stride < 0:
        raise ValueError("UNKNOWN: DMA element width/active stride is unresolved")
    return (rows - 1) * stride + cols * width


def _logical_runtime_operands(instructions, *, target):
    """Project replay-owned concrete addresses without altering the raw decode."""
    result = copy.deepcopy(instructions)
    if not any(row.get("runtime_buffer_bindings") is not None for row in result):
        return result
    hardware = _selected_hardware(target, {"runtime_dma"})
    load_configs, store_config = {}, {}
    for row in result:
        kind, decoded = row.get("class"), row.get("decoded") or {}
        if kind == "CONFIG_LD":
            load_configs[_known(decoded, "state_id")] = decoded
        elif kind == "CONFIG_ST":
            store_config = decoded
        bindings = row.get("runtime_buffer_bindings")
        if bindings is None or row.get("class") not in _LOADS | {"MVOUT"}:
            continue
        decoded = row["decoded"]
        operand = _known(decoded, "dram")
        if not isinstance(operand, dict) or operand.get("kind") != "const" or type(operand.get("raw")) is not int:
            raise ValueError("UNKNOWN: runtime memory operand is not a concrete address")
        address = operand["raw"]
        # A caller may pass an ELF object directly. Its exact object extent is
        # the same allocation; retain the source argument identity in that case.
        statics = [
            b
            for b in row.get("static_buffer_bindings", [])
            if not any((b["base"], b["byte_extent"]) == (a["base"], a["byte_extent"]) for a in bindings)
        ]
        bindings = [*bindings, *statics]
        matches = [b for b in bindings if b["base"] <= address < b["base"] + b["byte_extent"]]
        if len(matches) != 1:
            raise ValueError("UNKNOWN: runtime memory address has no unique logical buffer binding")
        binding = matches[0]
        if kind in _LOADS:
            state = hardware["load_states"].get(kind)
            if state is None:
                raise ValueError("UNKNOWN: DMA load state is underived")
            config = load_configs.get(state, {})
        else:
            config = store_config
        footprint = _dma_footprint(kind, decoded, config, hardware)
        if address + footprint > binding["base"] + binding["byte_extent"]:
            raise ValueError("DMA footprint exceeds its bound logical buffer")
        logical = (
            dict(kind="global", symbol=binding["symbol"])
            if "symbol" in binding
            else dict(kind="argbase", arg_index=binding["arg_index"])
        )
        decoded["dram"] = dict(**logical, offset=address - binding["base"])
    return result


def check_composition(trace, expected, *, target):
    """Both placement and same-region instruction evidence are mandatory.

    Evidence is supplied by the original grader, never the compiler. Missing
    evidence is UNKNOWN and prevents qualification, including count-only claims.
    """
    from .native_instruction_audit import require_observed_kernels

    try:
        composition = expected["composition"]
        evidence = trace["composition_evidence"]
        binding = require_same_region(evidence["frozen_placement"], composition, evidence)
        require_observed_kernels(evidence["native_instruction_audit"], {binding["symbol"]})
        instructions, ordinal = [], -1
        for row in trace["instructions"]:
            # Ordinals count every retired custom command, as the retirement
            # decoder and the scale-load provenance reader both do.
            if row.get("funct") is not None:
                ordinal += 1
            if row.get("kernel") == binding["symbol"] and row.get("region_id") == binding["region_id"]:
                instructions.append(dict(row, command_ordinal=ordinal) if row.get("funct") is not None else row)
        scale_loads = evidence.get("runtime_scale_loads")
        if scale_loads is not None and scale_loads.get("retirement_trace_sha256") != trace.get(
            "retirement_trace_sha256"
        ):
            raise ValueError("UNKNOWN: runtime scale-load evidence is bound to a different retirement trace")
        instructions = _logical_runtime_operands(instructions, target=target)
        obligations = expected.get("emission_obligations")
        obligation_report = check_emission_obligations(instructions, obligations, target=target) if obligations else {}
        producer = composition.get("producer")
        if producer == "contraction":
            producer_seen = any(row.get("class") in _COMPUTES for row in instructions)
        elif producer == "movement":
            producer_seen = any(row.get("class") in _LOADS for row in instructions)
        elif producer == "operand_sum":
            producer_seen = any(
                row.get("class") in _LOADS and row.get("decoded", {}).get("accumulate") is True for row in instructions
            )
        else:
            raise ValueError("UNKNOWN: composition has no declared producer family")
        if not producer_seen:
            raise ValueError("UNKNOWN: bound producer has no same-region instruction stream")
        scoped = expected.get("stage_scoped_modes")
        if not isinstance(scoped, dict) or not scoped:
            raise ValueError("UNKNOWN: composition has no stage_scoped_modes")
        for stage, modes in scoped.items():
            members = composition.get("stages", {}).get(stage)
            if not isinstance(members, list) or not members or not set(members) <= set(binding["source_op_indices"]):
                raise ValueError("UNKNOWN: audit stage is not bound to source chain members")
            if (
                not isinstance(modes, dict)
                or not modes
                or any(name not in MODES | {"relu", "acc_scale"} or value is not True for name, value in modes.items())
            ):
                raise ValueError("UNKNOWN: unsupported composition stage mode")
            parameters = expected.get("stage_scoped_mode_parameters", {}).get(
                stage, expected.get("mode_parameters", {})
            )
            check_modes(instructions, modes, parameters, target=target, runtime_scale_loads=scale_loads)
        return {"status": "pass", "violations": [], "binding": binding, "emission_obligations": obligation_report}
    except (KeyError, TypeError, ValueError, AttributeError, ImportError) as error:
        return {
            "status": "fail",
            "violations": [f"composition audit: {error}"],
            "evidence_status": "UNKNOWN"
            if "UNKNOWN" in str(error) or isinstance(error, (KeyError, TypeError, AttributeError, ImportError))
            else "failed",
        }
