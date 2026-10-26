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


def _selected_hardware(target, needs):
    """Only load facts needed by the requested modes; absent is never a default."""
    hardware = {}
    if "output_stationary" in needs:
        from merlin.targetgen.rtl.facts import load_facts

        facts = load_facts(target)
        structure = facts.get("facts", facts).get("timing_structure", {})
        dataflow = structure.get("dataflow", {})
        codes = [int(code) for code, name in dataflow.get("modes", {}).items() if name == "output_stationary"]
        if dataflow.get("status") != "derived" or len(codes) != 1:
            raise ValueError("UNKNOWN: output-stationary select code is underived")
        hardware["output_stationary"] = codes[0]
    if needs & {"scaled_accumulate_load", "k_extension", "contraction_lineage", "accumulator_preload", "runtime_dma"}:
        from merlin.targetgen.rocc.decode import _semantics

        semantics = _semantics(target)
        observer = getattr(semantics, "operand_observer", None) or semantics
        if needs & {"scaled_accumulate_load", "runtime_dma"}:
            mapping = getattr(observer, "load_states", None)
            if mapping is None:
                raise ValueError("UNKNOWN: load-class state selection is underived")
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
    needs = {"scaled_accumulate_load"} if (
        check in {"unit_mvin_scale", "nonnegative_stride"}
        and any(r.get("class") in _LOADS for r in instructions)
    ) else set()
    hardware = _selected_hardware(target, needs)
    commits, computes, loads = _events(instructions, hardware)
    transfers = [r["decoded"] for r in instructions if r.get("class") in _LOADS | {"MVOUT"}]
    if check == "unit_mvin_scale":
        return [_known(config, "scale") for _, config in loads]
    if check == "weight_stationary":
        from merlin.targetgen.rtl.facts import load_facts

        facts = load_facts(target)
        dataflow = facts.get("facts", facts).get("timing_structure", {}).get("dataflow", {})
        codes = [int(code) for code, mode in dataflow.get("modes", {}).items() if mode == "weight_stationary"]
        if dataflow.get("status") != "derived" or len(codes) != 1:
            raise ValueError("UNKNOWN: weight-stationary select code is underived")
        return ["WS" if _known(config, "dataflow") == codes[0] else "OS" for _, config in computes]
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
        return ([_known(config, "stride") for _, config in loads]
                + [_known(config, "out_stride_bytes") for _, config, _, _ in commits])
    if check == "max_columns":
        return [_known(store, "cols") for store, _, _, _ in commits]
    if check == "full_i32":
        return ["full_i32" if _known(store, "readout") == "i32" else _known(store, "readout")
                for store, _, _, _ in commits]
    if check == "primitive_k_bound":
        return [_known(compute, "a_cols") * 128 * 128 for compute, _ in computes]
    if check in {"mvout_pool_stride", "mvout_pool_size"}:
        return [_known(config, check.removeprefix("mvout_")) for _, config, _, _ in commits]
    if check in {"mvout_scale", "mvout_relu"}:
        field, identity = ("acc_scale", 1.0) if check == "mvout_scale" else ("acc_act", 0)
        return [_known(config, field) != identity for _, config, _, _ in commits]
    if check == "pool_off":
        return [_known(config, "pool_stride") == 0 and _known(config, "pool_size") == 0
                for _, config, _, _ in commits]
    if check == "raw_i32_exit":
        return [_known(store, "readout") == "i8" or (_known(store, "readout") == "i32" and (
                    _known(config, "pool_stride") == 0 and _known(config, "pool_size") == 0
                    and _known(config, "acc_scale") == 1.0 and _known(config, "acc_act") == 0
                    and 1 <= _known(store, "cols") <= 16))
                for store, config, _, _ in commits]
    if check == "pool_window":
        # _pool already binds the complete window to the declared mode. These
        # are the row's hardware limits, independently of that mode's values.
        return [_known(store, "readout") == "i8" and 1 <= _known(store, "cols") <= 16
                and all(1 <= _known(config, field) <= 3 for field in ("pool_size", "pool_stride"))
                and all(1 <= _known(config, field) <= 255
                        for field in ("pool_out_dim", "porows", "pocols", "orows", "ocols"))
                for store, config, _, _ in commits]
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
        if (not values and duty["check"] in {"weight_stationary", "primitive_k_bound"}
                and any(r.get("class") in _LOADS for r in instructions)):
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


def _readout(commits, parameters, *, relu=False, scaled=False, per_block=False, scratchpad=False):
    commits = _output_commits(commits, parameters)
    expected = _integers(parameters, "acc_scale_bits")
    stores_per_block = _integers(parameters, "stores_per_block") if per_block else None
    if stores_per_block is not None:
        if len(stores_per_block) != len(expected) or any(count <= 0 for count in stores_per_block):
            raise ValueError("UNKNOWN: column_block needs source-derived store counts for each block")
        expected = [bits for bits, count in zip(expected, stores_per_block) for _ in range(count)]
    observed, configs = [], []
    for store, config, index, _computes in commits:
        if _known(store, "readout") != "i8" or _known(store, "is_acc_addr") is not (not scratchpad):
            raise ValueError("scaled/activated readout requires a narrow i8 store")
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
    if per_block:
        boundary = 0
        for count in stores_per_block:
            if boundary and configs[boundary] == configs[boundary - 1]:
                raise ValueError("column_block requires per-block CONFIG_ST re-issue")
            boundary += count
        if observed != expected:
            raise ValueError("column_block requires ordered per-block multiplier values")


def _pool(commits, parameters):
    fields = parameters.get("fields")
    required = {"pool_stride", "pool_size", "pool_out_dim", "porows", "pocols", "orows", "ocols", "upad", "lpad"}
    if not isinstance(fields, dict) or set(fields) != required or any(type(v) is not int for v in fields.values()):
        raise ValueError("UNKNOWN: pooled_store needs the complete declared pool window")
    if fields["pool_stride"] <= 0:
        raise ValueError("pooled_store requires an enabled pooling stride")
    for _store, config, _index, _computes in commits:
        if any(_known(config, key) != value for key, value in fields.items()):
            raise ValueError("active pooled-store window differs from declaration")


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
    expected = _integers(parameters, "scale_bits")
    observed = []
    for load, config in loads:
        if _known(load, "is_acc_addr") is True and _known(load, "accumulate") is True:
            observed.append(_known(config, "scale_bits"))
    if Counter(observed) != Counter(expected):
        raise ValueError("scaled accumulator loads differ from declared operand multipliers")


def _k_extension(commits, parameters, hardware):
    source, extension = parameters.get("source_k"), parameters.get("extension_k")
    if type(source) is not int or source <= 0 or type(extension) is not int or extension <= 0:
        raise ValueError("UNKNOWN: k_extension needs source and coefficient-digit K extents")
    tile = hardware["tile_k"]
    original = (source + tile - 1) // tile
    extended = (source + extension + tile - 1) // tile
    if extended <= original or any(len(work) != extended for _, _, _, work in commits):
        raise ValueError("k_extension lacks the declared extra K tiles per output")
    mask = hardware["address_flags"]
    for store, _, _, work in commits:
        destination = _known(store, "acc_addr")
        if type(destination) is not int:
            raise ValueError("UNKNOWN: output accumulator address is unresolved")
        for _, preload in work:
            address = _known(preload or {}, "c_addr")
            if type(address) is not int or address & ~mask != destination & ~mask:
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


def check_modes(instructions, modes, parameters, *, target):
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
        exits = _output_commits(commits, parameters.get("i32_exit", {}))
        if any(_known(store, "readout") != "i32" for store, _, _, _ in exits):
            raise ValueError("i32_exit requires full-width stores")
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
        width = hardware["max_load_bytes"] if accumulator else hardware["input_bytes"]
        stride = _known(config, "stride")
    else:
        from merlin.common.quant_formats import storage_bits

        bits = storage_bits(_known(decoded, "readout"))
        if bits <= 0 or bits % 8:
            raise ValueError("UNKNOWN: DMA store element width is unresolved")
        width = bits // 8
        stride = _known(config, "out_stride_bytes")
        if _known(config, "pool_stride") != 0:
            raise ValueError("UNKNOWN: pooled DMA footprint is not established")
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
        decoded["dram"] = dict(kind="argbase", arg_index=binding["arg_index"], offset=address - binding["base"])
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
        instructions = [
            row
            for row in trace["instructions"]
            if row.get("kernel") == binding["symbol"] and row.get("region_id") == binding["region_id"]
        ]
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
            check_modes(instructions, modes, expected.get("mode_parameters", {}), target=target)
        return {"status": "pass", "violations": [], "binding": binding,
                "emission_obligations": obligation_report}
    except (KeyError, TypeError, ValueError, AttributeError, ImportError) as error:
        return {
            "status": "fail",
            "violations": [f"composition audit: {error}"],
            "evidence_status": "UNKNOWN"
            if "UNKNOWN" in str(error) or isinstance(error, (KeyError, TypeError, AttributeError, ImportError))
            else "failed",
        }
