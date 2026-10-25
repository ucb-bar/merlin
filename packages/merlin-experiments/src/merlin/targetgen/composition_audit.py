"""Ordered, source-bound composition observations over contract-decoded operands.

No instruction operand is unpacked here. Decoders own field layouts and flag
meanings; selected RTL facts own dataflow codes, load states and tile geometry.
"""

import math
from collections import Counter

from .frozen_placement import require_same_region

MODES = frozenset(
    {
        "readout_multipliers",
        "pooled_store",
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
    if needs & {"scaled_accumulate_load", "k_extension", "contraction_lineage"}:
        from merlin.targetgen.rocc.decode import _semantics

        semantics = _semantics(target)
        observer = getattr(semantics, "operand_observer", None) or semantics
        if "scaled_accumulate_load" in needs:
            mapping = getattr(observer, "load_states", None)
            if mapping is None:
                raise ValueError("UNKNOWN: load-class state selection is underived")
            hardware["load_states"] = dict(mapping.ports)
        if needs & {"k_extension", "contraction_lineage"}:
            isa = semantics.isa_constants(target)
        if "k_extension" in needs:
            dimension = isa.get("DIM")
            if type(dimension) is not int or dimension <= 0:
                raise ValueError("UNKNOWN: contraction tile geometry is underived")
            hardware["tile_k"] = dimension
        if needs & {"k_extension", "contraction_lineage"}:
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


def _output_commits(commits, parameters):
    """Select explicit logical output bindings for independent sibling exits."""
    outputs = parameters.get("outputs")
    if outputs is None:
        return commits
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("UNKNOWN: output-scoped mode needs explicit operand identities")
    wanted = {_identity(value) for value in outputs}
    selected = [event for event in commits if _identity(_known(event[0], "dram")) in wanted]
    observed = {_identity(_known(event[0], "dram")) for event in selected}
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
            resident = [span for span in resident if span[1] <= start or end <= span[0]]
        elif kind == "MVOUT":
            start, end, cols = _accumulator_span(decoded, "acc_addr", "rows", "cols", hardware)
            covered = start
            for left, right, width in sorted(resident):
                if width >= cols and left <= covered < right:
                    covered = right
            if covered < end:
                raise ValueError("readout accumulator rows/columns are not written by the contraction")


def _readout(commits, parameters, *, relu=False, scaled=False, per_block=False):
    commits = _output_commits(commits, parameters)
    expected = _integers(parameters, "acc_scale_bits")
    stores_per_block = _integers(parameters, "stores_per_block") if per_block else None
    if stores_per_block is not None:
        if len(stores_per_block) != len(expected) or any(count <= 0 for count in stores_per_block):
            raise ValueError("UNKNOWN: column_block needs source-derived store counts for each block")
        expected = [bits for bits, count in zip(expected, stores_per_block) for _ in range(count)]
    observed, configs = [], []
    for store, config, index, _computes in commits:
        if _known(store, "readout") != "i8" or _known(store, "is_acc_addr") is not True:
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
    expected = _identity(parameters.get("input"))
    live, loaded, outputs = {}, Counter(), {}
    used = set()
    for row in instructions:
        kind, decoded = row.get("class"), row.get("decoded") or {}
        if kind in _LOADS:
            destination = _known(decoded, "spad_addr")
            source = _identity(_known(decoded, "dram"))
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
            output = _identity(_known(decoded, "dram"))
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
        )
    elif needs & {"relu", "acc_scale"}:
        raise ValueError("UNKNOWN: fused activation/scale requires declared readout_multipliers")
    if "pooled_store" in needs:
        _pool(commits, parameters.get("pooled_store", {}))
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
        return {"status": "pass", "violations": [], "binding": binding}
    except (KeyError, TypeError, ValueError, AttributeError, ImportError) as error:
        return {
            "status": "fail",
            "violations": [f"composition audit: {error}"],
            "evidence_status": "UNKNOWN"
            if "UNKNOWN" in str(error) or isinstance(error, (KeyError, TypeError, AttributeError, ImportError))
            else "failed",
        }
