"""Diagnostic-only isolated primitive execution using the target's existing build/GSIM tools."""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from time import monotonic

from xdsl.printer import Printer

from merlin.perf.instruction_motif import initialized_compute_primitives
from merlin.perf.primitive_probe import extract_primitive_program, render_primitive_host_wrapper
from merlin.runtime.backends.base import get_backend
from merlin.targetgen import address_space, gsim_emulator
from merlin.targetgen.rocc import decode


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def runtime_elf_digest(path: str | Path) -> str:
    """Compare executable bytes including load addresses and symbols, excluding debug metadata.

    GCC embeds a random temporary assembly filename in a STT_FILE symbol on each identical link.
    Debug-only stripping removes this nondeterminism; it does not remove execution symbols such
    as the host-interface mailbox or change loadable bytes. The original ELF stays untouched and
    its full hash still identifies every actual measurement.
    """
    from tempfile import TemporaryDirectory
    from merlin.runtime.elf_audit import _tool
    objcopy = _tool("objcopy")
    if objcopy is None:
        raise ValueError("no object tool available for debug-metadata-independent ELF identity")
    with TemporaryDirectory() as temporary:
        stripped = Path(temporary) / "comparison.elf"
        result = subprocess.run([objcopy, "--strip-debug", str(path), str(stripped)],
                                capture_output=True, timeout=10)
        if result.returncode:
            raise ValueError("unable to derive executable ELF identity")
        return _sha(stripped.read_bytes())


def isolated_primitive_signature(domain: dict):
    """Describe an isolated reference state, never the model's contended live state.

    The exact instruction words name local addresses and execution modes. SRAM geometry comes
    from target facts; the named event is an atomic measurement endpoint, not a fabricated port.
    This domain licenses only a drained isolated cost, not cross-task timing composition.
    """
    from merlin.perf.activity_schedule import ActivityEvent
    from merlin.perf.mechanism_probe import derive_mechanism_signature
    from merlin.xdsl_dialects.lowering.global_plan import ValueRepresentation

    stores = address_space.derive_address_space("gemmini").stores
    if len(stores) != 2 or any(not all((s.row_bytes, s.total_rows, s.banks,
                                      s.row_elems, s.element_dtype)) for s in stores):
        raise ValueError("isolated primitive requires exactly derived operand/accumulator stores")
    operand = min(stores, key=lambda s: s.row_bytes)
    accumulator = max(stores, key=lambda s: s.row_bytes)
    if operand.row_bytes == accumulator.row_bytes:
        raise ValueError("physical operand and accumulator storage are ambiguous")
    operands = domain["initialized_operands"]
    if any(v["address"] < 0 or v["address"] + v["rows"] > operand.total_rows
           or v["cols"] > operand.row_elems for v in operands):
        raise ValueError("isolated primitive operands exceed derived physical storage")
    isa = decode.isa_constants("gemmini")
    destination = decode._pack_fields(domain["instructions"][0]["rs2"]["value"])
    raw_address = domain["instructions"][0]["rs2"]["value"] & decode.MASK32
    accumulator_row = raw_address & ~(isa["ACC_I8"] | isa["ACC_ACCUM"] | isa["FULL_C_BIT"])
    if (accumulator_row + destination["rows"] > accumulator.total_rows
            or destination["rows"] != operands[0]["rows"]
            or destination["cols"] != operands[1]["cols"]):
        raise ValueError("primitive accumulator extent differs or exceeds physical capacity")
    geometry = {s.name: json.dumps({"rows": s.total_rows, "row_elems": s.row_elems,
                                  "row_bytes": s.row_bytes, "banks": s.banks,
                                  "scope": "drained isolated reference state"}, sort_keys=True)
                for s in stores}
    shape = [operands[0]["rows"], operands[0]["cols"], operands[1]["cols"]]
    return derive_mechanism_signature(
        representations=[ValueRepresentation(
            s.name, "explicit-physical-rows", s.element_dtype,
            encoding=domain["target_isa_digest"], quantization="exact-execution-config-payload")
                         for s in stores],
        events=[ActivityEvent("body", "whole_motif", "compute", 0)],
        capacity_regime=geometry, tile_shape=shape,
        edge_cases=["full-operand-tile" if all(n == operand.row_elems for n in shape)
                    else "explicit-partial-operand-tile", "overwrite-not-accumulate",
                    "isolated-drained-boundaries-no-concurrent-external-traffic"],
        repetition_semantics="independent initialized overwrite; isolated reference cost only",
        instruction_semantics=[domain])


def _occupancy_bracket(include_roots, workdir: Path) -> tuple[dict, str, str, str]:
    """Reuse the target's derived seven-way busy partition, never all event counters."""
    from merlin.perf import hw_counters as hc
    from .gemmini_codegen_mlir import _counter_slots, _read_discovered_counter_header

    discovery = hc.counters_for_target("gemmini")
    if discovery.get("status") != "derived":
        raise ValueError("target occupancy counters are unavailable")
    header = _read_discovered_counter_header(discovery)
    counters = hc.derive_occupancy_counters(header)
    codes = hc.event_codes(header)
    capacity = _counter_slots()
    if capacity.get("status") != "derived" or not counters.complete():
        raise ValueError("counter capacity or complete occupancy partition is unavailable")
    partition = get_backend("gemmini").counter_partition_inputs()
    if partition.get("status") != "available":
        raise ValueError("occupancy counter Boolean semantics are unavailable")
    proof = hc.prove_occupancy_partition_from_circt(
        partition["hw_text"], counters, codes, module=partition["module"],
        counter_module=partition["counter_module"], source=partition["source"])
    if proof.get("status") != "proved":
        raise ValueError("occupancy counter partition was not proved: " + str(proof.get("why")))
    bracket = hc.counter_bracket_c(counters, codes, slots=capacity["slots"])
    header_path = Path(discovery["header"]).resolve()
    # Keep generated source independent of checkout/snapshot location while checking
    # the first header selected by the actual quoted-include search order.
    include_name = "include/gemmini_counter.h"
    resolved_header = next((root / include_name for root in
                            (workdir, *(Path(p) for p in include_roots))
                            if (root / include_name).is_file()), None)
    if resolved_header is None or _sha(resolved_header.read_bytes()) != discovery["header_sha256"]:
        raise ValueError("build recipe counter header differs from discovered event schema")
    include = f'#include "{include_name}"'
    before = (f'  printf("{hc.COUNTER_SCHEMA_MARKER} {discovery["header_sha256"]}\\n");\n'
              + bracket["prologue"])
    record = {"kind": "joint_engine_busy_cycles", "header": str(header_path),
              "compiled_header": str(resolved_header.resolve()), "include_spelling": include_name,
              "header_sha256": discovery["header_sha256"], "layout": counters.to_dict(),
              "event_codes": discovery["event_codes"], "slot_of": bracket["slot_of"],
              "partition_proof": proof, "slot_proof": capacity,
              "window": "reset after warm completion; snapshot after timed completion and before readback",
              "scope": "isolated primitive only; no surrounding model traffic"}
    return record, include, before, bracket["epilogue"]


def prepare_primitive_probe(source_artifact: str | Path, workdir: str | Path, *,
                        timeout_seconds: int = 600, profile_counters: bool = False,
                        include_operand_movement: bool = False, fixed_work_slice: bool = False) -> dict:
    """Build and run only the first initialized tile primitive, never its complete source kernel.

    All geometry, strides, pointer bindings, accumulator modes, and instructions are extracted from
    the already compiled short artifact. Setup loads are drained before the warm/body windows; only
    the first K tile is computed, and the golden explicitly sums that tile rather than the source
    capsule's entire K. Output readback and validation happen after the cycle counter stops.
    """
    if not 0 < timeout_seconds <= 600:
        raise ValueError("diagnostic primitive build + execution must fit 600 seconds")
    started = monotonic()
    deadline = started + timeout_seconds
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=False)

    def remaining() -> int:
        value = int(deadline - monotonic())
        if value <= 0:
            raise TimeoutError("primitive build/execution spent its full iteration budget")
        return value

    source = Path(source_artifact).read_bytes()
    module = decode._parse_module(source.decode())
    if module is None:
        raise ValueError("short source artifact does not parse")
    program = extract_primitive_program(module, target="gemmini",
                                        include_operand_movement=include_operand_movement,
                                        include_trailing_operand_movement=fixed_work_slice)
    trace = decode.decode_module(module, target="gemmini")
    primitive = initialized_compute_primitives(trace, target="gemmini")[0]
    rows = trace["instructions"]
    space = address_space.derive_address_space("gemmini")
    operand_store = min((s for s in space.stores if s.row_bytes), key=lambda s: s.row_bytes)
    bits = operand_store.element_bits
    if bits not in (8, 16, 32, 64):
        raise ValueError("primitive test-data emitter lacks a derived integral operand width")
    item_bytes = bits // 8
    capacities: dict[int, int] = {}
    loads = {}
    stride = None
    load_encoding = None
    load_end = (next(index for index, row in enumerate(rows) if "acc_addr" in row.get("decoded", {}))
                if fixed_work_slice else primitive["instruction_indices"][0])
    for i, row in enumerate(rows[:load_end]):
        payload = row.get("decoded", {})
        if payload.get("subtype") == "LD":
            stride = payload.get("stride")
            # This narrow runner supports one load channel at a fixed scale/width. A different
            # load configuration requires a channel-aware initializer, not an implicit default.
            encoding = row.get("rs1", {}).get("raw")
            if load_encoding is not None and encoding != load_encoding:
                raise ValueError("load channel/scale changes inside primitive initialization")
            load_encoding = encoding
        if "spad_addr" not in payload:
            continue
        if (not isinstance(operand_store.total_rows, int) or not isinstance(operand_store.row_elems, int)
                or payload["spad_addr"] < 0
                or payload["spad_addr"] + payload["rows"] > operand_store.total_rows
                or payload["cols"] > operand_store.row_elems):
            raise ValueError("primitive movement exceeds derived operand storage geometry")
        dram = payload.get("dram", {})
        if dram.get("kind") != "argbase" or not isinstance(stride, int) or stride <= 0:
            raise ValueError("primitive initialization has no exact pointer/row stride")
        arg, offset = dram["arg_index"], dram["offset"]
        nbytes = offset + (payload["rows"] - 1) * stride + payload["cols"] * item_bytes
        capacities[arg] = max(capacities.get(arg, 0), nbytes)
        loads[i] = {"arg": arg, "offset": offset, "stride": stride}
    inputs = primitive["initialization_provenance"]
    isolated_primitive_signature(primitive["domain"])  # Validate derived operand/accumulator capacity.
    if include_operand_movement and sum(capacities.values()) > 65536:
        raise ValueError("controlled prefix exceeds the diagnostic host-input byte budget")
    a, b = inputs
    if (a["operand"], b["operand"]) != ("activation", "weight") or a["cols"] != b["rows"]:
        raise ValueError("extracted primitive does not describe a conformant matrix tile")
    m, k, n = a["rows"], a["cols"], b["cols"]
    data = {arg: [((i * 5 + arg * 3) % 11) - 5 for i in range(size // item_bytes)]
            for arg, size in capacities.items()}
    if any(size % item_bytes for size in capacities.values()):
        raise ValueError("unaligned initialization extent")
    aa, bb = (loads[entry["producer_index"]] for entry in inputs)
    expected = []
    for i in range(m):
        for j in range(n):
            expected.append(sum(
                data[aa["arg"]][(aa["offset"] + i * aa["stride"]) // item_bytes + q]
                * data[bb["arg"]][(bb["offset"] + q * bb["stride"]) // item_bytes + j]
                for q in range(k)))
    output = next(row["decoded"] for row in rows if "acc_addr" in row.get("decoded", {}))
    output_bits = address_space.element_bits(output.get("readout"))
    if output_bits not in (8, 16, 32, 64) or (output["rows"], output["cols"]) != (m, n):
        raise ValueError("readback does not cover exactly the computed output tile")
    if any(not -(1 << (output_bits - 1)) <= value < (1 << (output_bits - 1)) for value in expected):
        raise ValueError("diagnostic inputs require output conversion semantics beyond this primitive")
    output_arg, output_offset = output["dram"].get("arg_index"), output["dram"].get("offset")
    store_config = next(row["decoded"] for row in reversed(rows)
                        if row.get("decoded", {}).get("subtype") == "ST")
    output_stride = store_config.get("out_stride_bytes")
    if (output_arg in capacities or output_offset != 0
            or not isinstance(output_stride, int) or output_stride < n * output_bits // 8
            or output_stride % (output_bits // 8)
            or store_config.get("acc_scale") != 1.0 or store_config.get("relu") is not False):
        raise ValueError("readback must be distinct, row-strided, unscaled and unactivated")
    output_row_elements = output_stride // (output_bits // 8)
    output_elements = (m - 1) * output_row_elements + n
    verify_index = "i" if output_row_elements == n else f"(i/{n})*{output_row_elements}+i%{n}"
    declarations = ['#include "include/gemmini_testutils.h"']
    for arg, values in sorted(data.items()):
        declarations.append(f"static int{bits}_t buffer_{arg}[{len(values)}] row_align(1) = {{"
                            + ",".join(str(value) for value in values) + "};")
    declarations.extend((f"static int{output_bits}_t buffer_{output_arg}[{output_elements}] row_align(1);",
                         f"static const int{output_bits}_t expected[{m*n}] = {{"
                         + ",".join(str(value) for value in expected) + "};",
                         f"static int verify_result(void) {{ for (int i=0;i<{m*n};++i) "
                         f"if (buffer_{output_arg}[{verify_index}] != expected[i]) "
                         "return 0; return 1; }"))
    if set(capacities) | {output_arg} != set(range(program.argument_count)):
        raise ValueError("not every source argument is accounted by initialization/readback")
    counter_profile, before_measurement, after_measurement = None, "", ""
    backend = get_backend("gemmini")
    recipe = backend.harness_build_recipe()
    if profile_counters:
        counter_profile, counter_include, before_measurement, after_measurement = _occupancy_bracket(
            recipe.include_roots, work)
        declarations.append(counter_include)
        if include_operand_movement:
            counter_profile["scope"] = (
                "controlled fixed-work slice with all selected loads and compute; future computes omitted symmetrically"
                if fixed_work_slice else
                "controlled source prefix with queued movement and compute; not full model context")
    wrapper = render_primitive_host_wrapper(
        program, declarations="\n".join(declarations),
        argument_expressions=[f"buffer_{i}" for i in range(program.argument_count)],
        cycle_reader="read_cycles", verify_call="verify_result()",
        before_measurement=before_measurement, after_measurement=after_measurement)
    stream = io.StringIO()
    Printer(stream=stream).print_op(program.module)
    mlir = stream.getvalue() + "\n"
    (work / "primitive.mlir").write_text(mlir)
    (work / "primitive.c").write_text(wrapper)
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    llvm_ir = lower_to_llvm_ir(mlir, workdir=work, timeout=remaining())
    llvm_file, obj = work / "primitive.ll", work / "primitive.o"
    llvm_file.write_text(llvm_ir)
    compile_env = dict(os.environ, MERLIN_COMPILE_TIMEOUT_S=str(remaining()))
    command = [sys.executable, "-c",
               "from merlin.llvmlower.codegen import compile_ll; import sys; "
               "compile_ll(sys.argv[1],sys.argv[2],'riscv')", str(llvm_file), str(obj)]
    compiled = subprocess.run(command, capture_output=True, text=True, env=compile_env, timeout=remaining())
    (work / "object_build.log").write_text(compiled.stdout + compiled.stderr)
    if compiled.returncode:
        raise RuntimeError(f"primitive object compilation failed: {compiled.stderr[-2000:]}")
    elf = work / "primitive.elf"
    command = [str(recipe.compiler), *recipe.cflags]
    for include in recipe.include_roots:
        command.extend(("-I", str(include)))
    command.extend(("-T", str(recipe.link_script), str(work / "primitive.c"), str(obj), "-o", str(elf)))
    command.extend(str(path) for path in recipe.support_sources)
    linked = subprocess.run(command, capture_output=True, text=True, timeout=remaining())
    (work / "link.log").write_text(linked.stdout + linked.stderr)
    if linked.returncode:
        raise RuntimeError(f"primitive link failed: {linked.stderr[-2000:]}")
    scope = ("controlled_fixed_work_slice" if fixed_work_slice else
             "controlled_source_prefix" if include_operand_movement else "isolated_primitive")
    domain = (_sha(json.dumps({"source_sha256": _sha(source), "primitive_domain": program.domain_digest,
                              "timed_instruction_indices": program.timed_instruction_indices,
                              "measurement_scope": scope}, sort_keys=True).encode())
              if include_operand_movement else program.domain_digest)
    prepared = {"schema": (f"{scope}_preparation_v1" if include_operand_movement
                            else "isolated_primitive_preparation_v1"), "workdir": str(work.resolve()),
                "source_artifact_sha256": _sha(source), "domain_digest": domain,
                "primitive_domain_digest": program.domain_digest, "measurement_scope": scope,
                "timed_instruction_indices": list(program.timed_instruction_indices),
                "output_row_stride_bytes": output_stride,
                "host_input_bytes": sum(capacities.values()),
                "output_storage_bytes": output_elements * (output_bits // 8),
                "source_instruction_count": len(rows),
                "timed_instruction_count": len(program.timed_instruction_indices),
                "primitive_mlir_sha256": _sha(mlir.encode()), "wrapper_sha256": _sha(wrapper.encode()),
                "elf_sha256": _sha(elf.read_bytes()), "computed_tile": {"m": m, "k": k, "n": n},
                "source_completion_instruction_index": program.source_completion_index,
                "elapsed_seconds": monotonic() - started, "timeout_seconds": timeout_seconds,
                "simulator_executed": False, "counter_profile": counter_profile}
    (work / "preparation_receipt.json").write_text(json.dumps(prepared, indent=2) + "\n")
    return prepared


def execute_prepared_primitive(prepared: dict, *, timeout_seconds: float = 600) -> dict:
    """Execute a prebuilt, hash-bound isolated ELF once (warm 1 + measured 1)."""
    if not 0 < timeout_seconds <= 600:
        raise ValueError("diagnostic primitive execution must fit 600 seconds")
    timeout_seconds = min(timeout_seconds, 600 - float(prepared["elapsed_seconds"]))
    if timeout_seconds <= 0:
        raise TimeoutError("primitive preparation exhausted its bounded diagnostic budget")
    started = monotonic()
    work = Path(prepared["workdir"])
    for name, key in (("primitive.mlir", "primitive_mlir_sha256"),
                      ("primitive.c", "wrapper_sha256"), ("primitive.elf", "elf_sha256")):
        if _sha((work / name).read_bytes()) != prepared[key]:
            raise ValueError(f"prepared primitive changed before execution: {name}")
    elf = work / "primitive.elf"
    backend = get_backend("gemmini")
    available, why = backend.gsim_status()
    if not available:
        raise RuntimeError(why)
    engine = gsim_emulator.citation("gemmini")
    try:
        console = backend.run_elf(elf, simulator="gsim", timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        console = (error.stdout or b"")
        if isinstance(console, bytes):
            console = console.decode(errors="replace")
        (work / "console.log").write_text(console)
        raise
    (work / "console.log").write_text(console)
    if _sha(elf.read_bytes()) != prepared["elf_sha256"]:
        raise ValueError("prepared primitive ELF changed during execution")
    if gsim_emulator.citation("gemmini") != engine:
        raise ValueError("simulator provenance changed during primitive execution")
    metric_rows = [line for line in console.splitlines() if line.startswith("MERLIN_PRIMITIVE ")]
    if len(metric_rows) != 1:
        raise ValueError("one measured primitive receipt was not observed")
    metric = dict(token.split("=", 1) for token in metric_rows[0].split()[1:])
    if metric.get("correct") != "1" or metric.get("warmup_runs") != "1" or metric.get("measured_runs") != "1":
        raise ValueError(f"primitive correctness/warm contract failed: {metric}")
    contextual = prepared.get("measurement_scope") in {"controlled_source_prefix", "controlled_fixed_work_slice"}
    result = {"schema": (f"{prepared['measurement_scope']}_execution_v1" if contextual else
                          "isolated_primitive_execution_v1"), "correct": True,
              "target": "gemmini", "simulator": "gsim", "engine_provenance": engine,
              "source_artifact_sha256": prepared["source_artifact_sha256"],
              "domain_digest": prepared["domain_digest"],
              "primitive_mlir_sha256": prepared["primitive_mlir_sha256"],
              "wrapper_sha256": prepared["wrapper_sha256"],
              "elf_sha256": _sha(elf.read_bytes()), "console_sha256": _sha(console.encode()),
              "warmup_runs": 1, "measured_runs": 1,
              "total_compute_cycles": int(metric["total_compute_cycles"]),
              "computed_tile": prepared["computed_tile"],
              "elapsed_seconds": monotonic() - started + prepared["elapsed_seconds"],
              "execution_elapsed_seconds": monotonic() - started,
              "timeout_seconds": timeout_seconds,
              "full_model_executed": False, "full_source_probe_executed": False,
              "measurement_scope": prepared.get("measurement_scope", "isolated_primitive"),
              "measured_scope": ("body call, all selected operand/competing loads and config changes, "
                                 "overwrite compute and completion" if contextual else
                                 "body call, original preload/compute pair and completion"),
              "excluded_from_cycles": (["host input initialization", "entry configuration/drain", "warm body",
                                         "output readback", "verification"] if contextual else
                                        ["operand initialization/drain", "warm body", "output readback", "verification"]),
              "in_context_cycles": None,
              "licence": ("controlled fixed-work slice; future computes omitted symmetrically; model traffic/cache UNKNOWN"
                          if prepared.get("measurement_scope") == "controlled_fixed_work_slice" else
                          "controlled queued source prefix; future model traffic/cache mapping UNKNOWN" if contextual
                          else "isolated primitive plus call/completion overhead; whole-model contention UNKNOWN")}
    profile = prepared.get("counter_profile")
    if profile is not None:
        from merlin.perf import hw_counters as hc
        if _sha(Path(profile["header"]).read_bytes()) != profile["header_sha256"]:
            raise ValueError("occupancy header changed since probe preparation")
        if hc.parse_counter_schema(console) != profile["header_sha256"]:
            raise ValueError("executed occupancy counter header identity is missing or changed")
        values = hc.parse_counter_output(console)
        names = set(profile["event_codes"])
        if set(values) != names:
            raise ValueError("isolated occupancy counter reading is incomplete or has extra events")
        active = sum(values.values())
        if active > result["total_compute_cycles"]:
            raise ValueError("busy counter partition exceeds the measured compute window")
        layout = profile["layout"]
        busy = {unit: sum(values[name] for combo, name in layout["by_combination"].items()
                          if unit in combo.split("+")) for unit in layout["engines"]}
        overlap = sum(values[name] for combo, name in layout["by_combination"].items()
                      if len(combo.split("+")) > 1)
        result["counter_profile"] = {
            **profile, "values": values, "busy_cycles_by_engine_token": busy,
            "overlap_any_engine_cycles": overlap, "active_union_cycles": active,
            "idle_cycles": result["total_compute_cycles"] - active,
            "status": (f"measured_{prepared['measurement_scope']}_counter_partition" if contextual
                       else "measured_isolated_counter_partition"),
            "in_context_occupancy": None,
            "source_engine_binding": "counter semantics from recorded core HW; execution engine lineage recorded separately"}
    (work / "execution_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_primitive_probe(source_artifact: str | Path, workdir: str | Path, *,
                        timeout_seconds: int = 600, profile_counters: bool = False,
                        include_operand_movement: bool = False, fixed_work_slice: bool = False) -> dict:
    """Convenience diagnostic; charge preparation and execution to the same deadline."""
    started = monotonic()
    prepared = prepare_primitive_probe(source_artifact, workdir, timeout_seconds=timeout_seconds,
                                      profile_counters=profile_counters,
                                      include_operand_movement=include_operand_movement,
                                      fixed_work_slice=fixed_work_slice)
    remaining = timeout_seconds - (monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("primitive preparation exhausted iteration budget")
    return execute_prepared_primitive(prepared, timeout_seconds=remaining)
