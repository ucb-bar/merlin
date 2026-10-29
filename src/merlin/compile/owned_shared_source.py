"""Original captured sibling source owners with private device-only compilation.

The frozen source, placement and ABI choose the host. The candidate compiles only
an exact raw integer source projection. Every original host quantizer and live
result remains in the linked original owner. Ordinary capsule numerical, trace,
object, memory and engine gates are still mandatory; no model credit is granted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "merlin.owned_sibling_source.v1"
RECEIPT = "merlin.owned_sibling_source_compilation.v1"


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def selected(capsule):
    return (capsule.get("owned_host_source") or {}).get("schema") == SCHEMA


def source_document_fields(selection):
    if (
        type(selection) is not dict
        or set(selection) != {"schema", "source_graph", "plan"}
        or selection["schema"] != SCHEMA
    ):
        raise ValueError("original sibling source needs its complete frozen document selection")
    return ("source_graph", "plan")


def _read(capsule, *, target):
    import yaml

    from merlin.compile.owned_spatial_source import _local_document
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
    from merlin.llvmlower.source_region_host import prepare_sibling_source_host, tensor_abi
    from merlin.targetgen.frozen_placement import read as placement

    root = Path(capsule["__dir__"]).absolute()
    if root != root.resolve() or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("original source owner must be a direct capsule custody directory")
    declaration_path = root / "capsule.yaml"
    if declaration_path.is_symlink() or not declaration_path.is_file():
        raise ValueError("original sibling source declaration is absent or indirect")
    raw = declaration_path.read_bytes()
    declaration = yaml.safe_load(raw)
    for key in (
        "name",
        "kind",
        "label",
        "inputs",
        "operation",
        "numeric_policy",
        "expected",
        "semantic",
        "interface_mlir",
        "linalg_mlir",
        "required_oracle_tiers",
        "owned_host_source",
        "software_screen",
        "component_program",
        "integer_partial_sum_bound",
    ):
        if declaration.get(key) != capsule.get(key):
            raise ValueError("original source owner differs from its frozen capsule declaration")
    selection = declaration["owned_host_source"]
    paths = [_local_document(root, selection[k]) for k in source_document_fields(selection)]
    source_path, source_bytes = paths[0]
    plan_path, plan_bytes = paths[1]
    interface = declaration["interface_mlir"]
    if Path(interface).name != interface or declaration.get("linalg_mlir") != interface:
        raise ValueError("original source owner requires its complete local linalg program")
    interface_path = root / interface
    if interface_path.is_symlink() or interface_path.read_bytes() != source_bytes:
        raise ValueError("original source graph differs from its frozen complete public interface")
    frozen = placement(declaration, root)
    plan = json.loads(plan_bytes)
    if (
        type(plan) is not dict
        or set(plan) != {"schema", "source_sha256", "preparation", "borrowed_effects"}
        or plan["schema"] != "merlin.original_sibling_source_plan.v1"
        or plan["source_sha256"] != _sha(source_bytes)
    ):
        raise ValueError("original sibling source plan differs from its exact source")
    effects = BorrowedBufferEffects(**plan["borrowed_effects"])
    source = parse_mlir_text(source_bytes.decode())
    prepared = prepare_sibling_source_host(source, effects=effects)
    if prepared.receipt != plan["preparation"]:
        raise ValueError("original source preparation differs from its frozen derived source plan")
    ins, outs = tensor_abi(source)
    if declaration["inputs"] != list(ins):
        raise ValueError("original source inputs differ from the frozen complete signature")
    program = declaration.get("component_program")
    if program != {
        "schema": "merlin.original_ssa_program.v1",
        "source_sha256": _sha(source_bytes),
        "outputs": list(outs),
    }:
        raise ValueError("original source outputs differ from the frozen complete result binding")
    # Reopen independent source eligibility. No emitted field may create a
    # placement, dtype or supporting-device grant. This initial route supports
    # exactly the source's signed integer sibling roots as accelerator work.
    from merlin.targetgen.frozen_source_eligibility import census as independent_frozen_source_eligibility

    certificate = independent_frozen_source_eligibility({**declaration, "__dir__": str(root)}, target=target)
    eligible = [row for row in certificate.get("regions", []) if row.get("target_eligible") is True]
    if (
        certificate.get("source_mlir_sha256") != _sha(source_bytes)
        or certificate.get("n_eligible") != len(prepared.receipt["outline"]["source_geometries"])
        or len(eligible) != certificate.get("n_eligible")
        or any(
            row.get("semantic_family") != "contraction"
            or row.get("carrier_op") not in {"linalg.generic", "linalg.matmul"}
            for row in eligible
        )
    ):
        raise ValueError("original source eligibility differs from its complete exact sibling projection")
    # Frozen and fresh admission must name the same accelerator carrier census.
    from collections import Counter

    actual = dict(Counter(row["carrier_op"] for row in eligible))
    if actual != frozen["accelerator_operations"]:
        raise ValueError("independent source eligibility differs from frozen accelerator placement")
    pins = [(declaration_path, raw), (interface_path, source_bytes), *paths]

    def validate():
        if any(path.is_symlink() or path.read_bytes() != payload for path, payload in pins):
            raise ValueError("original source owner inputs changed during compiler execution")

    validate()
    return prepared, declaration, pins, validate, certificate


def compiler_input(capsule, generated, *, target):
    """Derive the private compiler interface from the exact frozen original SSA."""
    from xdsl.dialects import func
    from xdsl.dialects.builtin import StringAttr

    from merlin.targetgen.contract.linalg_iface import bind_logical_io_metadata
    from merlin.xdsl_dialects._common import text

    prepared, declaration, pins, validate, certificate = _read(capsule, target=target)
    core = prepared.core.clone()
    f = next(op for op in core.body.block.ops if isinstance(op, func.FuncOp))
    f.sym_name = StringAttr("forward")
    core.attributes["prov.level"] = StringAttr("linalg-on-tensors")
    ins = [
        {
            "name": "core_arg" + str(i),
            "shape": list(t.get_shape()),
            "dtype": str(t.element_type),
            "role": "input" if i == 0 else "weight",
        }
        for i, t in enumerate(f.function_type.inputs)
    ]
    outs = [
        {"name": "core_output" + str(i), "shape": list(t.get_shape()), "dtype": str(t.element_type)}
        for i, t in enumerate(f.function_type.outputs)
    ]
    interface, binding = bind_logical_io_metadata(text(core, generic=True), ins, outs)
    root = Path(generated) / "original-source-projection"
    root.mkdir(parents=True, exist_ok=False)
    path = root / "interface.mlir"
    path.write_text(interface)
    (root / "receipt.json").write_text(
        json.dumps(
            {
                "schema": "merlin.original_sibling_source_projection.v1",
                "source_sha256": _sha(pins[1][1]),
                "preparation": prepared.receipt,
                "binding": binding,
                "interface_sha256": _sha(interface.encode()),
                "full_source_outputs": declaration["component_program"]["outputs"],
                "independent_source_eligibility": certificate,
                "execution": False,
            },
            indent=2,
        )
        + "\n"
    )
    validate()
    return path, validate


def full_command_buffer(capsule, private):
    """Retain the accelerator projection while publishing original host pointers."""
    from copy import deepcopy

    from merlin.compile.owned_spatial_source import _local_document
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.source_region_host import tensor_abi

    path, raw = _local_document(Path(capsule["__dir__"]), capsule["owned_host_source"]["source_graph"])
    ins, outs = tensor_abi(parse_mlir_text(raw.decode()))
    if set(private.get("tensors", {})).intersection(row["name"] for row in (*ins, *outs)):
        raise ValueError("private source projection aliases the original public pointer names")
    cb = {"abi_version": private["abi_version"], "target": private["target"], "tensors": {}, "commands": []}
    cb["tensors"].update({row["name"]: {k: v for k, v in row.items() if k != "name"} for row in ins})
    cb["tensors"].update(
        {row["name"]: {"shape": row["shape"], "dtype": row["dtype"], "role": "output"} for row in outs}
    )
    cb["outputs"] = [row["name"] for row in outs]
    cb["kernel_abi"] = {
        "kind": "whole_program",
        "args": [
            {"tensor": row["name"], "access": "read" if i < len(ins) else "write"}
            for i, row in enumerate((*ins, *outs))
        ],
        "outputs": cb["outputs"],
    }
    cb.setdefault("params", {})["source_owned_host_projection"] = {
        "schema": SCHEMA,
        "source_sha256": _sha(raw),
        "device_projection_only": True,
        "host_work_volume": "unknown; original host source retained separately",
        "device_command_buffer": deepcopy(private),
    }
    return cb


def emit_selected(capsule, candidate_llvm, command_buffer, output, *, target, timeout, candidate_package):
    """Compile the independently frozen host around the actual closed helper."""
    from dataclasses import asdict

    from xdsl.dialects import func, llvm
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.parser import Parser

    from merlin.compile.owned_spatial_source import normalize_unused_void_asm_results
    from merlin.llvmlower import codegen, target_data_layout, toolchain
    from merlin.llvmlower.arena_bind import heap_demand
    from merlin.llvmlower.borrowed_buffer_calls import from_writers
    from merlin.llvmlower.compilation_recipe import CompilationRecipe
    from merlin.llvmlower.device_shim import LogicalBuffer, emit_logical_translation_unit
    from merlin.llvmlower.host_builder import PTR, FnBuilder, append_void_region
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.source_region_host import emit_ranked_source_entry
    from merlin.runtime.backends.base import harness_build_recipe
    from merlin.targetgen.contract.harness_abi import for_target
    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
    from merlin.xdsl_dialects._common import text

    prepared, declaration, pins, validate, certificate = _read(capsule, target=target)
    if command_buffer.get("target") != target:
        raise ValueError("private source compiler returned another target")
    output = Path(output).absolute()
    if output != output.resolve() or any(p.is_symlink() for p in output.parents):
        raise ValueError("source host output must be a direct compilation owner")
    output.mkdir(parents=True, exist_ok=False)
    # Re-derive the exact interface used by the ordinary source projection.
    projected, project_validate = compiler_input(capsule, output, target=target)
    callee = next(op for op in prepared.core.body.block.ops if isinstance(op, func.FuncOp))
    expected = {
        "inputs": [
            {"name": "core_arg" + str(i), "shape": list(t.get_shape()), "dtype": str(t.element_type)}
            for i, t in enumerate(callee.function_type.inputs)
        ],
        "outputs": [
            {"name": "core_output" + str(i), "shape": list(t.get_shape()), "dtype": str(t.element_type)}
            for i, t in enumerate(callee.function_type.outputs)
        ],
    }
    input_names = [r["name"] for r in expected["inputs"]]
    output_names = [r["name"] for r in expected["outputs"]]
    expected_args = [{"tensor": n, "access": "read"} for n in input_names] + [
        {"tensor": n, "access": "write"} for n in output_names
    ]
    if command_buffer.get("kernel_abi") != {"kind": "whole_program", "args": expected_args, "outputs": output_names}:
        raise ValueError("actual private helper differs from the source-derived complete device ABI")
    for row in (*expected["inputs"], *expected["outputs"]):
        spec = command_buffer.get("tensors", {}).get(row["name"], {})
        if spec.get("shape") != row["shape"] or spec.get("dtype") != row["dtype"]:
            raise ValueError("actual private helper changes a source tensor format or shape")
    recipe = CompilationRecipe(output, producer=Path(__file__))
    for path, _ in pins:
        recipe.bind_preparation(path.name, path)
    candidate_source = output / "candidate-input.llvm.mlir"
    candidate_source.write_text(candidate_llvm)
    recipe.bind_preparation("candidate_llvm", candidate_source)
    source = output / "original-host.mlir"
    source.write_text(text(prepared.module, generic=True) + "\n")
    recipe.bind_preparation("prepared_source", source)
    selected_sha = _sha(candidate_llvm.encode())

    def unchanged():
        validate()
        project_validate()
        if _sha(candidate_source.read_bytes()) != selected_sha:
            raise ValueError("selected candidate helper changed during host compilation")

    roster = from_writers(
        source,
        (prepared.writer,),
        tuple((path, _sha(raw)) for path, raw in pins) + ((candidate_source, selected_sha),),
        unchanged,
    )

    def run(argv, inputs, dest):
        recipe.run(
            argv,
            runner=lambda args: codegen._run(args, inputs=inputs, outputs=(dest,), timeout_s=timeout),
            inputs=inputs,
            output=dest,
        )

    abi = for_target(target)
    build = harness_build_recipe(target).with_effective_abi()
    if not abi.fence_symbol:
        raise ValueError("source host requires the selected separate completion ABI")
    layout = target_data_layout.observe_index_width(
        toolchain.clang(), [codegen.RISCV_FLAGS[0], build.march(), build.mabi()]
    )
    if layout["index_bits"] != 64:
        raise ValueError("selected source transport requires independently observed 64-bit descriptors")
    host_ll = output / "host.ll"
    host_ll.write_text(
        lower_to_llvm_ir(
            source.read_text(),
            workdir=output / "host-lowering",
            timeout=timeout,
            hoist_static_allocs=False,
            index_bits=64,
            data_layout=layout["data_layout"],
            borrowed_buffer_calls=roster,
        )
    )
    callee = next(op for op in prepared.core.body.block.ops if isinstance(op, func.FuncOp))
    types = (*callee.function_type.inputs, *callee.function_type.outputs)
    names = (*input_names, *output_names)
    buffers = tuple(
        LogicalBuffer(
            name, "input" if i < len(input_names) else "output", name, tuple(t.get_shape()), str(t.element_type)
        )
        for i, (name, t) in enumerate(zip(names, types, strict=True))
    )
    shim = emit_logical_translation_unit(
        target,
        {
            "original_source_transport": (
                buffers,
                [{"role": b.role, "shape": list(b.shape), "dtype": b.dtype} for b in buffers],
            )
        },
        kernel_symbol_for=lambda _: "original_source_device",
    )
    if shim.skipped or shim.symbols != ("original_source_transport",):
        raise ValueError("ordinary descriptor transport did not cover the complete source core")
    n = len(types)
    bridge = (
        shim.text
        + "\nvoid _mlir_ciface_"
        + prepared.writer.borrowed_symbol
        + "("
        + ",".join("const merlin_lmemref_2d*d" + str(i) for i in range(n))
        + ") {original_source_transport_results result;_mlir_ciface_original_source_transport(&result,"
        + ",".join("d" + str(i) for i in range(n))
        + ");}\n"
    )
    producer = Parser(make_llvm_context(), candidate_llvm).parse_module()
    llvm_inputs = [host_ll]
    region_receipts = {}
    for original, renamed, nargs in (
        (abi.entry_symbol, "original_source_device", n),
        (abi.fence_symbol, abi.fence_symbol, 0),
    ):
        fs = [f for f in producer.body.block.ops if isinstance(f, llvm.FuncOp) and f.sym_name.data == original]
        if len(fs) != 1 or fs[0].function_type != llvm.LLVMFunctionType([PTR] * nargs, llvm.LLVMVoidType()):
            raise ValueError("actual authored helper/completion differs from the selected pointer ABI")
        b = FnBuilder([PTR] * nargs)
        region_receipts[renamed] = append_void_region(b, fs[0], b.entry.args)
        m = ModuleOp([llvm.FuncOp(renamed, fs[0].function_type, linkage=llvm.LinkageAttr("external"), body=b.finish())])
        m.verify()
        p, ll = output / (renamed + ".mlir"), output / (renamed + ".ll")
        p.write_text(text(m) + "\n")
        run([toolchain.mlir_translate(), "--mlir-to-llvmir", p, "-o", ll], (p,), ll)
        llvm_inputs.append(ll)
    # The selected compiler's headers are software ABI, not accelerator facts.
    # Use the original harness compiler's declared sysroot observation.
    from merlin.runtime.backends.spike_model import _mlir_runtime_compiler

    flags = [*codegen.RISCV_FLAGS[1:], build.march(), build.mabi()]
    compiler = _mlir_runtime_compiler(toolchain.clang(), build.compiler, flags)
    for name, code in (
        ("descriptor-transport", bridge),
        ("original-entry", emit_ranked_source_entry(prepared.source, entry_symbol=abi.entry_symbol)),
    ):
        p, ll = output / (name + ".c"), output / (name + ".ll")
        p.write_text(code)
        run([*compiler, "-S", "-emit-llvm", p, "-o", ll], (p,), ll)
        llvm_inputs.append(ll)
    # Exact existing shared software runtime, privately selected by the source
    # owner. Its bounded storage is a BSS object, never a guessed board address.
    from merlin.common.paths import runtime_dir

    runtime = runtime_dir()
    allocator = runtime / "baremetal/spike/merlin_malloc.c"
    math_runtime = runtime / "abi/mlir_runtime.c"
    demand = heap_demand([host_ll.read_text()])
    if demand.unbounded:
        raise ValueError("source host has an unbounded allocation demand")
    budget = ((demand.bytes + demand.allocations * 128 + 63) // 64) * 64
    if not 0 < budget < 1 << 63:
        raise ValueError("source host needs a bounded nonempty software workspace")
    runtime_c, runtime_ll = output / "source-runtime.c", output / "source-runtime.ll"
    runtime_c.write_text(
        "#include <stdint.h>\nstatic unsigned char original_source_arena["
        + str(budget)
        + "] __attribute__((aligned(64)));\n"
        + "#define MERLIN_ARENA_BASE_ADDR ((uintptr_t)original_source_arena)\n"
        + "#define MERLIN_ARENA_SIZE_BYTES "
        + str(budget)
        + 'ULL\n#include "'
        + str(allocator)
        + '"\n#include "'
        + str(math_runtime)
        + '"\n'
    )
    run(
        [
            *compiler,
            *[token for path in build.include_roots for token in ("-I", str(path))],
            "-S",
            "-emit-llvm",
            runtime_c,
            "-o",
            runtime_ll,
        ],
        (runtime_c, allocator, math_runtime),
        runtime_ll,
    )
    linked = output / "original-linked.ll"
    run([toolchain.llvm_install() / "bin/llvm-link", "-S", *llvm_inputs, "-o", linked], tuple(llvm_inputs), linked)
    resolved = output / "original-runtime-linked.ll"
    run(
        [toolchain.llvm_install() / "bin/llvm-link", "-S", linked, "--only-needed", runtime_ll, "-o", resolved],
        (linked, runtime_ll),
        resolved,
    )
    imported = output / "original-host.stock-import.mlir"
    run(
        [
            toolchain.mlir_translate(),
            "--import-llvm",
            "--mlir-print-op-generic",
            "--mlir-print-local-scope",
            resolved,
            "-o",
            imported,
        ],
        (resolved,),
        imported,
    )
    final = Parser(make_llvm_context(), imported.read_text()).parse_module()
    void_results = normalize_unused_void_asm_results(final)
    artifact = output / "original-host.llvm.mlir"
    artifact.write_text(text(final, generic=True) + "\n")
    verified = output / "original-host.verified.ll"
    run([toolchain.mlir_translate(), "--mlir-to-llvmir", artifact, "-o", verified], (artifact,), verified)
    unchanged()
    recipe.completed_product(artifact, kind="llvm_ir")
    report = {
        "schema": RECEIPT,
        "source_sha256": _sha(pins[1][1]),
        "artifact_sha256": _sha(artifact.read_bytes()),
        "candidate_llvm_sha256": selected_sha,
        "preparation": prepared.receipt,
        "writer": asdict(prepared.writer),
        "source_eligibility": certificate,
        "closed_regions": region_receipts,
        "workspace": {
            "heap_demand": demand.to_dict(),
            "arena_bytes": budget,
            "address": "ordinary linked aligned BSS",
            "reentrant": False,
        },
        "runtime": {str(p): _sha(p.read_bytes()) for p in (allocator, math_runtime)},
        "index_layout": layout,
        "unused_void_asm_results_normalized": void_results,
        "target_execution": False,
        "whole_model_gate_credit": False,
        "qualification": (
            "source/host compilation only; ordinary original-output numerical, trace, memory "
            "and engine gates remain required"
        ),
    }
    (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    return artifact.read_text(), report
