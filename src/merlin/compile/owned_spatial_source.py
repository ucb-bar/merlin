"""Compile an original declared spatial graph around a closed candidate helper.

The original capsule selects this compiler, its component source and ownership
plan. Candidate LLVM supplies only the separately checked private helper and
completion region. No candidate host wrapper, allocator, reference or ELF is
retained. The ordinary capsule object/link/engine/numerical gates remain next.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def normalize_unused_void_asm_results(module):
    """Represent stock-imported void asm without an unobservable SSA result.

    Void has no value. Only one unused void result is removable; every real
    result and every used result is retained/refused respectively. Operands,
    properties, attributes and operation order remain exact. This supplies no
    instruction-effect, completion or operand proof to a downstream checker.
    """
    from xdsl.dialects import llvm

    selected = [
        op
        for op in module.walk()
        if isinstance(op, llvm.InlineAsmOp)
        and len(op.results) == 1
        and isinstance(op.results[0].type, llvm.LLVMVoidType)
    ]
    if any(op.results[0].uses for op in selected):
        raise ValueError("a used void inline-assembly result has no supported source value")
    for op in selected:
        replacement = type(op).create(
            operands=list(op.operands), result_types=[], properties=dict(op.properties), attributes=dict(op.attributes)
        )
        op.parent.insert_op_before(replacement, op)
        op.parent.erase_op(op)
    module.verify()
    return len(selected)


def _local_document(root, record):
    if type(record) is not dict or set(record) != {"file", "sha256"}:
        raise ValueError("owned source needs an exact capsule-local document pin")
    name = record["file"]
    if type(name) is not str or Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("owned source documents must be direct capsule-local files")
    path = root / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1 << 20:
        raise ValueError("owned source document is indirect, absent or oversized")
    payload = path.read_bytes()
    if _sha(payload) != record["sha256"]:
        raise ValueError("owned source document differs from its frozen declaration")
    return path, payload


def source_document_fields(selected):
    """Closed, versioned source custody; graph documents are never optional."""
    if type(selected) is not dict:
        raise ValueError("owned source requires its original closed document declaration")
    schema = selected.get("schema")
    if schema == "merlin.owned_spatial_source.v1":
        return ("source", "plan")
    if schema == "merlin.owned_spatial_source.v2":
        return ("source", "plan", "source_graph", "source_rewrite")
    raise ValueError("owned source requires its original closed document declaration")


@dataclass(frozen=True)
class SpatialSourceDiagnostic:
    """Trusted owner selection for one exact private original source declaration.

    This selects host intake only. It grants no public visibility, source admission,
    arithmetic permission, numerical qualification or whole-model credit.
    """

    model: str
    declaration_sha256: str
    documents: tuple[tuple[str, str, str], ...]

    def __post_init__(self):
        def digest(value):
            return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

        if type(self.model) is not str or not self.model.strip() or not digest(self.declaration_sha256):
            raise ValueError("private spatial diagnostic requires exact model/declaration identities")
        if (
            type(self.documents) is not tuple
            or any(type(row) is not tuple or len(row) != 3 or type(row[1]) is not str
                   or Path(row[1]).name != row[1] or row[1] in ("", ".", "..") or not digest(row[2])
                   for row in self.documents)
            or tuple(row[0] for row in self.documents) != ("source", "plan", "source_graph", "source_rewrite")
        ):
            raise ValueError("private spatial diagnostic requires its complete original document identities")

    @classmethod
    def from_frozen_capsule(cls, root: Path, *, declaration_sha256: str):
        import yaml

        root = Path(root).absolute()
        path = root / "capsule.yaml"
        if root != root.resolve() or any(p.is_symlink() for p in (root, *root.parents)) or path.is_symlink():
            raise ValueError("private spatial diagnostic requires direct original source custody")
        raw = path.read_bytes()
        if _sha(raw) != declaration_sha256:
            raise ValueError("private spatial diagnostic declaration identity differs")
        declaration = yaml.safe_load(raw)
        selected = declaration.get("owned_host_source", {})
        if (
            declaration.get("label") != "hidden"
            or declaration.get("kind") != "model_slice"
            or declaration.get("source_role") != "model_derived"
            or type(declaration.get("model")) is not str
            or not declaration["model"].strip()
            or selected.get("schema") != "merlin.owned_spatial_source.v2"
        ):
            raise ValueError("private spatial diagnostic requires explicit original model source identity")
        documents = []
        for role in source_document_fields(selected):
            document, payload = _local_document(root, selected[role])
            documents.append((role, document.name, _sha(payload)))
        return cls(declaration["model"], declaration_sha256, tuple(documents))

    def matches(self, declaration, payload):
        """Bind visibility to exact bytes, permitting ordinary byte-identical relocation."""
        selected = declaration.get("owned_host_source", {})
        return (
            declaration.get("label") == "hidden"
            and declaration.get("kind") == "model_slice"
            and declaration.get("source_role") == "model_derived"
            and declaration.get("model") == self.model
            and selected.get("schema") == "merlin.owned_spatial_source.v2"
            and _sha(payload) == self.declaration_sha256
            and tuple((role, selected[role].get("file"), selected[role].get("sha256"))
                      for role in source_document_fields(selected)) == self.documents
        )


_diagnostic_source: ContextVar[SpatialSourceDiagnostic | None] = ContextVar("private_spatial_source", default=None)


@contextmanager
def scoped_spatial_source_diagnostic(selection: SpatialSourceDiagnostic):
    """Select a private source in the trusted caller, outside candidate compilation."""
    if type(selection) is not SpatialSourceDiagnostic:
        raise TypeError("private spatial source requires its typed frozen diagnostic selection")
    token = _diagnostic_source.set(selection)
    try:
        yield
    finally:
        _diagnostic_source.reset(token)


def _source_visibility_selected(declaration, payload):
    if declaration.get("label") == "public":
        return True
    diagnostic = _diagnostic_source.get()
    return diagnostic is not None and diagnostic.matches(declaration, payload)


def _source_graph(root, selected, source, binding):
    """Read actual source SSA for v2; retain the original explicit v1 route."""
    from merlin.llvmlower.connected_spatial import (
        extract_connected_spatial_region,
        group_connected_spatial,
        public_spatial_source_graph,
        spatial_pair_interface,
    )
    from merlin.llvmlower.connected_spatial_route import restore_group_rewrite
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir
    from merlin.xdsl_dialects._common import text

    if selected["schema"] == "merlin.owned_spatial_source.v1":
        return public_spatial_source_graph(source, binding)
    from merlin.frontends.linalg_mlir import parse_mlir_text

    _, graph_bytes = _local_document(root, selected["source_graph"])
    _, rewrite_bytes = _local_document(root, selected["source_rewrite"])
    module = parse_mlir_text(graph_bytes.decode())
    module.verify()
    rewrite = restore_group_rewrite(json.loads(rewrite_bytes))
    if rewrite.device != binding.target or rewrite.granularity != "group":
        raise ValueError("original source region differs from its selected target or group route")
    grouped = group_connected_spatial(module, rewrite)
    if len(grouped.receipt["groups"]) != 1:
        raise ValueError("owned current source must contain one closed spatial region")
    group = grouped.receipt["groups"][0]
    entry = grouped.rewrite.entries[group["symbol"]]
    if entry["components"] != source.get("components"):
        raise ValueError("original source SSA programs differ from the frozen source declaration")
    extracted = extract_connected_spatial_region(module, rewrite, tuple(group["components"]))
    if text(extracted.module, generic=True) != text(module, generic=True):
        raise ValueError("owned source graph contains code outside its complete extracted spatial region")
    # Check the same ordinary source export, including both quantizers and all
    # tensor roles. Neither the candidate nor its asserted effects selects it.
    from merlin.targetgen import corpus_spec as CS

    _, interface = CS.build(source, binding)
    actual = spatial_pair_interface(entry, lambda _: binding, source_reference=source["source_reference"])
    if parse_interface_mlir(actual) != parse_interface_mlir(interface):
        raise ValueError("actual source SSA export differs from its frozen complete interface")
    return module, rewrite, {
        "schema": "captured_spatial_source_graph_v1",
        "source_graph_sha256": _sha(graph_bytes),
        "source_rewrite_sha256": _sha(rewrite_bytes),
        "source_pair": group["components"],
        "current_source": grouped.receipt,
        "origin": "original extracted current SSA and ordinary group programs; no graph reconstruction",
        "target_execution": False,
    }


def _route(record, *, source_sha256, effects_basis, stack_budget_bytes):
    """Decode bounded data; no author module, callback or effects assertion loads."""
    from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
    from merlin.compile.scheduling.program_limits import AddressWindow
    from merlin.compile.scheduling.spatial_producer import SpatialProducerPlan, read_spatial_plan_record
    from merlin.compile.scheduling.spatial_readout import RoundedReadout, SpatialReadoutChain
    from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan, Window2D
    from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
    from merlin.llvmlower.connected_spatial_route import ConnectedSpatialRoute
    from merlin.llvmlower.fresh_tensor_writer import FreshTensorWriterContract, PrivateWorkspaceContract

    if type(record) is not dict or record.get("schema") not in (
        "operator.connected_spatial_original_route_inputs.v1",
        "operator.connected_spatial_original_route_inputs.v2",
    ):
        raise ValueError("owned source requires an explicit original spatial plan")
    packed = record["schema"] == "operator.connected_spatial_original_route_inputs.v2"
    if packed and (
        set(record)
        != {
            "schema",
            "plan_kind",
            "plan",
            "plan_sha256",
            "source_request_sha256",
            "required_scratch_bytes",
            "scratch_alignment",
            "writer_symbol",
            "result_argument",
            "borrowed_symbol",
            "allocation_alignment",
            "source_pair",
            "pointer_alignments",
            "helper_symbol",
        }
        or record["plan_kind"] != "staged_packed"
    ):
        raise ValueError("prospective packed spatial owner requires its exact explicit v2 plan declaration")
    if record["source_request_sha256"] != source_sha256:
        raise ValueError("owned source plan is not bound to the exact frozen source request")
    if packed:
        plan = read_spatial_plan_record(record["plan"], kind="staged_packed", expected_identity=record["plan_sha256"])
    else:
        p = record["plan"]
        if type(p) is not dict or "producer" not in p:
            raise ValueError("old spatial owner accepts only its original gather plan; packed plans require v2")
        producer = dict(p["producer"])
        producer["window"] = Window2D(**{k: tuple(v) for k, v in producer["window"].items()})

        def readout(value):
            return None if value is None else RoundedReadout(tuple(value["scales"]), value["relu"])

        plan = SpatialProducerPlan(
            CountedConvolutionPlan(**producer),
            SpatialMaxReadout(**{k: tuple(v) if isinstance(v, list) else v for k, v in p["spatial"].items()}),
            tuple(SpatialReadoutChain(**{k: readout(v) for k, v in c.items()}) for c in p["readouts"]),
            p["output_tile_width"],
            tuple(p["external_observers"]),
            p["readout_mode"],
            AddressWindow(**p["intermediate"]),
        )
    if plan.identity != record["plan_sha256"]:
        raise ValueError("owned source plan identity differs")
    # These are original reviewed source-call premises, not facts supplied by
    # the candidate or qualification of its body. Independent device/engine
    # checks still have to establish actual implementation correspondence.
    effects = BorrowedBufferEffects(effects_basis, True, True, True)
    workspace = PrivateWorkspaceContract(
        record["required_scratch_bytes"],
        record["scratch_alignment"],
        source_sha256,
        True,
        True,
        True,
    )
    if packed and (
        workspace.bytes < plan.workspace.start + plan.workspace.size or workspace.alignment % plan.panel_alignment
    ):
        raise ValueError("packed original owner does not cover its exact private workspace view and alignment")
    writer = FreshTensorWriterContract(
        record["writer_symbol"],
        record["result_argument"],
        (record["result_argument"],),
        record["borrowed_symbol"],
        record["allocation_alignment"],
        borrowed_effects=effects,
    )
    route = ConnectedSpatialRoute(
        tuple(record["source_pair"]),
        plan,
        writer,
        workspace,
        tuple(record["pointer_alignments"]),
        record["helper_symbol"],
        stack_budget_bytes,
    )
    route.record()
    return route


def emit_selected(capsule, candidate_llvm, command_buffer, output, *, target, timeout, candidate_package):
    """Original capsule selection for the ordinary package/capsule front half."""
    from merlin.compile.mesh import _mesh_tile_binding
    from merlin.llvmlower.codegen import RISCV_FLAGS
    from merlin.runtime.backends.base import harness_build_recipe
    from merlin.targetgen.contract.harness_abi import for_target

    binding = _mesh_tile_binding(target, "i8", "i32", numeric_policy=capsule["numeric_policy"])
    recipe = harness_build_recipe(target).with_effective_abi()
    abi = for_target(target)
    if not abi.fence_symbol:
        raise ValueError("owned spatial source requires the selected separate completion ABI")
    return emit(
        capsule,
        candidate_llvm,
        command_buffer,
        output,
        target=target,
        candidate_package=candidate_package,
        binding=binding,
        compile_flags=(*RISCV_FLAGS, recipe.march(), recipe.mabi()),
        completion_symbol=abi.fence_symbol,
        entry_symbol=abi.entry_symbol,
        timeout=timeout,
    )


def emit(
    capsule,
    candidate_llvm,
    command_buffer,
    output,
    *,
    target,
    candidate_package,
    binding,
    compile_flags,
    completion_symbol,
    entry_symbol,
    timeout=180,
):
    """Emit complete original-host LLVM for an explicitly frozen public source.

    This is a host compiler operation, not a grading shortcut. It accepts no
    expected values, saved result, executable, arbitrary source path or author
    host compiler. Existing ordinary emission invokes the candidate first;
    only its literal closed private regions survive this construction.
    """
    import yaml
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.parser import Parser

    from merlin.llvmlower import codegen, toolchain
    from merlin.llvmlower.compilation_recipe import CompilationRecipe
    from merlin.llvmlower.connected_spatial import group_connected_spatial
    from merlin.llvmlower.connected_spatial_host import select_closed_spatial_device_helper
    from merlin.llvmlower.device_build import DeviceRouting, verify_object_symbol_binding
    from merlin.llvmlower.device_preparation import (
        freeze_device_offload_selection,
        prepare_device_route,
        require_device_offload_artifacts,
    )
    from merlin.llvmlower.host_builder import FnBuilder, append_void_region
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.spatial_scratch_abi import emit_spatial_raw_entry
    from merlin.llvmlower.tensor_helper_inlining import composition_features
    from merlin.targetgen import corpus_spec as CS
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir
    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
    from merlin.xdsl_dialects._common import text

    root = Path(capsule["__dir__"]).absolute()
    if root.resolve() != root or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("owned source requires the ordinary direct frozen capsule")
    declaration_path = root / "capsule.yaml"
    declaration_bytes = declaration_path.read_bytes()
    declaration = yaml.safe_load(declaration_bytes)
    selected = declaration.get("owned_host_source")
    documents = source_document_fields(selected) if type(selected) is dict else ()
    if (
        type(selected) is not dict
        or not documents
        or set(selected) != set(documents) | {"schema", "stack_budget_bytes", "effects_basis"}
        or declaration.get("kind") != "model_slice"
        or not _source_visibility_selected(declaration, declaration_bytes)
        or declaration.get("operation", {}).get("op") != "spatial_producer_pool"
        or not declaration.get("semantic", {}).get("must_accelerate")
        or type(selected["effects_basis"]) is not str
        or not selected["effects_basis"].strip()
    ):
        raise ValueError("owned source lacks its explicit frozen public spatial declaration")
    for key in (
        "name",
        "kind",
        "label",
        "model",
        "source_role",
        "source_reference",
        "inputs",
        "operation",
        "numeric_policy",
        "expected",
        "semantic",
        "interface_mlir",
        "required_oracle_tiers",
        "owned_host_source",
        "input_palette",
    ):
        if capsule.get(key) != declaration.get(key):
            raise ValueError("owned source differs from its frozen capsule declaration")
    from merlin.targetgen import frozen_placement

    frozen_placement.read(declaration, root)
    if declaration.get("software_screen", {}).get("status") != "admitted":
        raise ValueError("owned source has no frozen Phase 0 admission")
    source_path, source_bytes = _local_document(root, selected["source"])
    plan_path, plan_bytes = _local_document(root, selected["plan"])
    extra_documents = tuple(_local_document(root, selected[key]) for key in documents if key not in ("source", "plan"))
    source = json.loads(source_bytes)
    rebuilt, interface = CS.build(source, binding)
    for key in ("inputs", "operation", "numeric_policy", "expected"):
        if rebuilt[key] != declaration[key]:
            raise ValueError("owned graph changes original frozen source semantics: " + key)
    interface_name = declaration["interface_mlir"]
    if Path(interface_name).name != interface_name:
        raise ValueError("owned source requires a capsule-local interface")
    interface_path = root / interface_name
    interface_bytes = interface_path.read_bytes()
    if parse_interface_mlir(interface) != parse_interface_mlir(interface_bytes.decode()):
        raise ValueError("owned graph does not state the complete original frozen interface")
    if command_buffer.get("target") != target or binding.target != target:
        raise ValueError("owned graph target differs from ordinary emission")
    attrs = declaration["operation"]["attributes"]
    producer = attrs["producer"]["attributes"]
    names = [producer["ifm"], producer["weight"], producer["bias"], attrs["out"]]
    logical_abi = {
        "kind": "whole_program",
        "args": [{"tensor": name, "access": "write" if index == 3 else "read"} for index, name in enumerate(names)],
        "outputs": [attrs["out"]],
    }
    if command_buffer.get("kernel_abi") != logical_abi or command_buffer.get("outputs") != [attrs["out"]]:
        raise ValueError("owned source requires its complete original logical ABI and outputs")
    route = _route(
        json.loads(plan_bytes),
        source_sha256=_sha(interface_bytes),
        effects_basis=selected["effects_basis"],
        stack_budget_bytes=selected["stack_budget_bytes"],
    )
    original_route = route.record()
    output = Path(output).absolute()
    if output.resolve() != output or any(p.is_symlink() for p in output.parents):
        raise ValueError("owned source output must be in the original direct compilation owner")
    output.mkdir(parents=True, exist_ok=False)
    recipe = CompilationRecipe(output, producer=Path(__file__).resolve())
    for name, path in (("source", source_path), ("plan", plan_path), ("capsule", declaration_path)):
        recipe.bind_preparation(name, path)
    for path, _ in extra_documents:
        recipe.bind_preparation(path.name, path)
    candidate_source = output / "candidate-input.llvm.mlir"
    candidate_source.write_text(candidate_llvm)
    recipe.bind_preparation("candidate_llvm", candidate_source)

    def run(argv, inputs, destination):
        recipe.run(
            argv,
            runner=lambda args: codegen._run(args, inputs=inputs, outputs=(destination,), timeout_s=timeout),
            inputs=inputs,
            output=destination,
        )

    module, rewrite, graph = _source_graph(root, selected, source, binding)
    grouped = group_connected_spatial(module, rewrite)
    if len(grouped.receipt["groups"]) != 1:
        raise ValueError("owned graph must have one complete original spatial pair")
    group = grouped.receipt["groups"][0]
    route = replace(route, writer=replace(route.writer, symbol=group["symbol"]), source_pair=tuple(group["components"]))
    host_source = output / "source-host.mlir"
    host_source.write_text(text(module, generic=True) + "\n")
    rewrite.write_sidecar(output)
    device = DeviceRouting(
        target,
        Path(candidate_package),
        binding.operand_dtype,
        binding.accum_dtype,
        granularity="group",
        connected_spatial=route,
    )
    invocation = freeze_device_offload_selection(device)
    prepare_device_route(device, host_source, output, early_complete_groups=True, selection=invocation)
    host = invocation.spatial_host
    producer = Parser(make_llvm_context(), candidate_llvm).parse_module()
    private_symbol = host.scratch.receipt["raw_private_helper"]
    helper, helper_receipt = select_closed_spatial_device_helper(
        producer,
        host,
        route.plan,
        helper_symbol=route.helper_symbol,
        entry_symbol=private_symbol,
    )
    functions = [
        f for f in producer.body.block.ops if isinstance(f, llvm.FuncOp) and f.sym_name.data == completion_symbol
    ]
    if len(functions) != 1 or functions[0].function_type != llvm.LLVMFunctionType([], llvm.LLVMVoidType()):
        raise ValueError("actual authored completion differs from the selected ordinary ABI")
    builder = FnBuilder([])
    completion_receipt = append_void_region(builder, functions[0], [])
    completion = ModuleOp(
        [
            llvm.FuncOp(
                completion_symbol,
                functions[0].function_type,
                linkage=llvm.LinkageAttr("external"),
                body=builder.finish(),
            )
        ]
    )
    completion.verify()
    llvm_inputs = []
    for name, item in (("private-helper", helper), ("completion", completion)):
        mlir, ll = output / (name + ".mlir"), output / (name + ".ll")
        mlir.write_text(text(item) + "\n")
        run([toolchain.mlir_translate(), "--mlir-to-llvmir", mlir, "-o", ll], (mlir,), ll)
        llvm_inputs.append(ll)
    host_ll = output / "host.ll"
    host_ll.write_text(
        lower_to_llvm_ir(
            text(host.scratch.module, generic=True),
            workdir=output / "host-lowering",
            vectorize=False,
            features=composition_features(None, has_writers=True),
            timeout=timeout,
        )
    )
    if "@malloc" in host_ll.read_text() or "@free" in host_ll.read_text():
        raise ValueError("selected owned source still requires an unavailable heap owner")
    llvm_inputs.append(host_ll)
    for name, code in (
        ("private-bridge", host.scratch.bridge_source),
        ("public-entry", emit_spatial_raw_entry(host.scratch, route.plan, entry_symbol=entry_symbol, standalone=True)),
    ):
        c, ll = output / (name + ".c"), output / (name + ".ll")
        c.write_text(code)
        run([toolchain.clang(), *compile_flags, "-S", "-emit-llvm", c, "-o", ll], (c,), ll)
        llvm_inputs.append(ll)
    for name in ("private-helper", "private-bridge"):
        ll, obj = output / (name + ".ll"), output / (name + ".o")
        run([toolchain.clang(), *compile_flags, "-c", ll, "-o", obj], (ll,), obj)
    symbol_binding = verify_object_symbol_binding(
        output / "private-helper.o",
        output / "private-bridge.o",
        entry_symbol="_mlir_ciface_" + route.writer.borrowed_symbol,
        kernel_symbol=private_symbol,
        original_kernel_symbol=entry_symbol,
        timeout=timeout,
    )
    linked = output / "original-host.ll"
    run([toolchain.llvm_install() / "bin/llvm-link", "-S", *llvm_inputs, "-o", linked], tuple(llvm_inputs), linked)
    imported = output / "original-host.stock-import.mlir"
    run(
        [
            toolchain.mlir_translate(),
            "--import-llvm",
            "--mlir-print-op-generic",
            "--mlir-print-local-scope",
            linked,
            "-o",
            imported,
        ],
        (linked,),
        imported,
    )
    final_module = Parser(make_llvm_context(), imported.read_text()).parse_module()
    void_results = normalize_unused_void_asm_results(final_module)
    artifact = output / "original-host.llvm.mlir"
    artifact.write_text(text(final_module, generic=True) + "\n")
    verified = output / "original-host.verified.ll"
    run([toolchain.mlir_translate(), "--mlir-to-llvmir", artifact, "-o", verified], (artifact,), verified)
    recipe.completed_product(artifact, kind="llvm_ir")
    require_device_offload_artifacts(invocation)
    if (
        declaration_path.read_bytes() != declaration_bytes
        or source_path.read_bytes() != source_bytes
        or plan_path.read_bytes() != plan_bytes
        or interface_path.read_bytes() != interface_bytes
        or any(path.read_bytes() != payload for path, payload in extra_documents)
        or original_route
        != _route(
            json.loads(plan_bytes),
            source_sha256=_sha(interface_bytes),
            effects_basis=selected["effects_basis"],
            stack_budget_bytes=selected["stack_budget_bytes"],
        ).record()
    ):
        raise ValueError("owned source changed during compilation")
    receipt = dict(
        schema="merlin.owned_spatial_source_compilation.v1",
        graph=graph,
        host=host.receipt,
        private_helper=helper_receipt,
        completion=completion_receipt,
        object_binding=symbol_binding,
        source_sha256=_sha(source_bytes),
        plan_sha256=_sha(plan_bytes),
        candidate_llvm_sha256=_sha(candidate_llvm.encode()),
        artifact_sha256=_sha(artifact.read_bytes()),
        logical_abi="original inputs/output only; scratch owned by original host",
        visibility=declaration["label"],
        private_diagnostic_only=declaration["label"] == "hidden",
        target_execution=False,
        qualification="ordinary structural/numerical/engine gates still required",
        unused_void_asm_results_normalized=void_results,
        whole_model_gate_credit=False,
    )
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return artifact.read_text(), receipt
