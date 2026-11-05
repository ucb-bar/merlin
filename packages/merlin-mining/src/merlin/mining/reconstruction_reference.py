"""Automatic source/capture/target binding for reference-guided experiments.

Select the capture protocol and backend contract once. Each reference's source
semantics, shapes and CCA determine the ordinary reconstruction job; no handwritten
per-kernel graph or schedule enters the compiler. Physical packing witnesses and
preparation boundaries remain distinct from logical mathematical equivalence.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path

from merlin.kernels.kernel_evidence import (
    RegionOperation,
    RegionTensor,
    SemanticRegion,
)
from merlin.kernels.reference_inventory import digest, read_pinned


def _endpoint(binding, *, root, backend=False):
    selection = binding.get("factory")
    if type(selection) is not str:
        raise ValueError("selected reference protocol requires a factory")
    module_name, separator, name = selection.partition(":")
    if not separator or not name.isidentifier() or not all(part.isidentifier() for part in module_name.split(".")):
        raise ValueError("protocol factory must be module:callable")
    search = None
    backend_root = None
    if backend:
        backend_root = (root / binding["backend_root"]).resolve()
        search = (root / binding.get("module_root", binding["backend_root"])).resolve()
        if not search.is_dir():
            raise ValueError("selected backend root is unavailable")
        # A fresh CLI process imports the selected backend once. Reusing an
        # already imported package from another checkout is an identity error.
        package = sys.modules.get(module_name.split(".")[0])
        if package is not None and any(
            not Path(path).resolve().is_relative_to(search) for path in getattr(package, "__path__", ())
        ):
            raise ValueError("a different backend package is already imported; use a fresh compiler process")
        sys.path.insert(0, str(search))
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module = importlib.import_module(module_name)
    finally:
        sys.dont_write_bytecode = previous_bytecode
        if search is not None:
            sys.path.remove(str(search))
    factory = getattr(module, name)
    source = inspect.getsourcefile(factory)
    if not callable(factory) or source is None:
        raise ValueError("selected protocol has no inspectable callable source")
    source = Path(source).resolve()
    if search is not None and not source.is_relative_to(search):
        raise ValueError("selected backend factory resolved outside its pinned root")
    data = source.read_bytes()
    if binding.get("sha256") is not None:
        read_pinned(source, binding["sha256"])
    return factory, dict(entrypoint=selection, source=str(source), sha256=digest(data)), backend_root


def _cca(record):
    if record is None or hasattr(record, "scope_problems"):
        return record
    from merlin.kernels import cca

    names = {
        "compute": cca.ComputeFacet,
        "vector": cca.VectorFacet,
        "memory": cca.MemoryFacet,
        "envelope": cca.EnvelopeFacet,
        "spatial": cca.SpatialFacet,
        "simt": cca.SimtFacet,
        "dispatch": cca.DispatchFacet,
        "layout": cca.LayoutFacet,
        "coverage": cca.CoverageFacet,
        "communication": cca.CommunicationFacet,
    }
    values = dict(record)
    for name, cls in names.items():
        if values.get(name) is not None:
            row = dict(values[name])
            if set(row) - {field.name for field in fields(cls)}:
                raise ValueError("capture CCA uses an unsupported facet field")
            values[name] = cls(**row)
    result = cca.CCA(**values)
    if result.scope_problems():
        raise ValueError("capture CCA has incompatible scope")
    return result


def dense_source(problem):
    """Construct plain source SSA from established raw integer dense semantics."""
    from merlin.frontends.linalg_mlir import parse_mlir_text

    m, k, n = (problem[key] for key in ("m", "k", "n"))
    if any(type(value) is not int or value <= 0 or value >= 2**31 for value in (m, k, n)):
        raise ValueError("dense source needs complete positive static source geometry")
    if tuple(problem.get("dtypes", ())) != ("i8", "i8", "i32"):
        raise ValueError("automatic dense binding currently requires signed i8 inputs and raw i32 output")
    numeric = problem.get("numeric", {})
    expected = dict(
        accumulator_dtype="i32", bias="none", scale=1, activation="none", rounding="none", saturation="none"
    )
    if dict(numeric) != expected:
        raise ValueError("raw dense reconstruction cannot drop bias, quantization or epilogue semantics")
    return parse_mlir_text(f"""module {{
      func.func @reference_region(%lhs: tensor<{m}x{k}xi8>, %rhs: tensor<{k}x{n}xi8>) -> tensor<{m}x{n}xi32> {{
        %zero = arith.constant 0 : i32
        %empty = tensor.empty() : tensor<{m}x{n}xi32>
        %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<{m}x{n}xi32>) -> tensor<{m}x{n}xi32>
        %out = linalg.generic {{indexing_maps = [affine_map<(m,n,k)->(m,k)>,
            affine_map<(m,n,k)->(k,n)>, affine_map<(m,n,k)->(m,n)>],
            iterator_types = ["parallel", "parallel", "reduction"]}}
          ins(%lhs, %rhs : tensor<{m}x{k}xi8>, tensor<{k}x{n}xi8>) outs(%init : tensor<{m}x{n}xi32>) {{
          ^bb0(%a: i8, %b: i8, %acc: i32):
            %a32 = arith.extsi %a : i8 to i32
            %b32 = arith.extsi %b : i8 to i32
            %product = arith.muli %a32, %b32 : i32
            %sum = arith.addi %acc, %product : i32
            linalg.yield %sum : i32
        }} -> tensor<{m}x{n}xi32>
        func.return %out : tensor<{m}x{n}xi32>
      }}
    }}""")


def dense_source_region(source):
    """Canonicalize only independently recognized complete source computation."""
    from xdsl.dialects import arith, func, tensor
    from xdsl.dialects.linalg.ops import FillOp, GenericOp

    from merlin.transforms.dense_ir import static_zero_integer_matmul_geometry

    functions = tuple(source.body.block.ops)
    if len(functions) != 1 or type(functions[0]) is not func.FuncOp or len(functions[0].body.blocks) != 1:
        raise ValueError("source region is not one complete standalone dense function")
    body = tuple(functions[0].body.block.ops)
    contractions = [op for op in body if type(op) is GenericOp]
    if len(contractions) != 1:
        raise ValueError("source region has no unique integer contraction")
    contraction = contractions[0]
    geometry = static_zero_integer_matmul_geometry(contraction, ("i8", "i8", "i32"))
    if geometry is None:
        raise ValueError("source scalar/indexing body is not ordered zero-seeded integer matmul")
    initializer = contraction.outputs[0].owner
    if type(initializer) is not FillOp:
        raise ValueError("source initializer is not the original zero fill")
    allowed = {contraction, initializer, initializer.inputs[0].owner, initializer.outputs[0].owner, body[-1]}
    if (
        set(body) != allowed
        or type(initializer.inputs[0].owner) is not arith.ConstantOp
        or type(initializer.outputs[0].owner) is not tensor.EmptyOp
        or type(body[-1]) is not func.ReturnOp
        or tuple(body[-1].operands) != tuple(contraction.results)
        or tuple(contraction.inputs) != tuple(functions[0].body.block.args)
    ):
        raise ValueError("source contains additional computation or externally observed intermediates")
    m, k, n = geometry
    return SemanticRegion(
        "compiler-source",
        (
            RegionTensor("lhs", (m, k), "i8", "logical"),
            RegionTensor("rhs", (k, n), "i8", "logical"),
            RegionTensor("out", (m, n), "i32", "logical"),
        ),
        (
            RegionOperation(
                "dense",
                "matmul",
                ("lhs", "rhs"),
                ("out",),
                attributes=dict(m=m, k=k, n=n, reduction_order="ordered_k", initialization="zero"),
                numeric=dict(
                    accumulator_dtype="i32", bias="none", scale=1, activation="none", rounding="none", saturation="none"
                ),
                numeric_complete=True,
            ),
        ),
        ("lhs", "rhs"),
        ("out",),
        external_uses={"out": len(tuple(contraction.results[0].uses))},
        complete=True,
    )


def check_rhs_view(problem, view):
    """Check the exact panel permutation before comparing logical operands.

    wp[n//P,k,n%P] = rhs[k,n]. Euclidean division gives a unique inverse
    n=panel*P+lane on these complete domains. This establishes logical element
    identity, not common input bytes or a free packing/preparation operation.
    """
    k, n = problem["k"], problem["n"]
    if view is None:
        raise ValueError("physical RHS layout needs a checked logical-view witness")
    if view.get("column_offset", 0) != 0:
        raise ValueError("RHS subviews need a separately checked bounded offset map")
    if view.get("kind") in ("row_major", "identity"):
        if tuple(view.get("physical_shape", ())) != (k, n):
            raise ValueError("row-major RHS shape differs from logical source")
    elif view.get("kind") == "column_panels":
        panel = view.get("panel")
        if (
            type(panel) is not int
            or panel <= 0
            or n % panel
            or tuple(view.get("physical_shape", ())) != (n // panel, k, panel)
            or tuple(view.get("logical_shape", ())) != (k, n)
        ):
            raise ValueError("packed RHS view does not form a complete bijective panel permutation")
    else:
        raise ValueError("RHS packing requires an explicit supported index-map witness")
    if not view.get("source_evidence"):
        raise ValueError("RHS view must identify actual source/trace access evidence")
    return {
        **view,
        "checked": True,
        "logical_shape": (k, n),
        "check": "exact domain cardinality and unique Euclidean quotient/remainder inverse",
        "preparation_cost": "not established",
        "same_input_buffer": False,
    }


def _captured_index(captured, workspace):
    """Index large structural evidence without embedding its full command list."""
    region = captured.get("semantic_region")
    if hasattr(region, "to_dict"):
        region = region.to_dict()
    record = dict(
        semantic_region=region,
        capture_path=captured.get("capture"),
        problems=captured.get("problems"),
        tile_dataflow=captured.get("tile_dataflow"),
        reuse_scope=captured.get("reuse_scope"),
    )
    model = captured.get("schedule_model")
    if model is not None:
        value = model.to_dict() if hasattr(model, "to_dict") else model
        path = workspace / "reference-schedule.json"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as output:
            json.dump(value, output, separators=(",", ":"), allow_nan=False)
        record["reference_schedule"] = dict(
            path=str(path),
            sha256=digest(path.read_bytes()),
            command_count=value.get("command_count"),
            complete=value.get("complete"),
            source_kind=value.get("source_kind"),
            evidence_id=value.get("evidence_id"),
        )
    return record


def reference_job(document, index, *, root, workspace):
    """Bind real capture semantics to one selected executable target contract."""
    if workspace is None:
        raise ValueError("automatic reference reconstruction requires an explicit owned workspace")
    root, workspace = Path(root), Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    capture_binding = document.get("capture_binding")
    target_binding = document.get("target_binding")
    if not isinstance(capture_binding, dict) or not isinstance(target_binding, dict):
        raise ValueError("automatic reconstruction requires selected capture and target bindings")
    capture, capture_pin, _ = _endpoint(capture_binding, root=root)
    # The selected target package also owns the source/ISA observer. Its
    # manifest binding must suffice without an ambient editable installation.
    module_root = (root / target_binding.get("module_root", target_binding["backend_root"])).resolve()
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(module_root))
    try:
        captured = capture(document, index, root=root, workspace=workspace / "capture")
    finally:
        sys.path.remove(str(module_root))
        sys.dont_write_bytecode = previous_bytecode
    if not isinstance(captured, dict):
        raise ValueError("capture protocol must return actual semantic/evidence observations")
    captured_index = _captured_index(captured, workspace)
    problem = captured.get("problem")
    if not isinstance(problem, dict):
        return dict(
            gaps=[
                dict(
                    kind="adapter_gap",
                    reason=(
                        "complete captured region has "
                        + str(len((captured_index.get("semantic_region") or {}).get("operations", ())))
                        + " contractions with separate output numerical boundaries; "
                        "selected target binding accepts only one raw i32 dense output"
                    ),
                    goals=("source.complete_fused_region", "numeric.distinct_output_boundaries"),
                    witnesses=({"semantic_region": captured_index.get("semantic_region")},),
                    next_action="bind_complete_fusion_aware_source_and_library_adapter",
                )
            ],
            automatic_reference=dict(
                selected_reference=captured.get("selected_reference"),
                capture_binding=capture_pin,
                semantic_authority="complete captured fused/numerical region retained",
                **captured_index,
            ),
        )
    try:
        source = dense_source(problem)
    except ValueError as error:
        return dict(
            gaps=[
                dict(
                    kind="adapter_gap",
                    reason=str(error),
                    goals=("numeric.complete_source_epilogue",),
                    witnesses=({"problem": problem},),
                    next_action="bind_existing_complete_source_numerics_or_fusion_consumer",
                )
            ],
            automatic_reference=dict(
                selected_reference=captured.get("selected_reference"),
                capture_binding=capture_pin,
                problem=problem,
                semantic_authority="unsupported captured source semantics retained",
                **captured_index,
            ),
        )
    reference = captured["semantic_region"]
    if isinstance(reference, dict):
        reference = SemanticRegion.from_record(reference)
    source_region = dense_source_region(source)
    # Literal 1 and 1.0 are the same identity epilogue on raw accumulator
    # output. Normalize only this proved identity, retaining original evidence.
    if len(reference.operations) == 1 and reference.operations[0].numeric.get("scale") == 1:
        op = reference.operations[0]
        reference = replace(reference, operations=(replace(op, numeric=dict(op.numeric, scale=1)),))
    try:
        view = check_rhs_view(problem, captured.get("rhs_view", problem.get("rhs_view")))
    except ValueError as error:
        return dict(
            gaps=[
                dict(
                    kind="missing_evidence", reason=str(error), next_action="recover_and_check_actual_input_layout_map"
                )
            ],
            automatic_reference=dict(
                selected_reference=captured.get("selected_reference"),
                capture_binding=capture_pin,
                problem=problem,
                semantic_authority="physical operand mapping is not established",
                **captured_index,
            ),
        )
    binder, backend_pin, backend_root = _endpoint(target_binding, root=root, backend=True)
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(backend_root))
    try:
        options = {"backend_root": backend_root}
        if "output_dir" in inspect.signature(binder).parameters:
            options["output_dir"] = workspace / "target"
        adapter = binder(source, **options)
    finally:
        sys.path.remove(str(backend_root))
        sys.dont_write_bytecode = previous_bytecode
    from merlin.transforms.certificates import ir_text

    text = ir_text(source)
    source_path = workspace / "source.mlir"
    source_path.write_text(text)
    source_path.chmod(0o600)
    evidence = captured.get("evidence")
    if hasattr(evidence, "to_dict"):
        evidence = evidence.to_dict()
    reference_cca = _cca(captured.get("cca"))
    comparison_gaps = []
    original_observe = adapter.observe
    reference_count_kind = (evidence or {}).get(
        "capture_kind", (reference_cca.provenance.get("level") if reference_cca is not None else None)
    )

    def observe_comparable(candidate):
        observation = original_observe(candidate)
        ours = observation.cca
        if (
            reference_cca is None
            or reference_cca.dispatch is None
            or reference_cca.dispatch.n_dispatches is None
            or ours is None
            or ours.dispatch is None
            or ours.dispatch.n_dispatches is None
            or reference_count_kind not in ("native_issue", "committed_trace", "normalized_commands")
            or ours.provenance.get("level") != "checked_counted_commands"
            or ours.provenance.get("dispatch_count_scope") == "target_instruction_issue"
        ):
            return observation
        # A target macro can expand one logical call into several actual issue
        # commands. Keep the checked logical count, but never compare its unit
        # with native instructions before an actual emitted-code lift exists.
        ours = deepcopy(ours)
        logical_count = ours.dispatch.n_dispatches
        ours.dispatch.n_dispatches = None
        ours.provenance["uncompared_dispatch"] = dict(value=logical_count, basis="logical_counted_commands")
        ours.provenance.setdefault("axis_evidence", {})["dispatch.n_dispatches"] = dict(
            status="unknown",
            reason="native primitive issue count and pre-emission logical command count have different boundaries",
            evidence=[candidate.emission.schedule_sha256],
        )
        witness = dict(
            program_sha256=digest(candidate.emission.data),
            logical_count=logical_count,
            reference_basis=reference_count_kind,
            candidate_basis="logical_counted_commands",
        )
        if not comparison_gaps:
            comparison_gaps.append(
                dict(
                    kind="adapter_gap",
                    goals=("dispatch.actual_instruction_projection",),
                    reason="target instruction expansion has no matched raw-issue count projection",
                    next_action="lift_actual_emitted_instructions_before_comparing_dispatch_counts",
                    witnesses=[],
                )
            )
        comparison_gaps[0]["witnesses"].append(witness)
        return replace(observation, cca=ours)

    adapter = replace(adapter, observe=observe_comparable)
    scope_goals, scope_requirements = [], []
    for requirement in captured.get("mechanism_requirements", ()):
        if requirement.get("status") in ("observed", "inferred"):
            if not requirement.get("evidence") or type(requirement.get("goal")) is not str:
                raise ValueError("known stronger mechanism requirements need actual evidence")
            scope_goals.append(requirement["goal"])
            scope_requirements.append(requirement)
    residency = reference_cca.memory.onchip_resident if reference_cca is not None and reference_cca.memory else None
    for role, values in (("lhs", ("a", "both")), ("rhs", ("b", "both"))):
        if residency in values and not any(role in goal for goal in scope_goals):
            scope_goals.append("memory." + role + "_retention_scope_covered")
            scope_requirements.append(
                dict(
                    goal=scope_goals[-1],
                    status="unknown",
                    reason="CCA residency alone does not distinguish macro-panel reuse from complete expert lifetimes",
                )
            )
    job = dict(
        adapter=adapter,
        reference_region=reference,
        source_region=source_region,
        reference_cca=reference_cca,
        comparison_gaps=comparison_gaps,
        goals=tuple(dict.fromkeys(("schedule.consumed", *scope_goals))),
        automatic_reference=dict(
            selected_reference=captured.get("selected_reference"),
            capture_binding=capture_pin,
            target_binding=backend_pin,
            evidence=evidence,
            problem=problem,
            rhs_view=view,
            mechanism_requirements=scope_requirements,
            source=dict(path=str(source_path), sha256=digest(text.encode())),
            semantic_authority="source-derived expert math and independently admitted complete frontend SSA",
            physical_layout_comparison="explicit permutation bridge; no same-buffer or preparation-cost claim",
            measurement_status="unknown; matched input preparation and elapsed cycles not established",
            capture_artifacts=captured.get("artifacts", ()),
            **captured_index,
        ),
    )
    # A backend may expose its ordinary independent owner hooks. Missing gates
    # become machine tasks rather than accepting unverified agent patches.
    owner = getattr(importlib.import_module(binder.__module__), "learning_job", None)
    if callable(owner):
        job["learning"] = owner(
            source=source,
            reference_cca=job["reference_cca"],
            backend_root=backend_root,
            workspace=workspace / "learning",
            reference_region=reference,
        )
    else:
        job["learning_gaps"] = [
            dict(
                kind="adapter_gap",
                reason="selected backend has no independent learning gate owner",
                next_action="bind_existing_source_numeric_storage_target_emission_owner",
            )
        ]
    return job
