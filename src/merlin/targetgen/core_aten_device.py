"""Provider-selected offload and executed-instruction evidence for Core ATen shards."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from merlin.targetgen.plugins import load_module
from merlin.targetgen.rtl.facts import ensure_facts, load_facts, observed_facts


def load_execution_provider(target: str, path: Path | None = None):
    """Select target-edge support explicitly; no compiler policy lives in this adapter."""
    if not target or Path(target).name != target or target in {".", ".."}:
        raise ValueError("target must be a single name")
    explicit = path or os.environ.get("MERLIN_CORE_ATEN_EXECUTION_PROVIDER")
    if not explicit:
        raise ValueError(
            "Core ATen device execution requires an explicit target-owned execution provider; "
            "pass --execution-provider or set MERLIN_CORE_ATEN_EXECUTION_PROVIDER"
        )
    selected = Path(explicit)
    if not selected.is_file():
        raise ValueError(f"selected Core ATen execution provider does not exist: {selected}")
    from merlin.targetgen.target_registry import resolve

    info = resolve(target)
    if info.contract_path.is_file() or os.environ.get("MERLIN_TARGET_CONTRACT"):
        info.load_contract()
    identity = hashlib.sha256(str(selected.resolve()).encode()).hexdigest()
    module = load_module(selected.parent, selected.name, package_name=f"core_aten_execution_{identity}")
    if getattr(module, "TARGET", None) != target:
        raise ValueError("execution provider does not declare the selected target")
    return module


@contextmanager
def selected_facts(target: str, explicit: Path | None):
    """Pin nested consumers and lowering subprocesses to the same selected facts file."""
    path = ensure_facts(target, explicit=explicit)
    document = load_facts(target, explicit=path)
    previous = os.environ.get("MERLIN_RTL_FACTS")
    os.environ["MERLIN_RTL_FACTS"] = str(path.resolve())
    try:
        with observed_facts(target, document, path):
            yield document
    finally:
        if previous is None:
            os.environ.pop("MERLIN_RTL_FACTS", None)
        else:
            os.environ["MERLIN_RTL_FACTS"] = previous


def routing_for_bundle(directory: Path, target: str, package: Path, provider, facts: dict, *, compile_products=None):
    """Report structural refusals, then ask the provider to bind supported source operations."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.kernels.shapes import accumulator_init, observe_contractions, zero_initialised
    from merlin.system.offload import device_contraction_ranks, device_dtype_triples, why_not
    from merlin.targetgen.core_aten_spec_int8 import provider_coverage

    module = parse_mlir_file(directory / "model.mlir")
    triples, ranks = device_dtype_triples(target), device_contraction_ranks(target)
    eligible, declined = [], []
    for op, shape in observe_contractions(module):
        reason = why_not(shape, triples=triples, ranks=ranks)
        if reason is None and not zero_initialised(op):
            reason = "contraction destination is not proven zero initialized"
            admission_path = directory / "spec-int8-admission.json"
            if accumulator_init(op) == "int32_seed" and admission_path.is_file():
                admission = json.loads(admission_path.read_text())
                prove = getattr(provider, "admit_numeric_schedule", None)
                if callable(prove):
                    try:
                        proof = prove(str(module), package=package, numeric_policy=admission["selection"]["spec"])
                        reason = provider_coverage(module, proof, op)
                    except (ValueError, KeyError) as exc:
                        reason = f"selected provider refuses seeded contraction: {exc}"
        if reason:
            declined.append(reason)
        else:
            eligible.append((op, shape))
    if not eligible:
        return None, "; ".join(declined) or "no eligible contraction in captured source"
    if compile_products is None:
        routing = provider.routing(directory, target=target, package=package, facts=facts, eligible=eligible)
    else:
        routing = provider.routing_candidate_products(
            directory, target=target, package=package, facts=facts, eligible=eligible, compile_products=compile_products
        )
    if routing is None:
        return None, "provider declines captured contraction semantics"
    if routing.device != target or Path(routing.package_dir).resolve() != package.resolve():
        raise ValueError("provider routing differs from selected target/package")
    if routing.select is not None or (
        routing.catalog_builder is None and routing.exact_selection is None and routing.region_offload is None
    ):
        raise ValueError("batch device mode requires source-bound catalog or exact selection")
    return routing, "provider selected source-bound contractions"


def source_route(directory: Path, report, plan, *, region_capable: bool) -> tuple[str | None, str]:
    """``(route, reason)`` the Phase 0 lane selects for one prepared bundle, from trusted inputs only.

    ``route`` is ``"region"`` (admitted exact non-contraction sites, for a provider with a region
    offload), ``"contraction"`` (the provider's catalog routing decides) or None (host, with the reason).
    ``plan`` is the bundle's spec_int8 numeric preparation (None for other policies). No provider or
    submitted code runs here, so a grader can decide the route before isolating that code.
    """
    if plan is None:
        return "contraction", "provider routing"
    # Borrowed outputs need source-bound movement plus an exact
    # result-layout copy-out, rather than a disjoint result alone.
    bundled = [case for case in report["cases"] if case.get("status") == "bundled"]
    views = [case.get("overload") for case in bundled if (case.get("case") or {}).get("output_input_aliases")]
    from merlin.targetgen.materialized_movement import alias_materialization_valid

    materialized_views = alias_materialization_valid(directory, report, plan)
    if plan["quantized_contractions"] or plan.get("native_contractions", 0):
        return "contraction", "provider routing"
    if (
        (plan.get("quantized_non_contraction_sites", 0) or plan.get("exact_dtype_sites", 0))
        and region_capable
        and (materialized_views or not views)
    ):
        # Admitted int8 non-contraction sites: the provider binds and lowers its own regions.
        return "region", "provider region offload of admitted exact source regions"
    if views and not materialized_views:
        return None, f"view result aliases its input ({views[0]}); host metadata, not device work"
    reason = "; ".join(dict.fromkeys(reason for site in plan["sites"] for reason in site["reasons"]))
    return None, reason or plan.get(
        "host_reason"
    ) or "no contraction in captured source; original host execution retained"


def bind_source_route(
    route, reason, directory: Path, target: str, device_package: Path, provider, facts, *, compile_products=None
):
    """``(routing, reason)``: the provider's DeviceRouting for a route :func:`source_route` selected."""
    if route == "region":
        if compile_products is not None:
            return None, "provider region offload does not consume candidate product transport"
        return region_routing_for_bundle(target, device_package, provider.core_aten_offload), reason
    if route == "contraction":
        options = {} if compile_products is None else {"compile_products": compile_products}
        return routing_for_bundle(directory, target, device_package, provider, facts, **options)
    return None, reason


def select_source_routing(
    directory: Path, target: str, device_package: Path, provider, facts, report, plan, *, compile_products=None
):
    """``(routing, reason)`` for one prepared bundle: the Phase 0 lane's own device-route decision."""
    region_capable = callable(getattr(provider, "core_aten_offload", None))
    route, reason = source_route(directory, report, plan, region_capable=region_capable)
    return bind_source_route(
        route, reason, directory, target, device_package, provider, facts, compile_products=compile_products
    )


def submission_interface(provider: Path) -> list[dict]:
    """Every submission member a Core ATen execution provider uses, read from its source (never run).

    A provider loads modules from the submitted package (``load_module(package, "<module>")``, directly
    or through a one-line helper that formats the module name). Returns one row per (module, member,
    provider hook): how many positional arguments and which keywords the provider passes, and how many
    values it unpacks from the result, or ``None`` when the member is read rather than called. This is
    the call ABI the candidate's package must implement; no provider logic is reported.
    """
    import ast

    tree = ast.parse(Path(provider).read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

    def loads(call):
        return isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "load_module"

    helpers = {}  # helper name -> (module prefix, index of the parameter that completes the name)
    for function in functions:
        params = [arg.arg for arg in function.args.args]
        for node in ast.walk(function):
            if loads(node) and len(node.args) > 1 and isinstance(node.args[1], ast.JoinedStr):
                parts = node.args[1].values
                if (
                    len(parts) == 2
                    and isinstance(parts[0], ast.Constant)
                    and isinstance(parts[1], ast.FormattedValue)
                    and isinstance(parts[1].value, ast.Name)
                    and parts[1].value.id in params
                ):
                    helpers[function.name] = (parts[0].value, params.index(parts[1].value.id))

    def module_of(call):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            return None
        if loads(call) and len(call.args) > 1 and isinstance(call.args[1], ast.Constant):
            return call.args[1].value
        prefix, index = helpers.get(call.func.id, (None, None))
        argument = call.args[index] if prefix is not None and len(call.args) > index else None
        return prefix + argument.value if isinstance(argument, ast.Constant) else None

    rows: dict[tuple, dict] = {}
    for function in functions:
        bound = {}
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                module = module_of(node.value)
                if module:
                    bound[node.targets[0].id] = module
        unpacked = {
            id(node.value.func): len(node.targets[0].elts)
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Tuple)
            and isinstance(node.value, ast.Call)
        }
        called = {id(node.func): node for node in ast.walk(function) if isinstance(node, ast.Call)}
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in bound
                and not node.attr.startswith("__")
            ):
                call = called.get(id(node))
                use = None
                if call is not None:
                    use = {
                        "positional": len(call.args),
                        "keywords": sorted(k.arg for k in call.keywords if k.arg),
                        "unpacks": unpacked.get(id(node)),
                    }
                key = (bound[node.value.id], node.attr, function.name)
                rows.setdefault(key, {"module": key[0], "member": key[1], "hook": key[2], "calls": [], "fields": []})
                if use is not None and use not in rows[key]["calls"]:
                    rows[key]["calls"].append(use)
        # Fields the provider reads from a member's result, and work-directory files it reads afterwards
        # (a path it only writes itself is not part of the submission's obligation).
        results, written, files = {}, set(), []
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                func = node.value.func
                targets = node.targets[0].elts if isinstance(node.targets[0], ast.Tuple) else node.targets
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in bound:
                    unpack = isinstance(node.targets[0], ast.Tuple)
                    for index, target in enumerate(targets):
                        if isinstance(target, ast.Name) and target.id != "_":
                            source = ((bound[func.value.id], func.attr, function.name), index if unpack else None)
                            results.setdefault(target.id, []).append((node.lineno, source))
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"write_text", "write_bytes"}
                and isinstance(node.func.value, ast.Name)
            ):
                written.add(node.func.value.id)
        named = {
            node.targets[0].id: node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.BinOp)
        }
        for node in ast.walk(function):
            field = None
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and isinstance(node.slice, ast.Constant)
            ):
                owner, field = node.value.id, node.slice.value
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                owner, field = node.func.value.id, node.args[0].value
            # A field belongs to the latest binding of its name before the read.
            bindings = [b for b in results.get(owner, ()) if b[0] <= node.lineno] if isinstance(field, str) else []
            if bindings:
                key, index = max(bindings, key=lambda b: b[0])[1]
                label = field if index is None else f"[{index}]{field}"
                if label not in rows[key]["fields"]:
                    rows[key]["fields"].append(label)
            if (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Div)
                and isinstance(node.left, ast.Name)
                and isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, str)
                and "." in node.right.value
                and not any(value is node and name in written for name, value in named.items())
                and node.right.value not in files
            ):
                files.append(node.right.value)
        for key, row in rows.items():
            if key[2] == function.name:
                row["reads"] = files
    # Report each use under the PUBLIC provider hooks that reach it: a private helper's name says how the
    # support is structured, not what the candidate implements. Uses no public hook reaches are dropped.
    names = {function.name for function in functions}
    callees = {
        function.name: {
            node.func.id
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names
        }
        for function in functions
    }

    def reach(name, seen):
        for callee in callees[name] - seen:
            seen.add(callee)
            reach(callee, seen)
        return seen

    public = {name: reach(name, {name}) for name in names if not name.startswith("_")}
    merged: dict[tuple, dict] = {}
    for (module, member, hook), row in sorted(rows.items()):
        if member.startswith("_"):
            continue
        for entry in sorted(name for name, reached in public.items() if hook in reached):
            out = merged.setdefault((module, member, entry), {**row, "hook": entry, "calls": [], "fields": []})
            out["calls"] += [call for call in row["calls"] if call not in out["calls"]]
            out["fields"] += [field for field in row["fields"] if field not in out["fields"]]
            out["reads"] = list(dict.fromkeys([*out.get("reads", []), *row.get("reads", [])]))
    return [merged[key] for key in sorted(merged)]


def region_routing_for_bundle(target: str, package: Path, region_offload):
    """A DeviceRouting that hands the prepared program to the provider's own region offload."""
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.device_build import DeviceRouting

    def offload(prepared: Path, workdir: Path):
        return region_offload(prepared, workdir, package=package, llvm_bin=toolchain.llvm_install() / "bin")

    return DeviceRouting(
        device=target, package_dir=package, operand_dtype="i8", accum_dtype="i32", region_offload=offload
    )


def executed_device_instructions(trace: Path, elf: Path, target: str) -> dict[str, Any]:
    """Count retired-log instruction words in executable ELF ranges; static presence is insufficient."""
    from merlin.perf.isa_prohibition import MAJOR_OPCODE_MASK
    from merlin.targetgen.elf_lanes import accelerator_opcode, executable_sections

    opcode, source = accelerator_opcode(target)
    if opcode is None:
        raise ValueError(source)
    blob = elf.read_bytes()
    sections = executable_sections(blob)
    if not sections:
        raise ValueError("ELF has no executable sections")
    count = 0
    digest = hashlib.sha256()
    with trace.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            fields = raw.decode("utf-8", errors="replace").split()
            # Spike -l: core <hart>: <pc> (<instruction word>) <disassembly>.
            if len(fields) < 4 or fields[0] != "core" or not fields[1].endswith(":"):
                continue
            if not fields[3].startswith("(0x") or not fields[3].endswith(")"):
                continue
            try:
                pc, word = int(fields[2], 16), int(fields[3][1:-1], 16)
            except ValueError:
                continue
            if word & MAJOR_OPCODE_MASK != opcode:
                continue
            for _, offset, size, address in sections:
                if address <= pc and pc + 4 <= address + size:
                    at = offset + pc - address
                    # Match the actual instruction at the logged PC, not arbitrary trace text.
                    byteorder = "little" if blob[5] == 1 else "big"
                    if int.from_bytes(blob[at : at + 4], byteorder) != word:
                        raise ValueError("trace word differs from linked executable at logged PC")
                    count += 1
                    break
    facts_path = ensure_facts(target)
    return {
        "status": "measured",
        "facts_path": str(facts_path.resolve()),
        "facts_sha256": hashlib.sha256(facts_path.read_bytes()).hexdigest(),
        "elf": str(elf.resolve()),
        "facts_provenance": {key: value for key, value in load_facts(target).items() if key != "facts"},
        "target": target,
        "opcode": opcode,
        "opcode_source": source,
        "executed_instructions": count,
        "trace": str(trace),
        "trace_sha256": digest.hexdigest(),
        "elf_sha256": hashlib.sha256(blob).hexdigest(),
        "attribution": "single_case_shard",
    }


def verify_execution_evidence(evidence: dict) -> bool:
    """Regrade retained evidence, refusing deleted/changed traces, executables or fact pins."""
    try:
        facts_path = Path(evidence["facts_path"])
        if hashlib.sha256(facts_path.read_bytes()).hexdigest() != evidence["facts_sha256"]:
            return False
        with selected_facts(evidence["target"], facts_path):
            actual = executed_device_instructions(Path(evidence["trace"]), Path(evidence["elf"]), evidence["target"])
        return (
            all(
                actual[key] == evidence.get(key)
                for key in ("opcode", "trace_sha256", "elf_sha256", "executed_instructions", "attribution")
            )
            and actual["executed_instructions"] > 0
        )
    except (KeyError, OSError, ValueError):
        return False


def _write_json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_spike_bundle(
    directory: Path,
    report: dict,
    *,
    arena_mb: int,
    timeout: int,
    target: str | None = None,
    device_package: Path | None = None,
    execution_provider: Path | None = None,
    rtl_facts: Path | None = None,
    numeric_policy: str = "fp32",
    software_spec: Path | None = None,
    host_capabilities: Path | None = None,
    dram_bytes: int | None = None,
) -> tuple[dict, dict]:
    from merlin.llvmlower.contraction_quantization import POLICIES
    from merlin.llvmlower.device_offload import load_sidecar
    from merlin.runtime.backends import spike_model
    from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch

    if numeric_policy not in POLICIES:
        raise ValueError(f"unknown numeric policy {numeric_policy!r}")
    report["numeric_policy"] = numeric_policy
    for case in report["cases"]:
        case["numeric_policy"] = numeric_policy
        case["scales"] = []
    output_bytes = None
    output_shapes = None
    semantic_readback = None
    options = {}
    stage = "routing"
    selected_provider = None
    try:
        bundled = [case for case in report["cases"] if case["status"] == "bundled"]
        if target:
            for case in bundled:
                case["routing"] = {
                    "lane": "host",
                    "routed": False,
                    "reason": "device routing not completed",
                    "target": target,
                }
        if target and (device_package is None or len(bundled) != 1):
            raise ValueError("device execution requires a package and exactly one case per shard")
        if numeric_policy != "fp32" and not target:
            raise ValueError("integer numeric policies require a selected target capability manifest")
        with selected_facts(target, rtl_facts) if target else nullcontext(None) as facts:
            routing, reason, options = None, "scalar mode", {}
            if target:
                provider = load_execution_provider(target, execution_provider)
                selected_provider = Path(provider.__file__)
                if numeric_policy == "spec_int8":
                    stage = "numeric_preparation"
                    from merlin.targetgen.core_aten_numeric import prepare_spec_int8

                    spec_path = software_spec or os.environ.get("MERLIN_CORE_ATEN_SOFTWARE_SPEC")
                    host_path = host_capabilities or os.environ.get("MERLIN_CORE_ATEN_HOST_CAPABILITIES")
                    if not spec_path or not host_path:
                        raise ValueError("spec_int8 requires explicit software-spec and host-capabilities selection")
                    report["numeric_preparation"] = prepare_spec_int8(
                        directory,
                        report,
                        target=target,
                        facts=facts,
                        software_spec=Path(spec_path),
                        host_capabilities=Path(host_path),
                        provider=provider,
                        package=device_package,
                    )
                if numeric_policy == "int8_w8a8":
                    stage = "numeric_preparation"
                    from merlin.system.offload import device_contraction_ranks, device_dtype_triples
                    from merlin.targetgen.core_aten_numeric import prepare_w8a8

                    report["numeric_preparation"] = prepare_w8a8(
                        directory, report, triples=device_dtype_triples(target), ranks=device_contraction_ranks(target)
                    )
                stage = "routing"
                provider = load_execution_provider(target, execution_provider)
                selected_provider = Path(provider.__file__)
                prepare = getattr(provider, "prepare_bundle", None)
                if prepare is not None and numeric_policy != "spec_int8":
                    preparation = prepare(directory, target=target, facts=facts)
                    report["source_preparation"] = preparation
                    bundled[0]["source_preparation"] = preparation
                routing, reason = select_source_routing(
                    directory,
                    target,
                    device_package,
                    provider,
                    facts,
                    report,
                    report["numeric_preparation"] if numeric_policy == "spec_int8" else None,
                )
                if numeric_policy == "int8_w8a8" and report.get("numeric_preparation", {}).get(
                    "quantized_contractions"
                ):
                    from dataclasses import replace

                    from merlin.targetgen.core_aten_numeric import checked_w8a8_builder

                    if routing is None or routing.catalog_builder is None:
                        raise ValueError("W8A8 requires source-bound provider primitive dispatch evidence")
                    routing = replace(
                        routing,
                        catalog_builder=checked_w8a8_builder(
                            routing.catalog_builder,
                            target=target,
                            package=device_package,
                            directory=directory,
                            report=report,
                        ),
                    )
                options = provider.runner_options()
            for case in bundled:
                case["routing"] = {
                    "lane": "device" if routing is not None else "host",
                    "routed": False,
                    "reason": reason,
                    "target": target,
                }
            stage = "build"
            from merlin.targetgen.core_aten_components import require_component_identity

            require_component_identity(directory, report.get("numeric_preparation", {}))
            build = spike_model.build(
                directory,
                directory / "spike-build",
                inputs_npz=directory / "inputs.npz",
                arena_mb=arena_mb,
                dump_all_outputs=True,
                backend="scalar",
                device=routing,
                # A packed map inside the selected DRAM (an RTL simulator's), instead of the functional
                # simulator's sparse high-address map, so the same ELF can run on the RTL.
                **({"dram_bytes": dram_bytes} if dram_bytes is not None else {}),
            )
            from merlin.llvmlower.device_build import load_region_offload

            sidecar = (
                (load_region_offload(directory / "spike-build") or load_sidecar(directory / "spike-build"))
                if routing is not None
                else {}
            )
            if routing is not None and not sidecar.get("routed"):
                raise RuntimeError("selected device routing emitted no source-bound calls")
            if routing is not None:
                bundled[0]["routing"].update(routed=True, routed_operations=sidecar["routed"])
            trace = directory / "spike-trace.log" if routing is not None else None
            stage = "execution"
            require_component_identity(directory, report.get("numeric_preparation", {}))
            run = spike_model.run(
                build["elf"],
                mem_bytes=build["mem_bytes"],
                timeout=timeout,
                vlen=build.get("vlen"),
                trace_path=trace,
                **options,
            )
            if build.get("build_hash") is not None and run.get("metrics", {}).get("build_hash") != build["build_hash"]:
                raise RuntimeError("console build hash does not match the linked executable")
            if run.get("metrics", {}).get("memref_rank_mismatch", 0) != 0:
                raise RuntimeError("runtime refused a memref copy with mismatched ranks")
            if routing is not None:
                evidence = executed_device_instructions(trace, Path(build["elf"]), target)
                evidence["case_id"] = str(bundled[0].get("case_id") or bundled[0]["overload"])
                bundled[0]["routing"] = {
                    "lane": "device",
                    "routed": True,
                    "reason": reason,
                    "target": target,
                    "routed_operations": sidecar["routed"],
                    "execution_evidence": evidence,
                }
                if evidence["executed_instructions"] < 1:
                    raise RuntimeError("device calls routed but no target instruction executed")
        output_bytes = run["output_bytes"]
        if len(output_bytes) != report["output_count"]:
            raise RuntimeError(f"hardware emitted {len(output_bytes)} outputs; bundle expects {report['output_count']}")
        (directory / "spike-console.txt").write_text(run["console"], encoding="utf-8")
        semantic_readback = run.get("semantic_readback")
        if semantic_readback is not None:
            _write_json(directory / "spike-semantic-readback.json", semantic_readback)
        output_shapes = run.get("output_shapes") or None
        _write_json(directory / "spike-output-shapes.json", output_shapes)
        _write_json(directory / "spike-output-bytes.json", [item.hex() for item in output_bytes])
        execution = {
            "status": "ran",
            "elf": str(build["elf"]),
            "build_hash": build.get("build_hash"),
            "output_count": len(output_bytes),
        }
    except Exception as exc:  # each stage failure remains an explicit artifact
        execution = {"status": "failed", "stage": stage, "error": f"{type(exc).__name__}: {exc}"}
        for case in report["cases"]:
            if case["status"] == "bundled":
                case["routing"]["failure_reason"] = execution["error"]
    from merlin.targetgen.core_aten_provenance import batch_provenance

    report["provenance"] = batch_provenance(
        directory,
        target=target,
        package=device_package,
        runner_options=options,
        rtl_facts=rtl_facts,
        execution_provider=selected_provider,
    )
    execution["provenance"] = report["provenance"]
    _write_json(directory / "core_aten_batch_map.json", report)
    _write_json(directory / "spike-execution.json", execution)
    verdict = grade_core_aten_batch(
        report,
        output_bytes,
        output_shapes=output_shapes,
        semantic_readback=semantic_readback,
        execution_error=execution.get("error") if execution["status"] != "ran" else None,
    )
    for case_verdict in verdict["cases"].values():
        observation = case_verdict.get("observation")
        if observation is not None:
            observation["evidence"].update(
                target=target,
                numeric_policy=numeric_policy,
                scales=case_verdict["scales"],
                lane=case_verdict["lane"],
                receipt=execution.get("build_hash"),
                elf=execution.get("elf"),
                artifact_digests=report["provenance"]["artifact_digests"],
            )
    _write_json(directory / "core_aten_batch_verdict.json", verdict)
    return execution, verdict
