"""Spike functional oracle for the Core ATen full-call capsule contract.

The plain host adapter executes submitted LLVM-dialect emission. Provider mode uses the
submitted source-bound device catalog with trusted host lowering. The bundle supplies the call boundary
(``semantic_io.json``, ``inputs.npz``); the return value carries lossless output bytes and the
semantic readback for the round-2 full-call grader.

The ISA, simulator options and optional extension come from the caller (the target contract or
the provider), not from this module, so any plain-ISA Spike target can share it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.targetgen import oracle_policy as _oracle_policy

DEFAULT_ARENA_MB = 256
SPEC_QUALIFICATION_SCHEMA = "merlin.spec_qualified_device.v1"
SPEC_QUALIFICATION_CHECKS = (
    "private_answers",
    "selected_policy",
    "admission",
    "source",
    "schedule",
    "partial_sums",
    "raw_i32",
    "ordered_f32",
    "full_call",
    "device_binding",
)
SPEC_EXECUTION_IDENTITY = ("case_sha256", "source_sha256", "elf_sha256", "trace_sha256")
# Python caches imported provider/dependency modules by namespace. Refuse changed
# owners in this process rather than executing old code against new source bytes.
_FULL_CALL_SOURCE_IDENTITIES: dict[Path, str] = {}


def unqualified_spec_device(reason: str, *, policy: str = "unknown") -> dict:
    """Answer-free absence of a selected-software-spec device qualification."""
    policy = (
        policy if isinstance(policy, str) and policy in {"unknown", "fp32", "spec_int8", "int8_w8a8"} else "unknown"
    )
    return dict(
        schema=SPEC_QUALIFICATION_SCHEMA,
        status="unqualified",
        policy=policy,
        scope="selected_software_spec",
        reasons=[reason],
        bindings={},
        checks=dict.fromkeys(SPEC_QUALIFICATION_CHECKS, False),
    )


def _digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def full_call_execution_identity(capsule: dict, bundle: Path, execution: dict) -> dict:
    """Join a reopened private commitment to the actual model and verified execution."""
    from merlin.targetgen.core_aten_capsules import load_private
    from merlin.targetgen.core_aten_device import verify_execution_evidence

    load_private(capsule)
    evidence = (execution.get("routing") or {}).get("execution_evidence")
    if not isinstance(evidence, dict) or not verify_execution_evidence(evidence):
        return {}
    top = execution.get("execution_evidence")
    if top is not None and (
        not isinstance(top, dict) or any(top.get(k) != evidence.get(k) for k in ("elf_sha256", "trace_sha256"))
    ):
        return {}
    identity = dict(
        case_sha256=capsule["full_call_golden"]["golden.yaml"],
        source_sha256=hashlib.sha256((bundle / "model.mlir").read_bytes()).hexdigest(),
        elf_sha256=evidence.get("elf_sha256"),
        trace_sha256=evidence.get("trace_sha256"),
    )
    return identity if all(_digest(v) for v in identity.values()) else {}


class _Unqualified(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise _Unqualified(reason)


def _join_catalog_source(bundle: Path, catalog_digest: str, prepared_digest: str) -> None:
    """Reproduce the ordinary trusted normalization before joining catalog bytes.

    A device catalog sees the normalized source, whereas the capsule commits
    the prepared frontend source. Different digests require complete byte
    equality with this fixed trusted normalization, never a candidate claim.
    """
    if catalog_digest == prepared_digest:
        return
    from merlin.runtime.backends.zephyr_model import _prepare_model_mlir

    signatures = json.loads((bundle / "spike-build/device_signatures.json").read_text())
    catalog_path = Path(signatures["catalog_manifest"])
    build = (bundle / "spike-build").resolve()
    _require(
        catalog_path.is_file() and catalog_path == catalog_path.resolve() and catalog_path.is_relative_to(build),
        "catalog_source_owner_invalid",
    )
    catalog = json.loads(catalog_path.read_text())
    source = Path(catalog["source_snapshot"])
    _require(
        source.is_file() and source == source.resolve() and source.is_relative_to(build),
        "catalog_source_owner_invalid",
    )
    _require(hashlib.sha256(source.read_bytes()).hexdigest() == catalog_digest, "catalog_source_identity_changed")
    work = Path(tempfile.mkdtemp(prefix="spec-source-normalization-", dir=bundle))
    expected = _prepare_model_mlir(bundle / "model.mlir", work, bundle_dir=bundle)
    _require(expected.read_bytes() == source.read_bytes(), "catalog_source_normalization_differs")


def _private_numeric_boundary(capsule, bundle, execution, verdict, selection):
    """Reuse private grading and captured admission; verify nonvacuous observation links."""
    from xdsl.dialects.builtin import FunctionType, StringAttr

    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.kernels.shapes import observe_contractions
    from merlin.llvmlower.result_layout import ATTRIBUTE
    from merlin.targetgen import core_aten_capsules as C
    from merlin.targetgen.core_aten_spec_int8 import admission_plan, host_readout_chain, source_declaration_continuity

    golden = C.load_private(capsule)
    report = golden["batch_map"]
    _require(len(report["cases"]) == 1, "single_case_required")
    record = report["cases"][0]
    plan = report.get("numeric_preparation")
    _require(isinstance(plan, dict) and plan.get("status") == "admitted", "private_admission_missing")
    _require(
        record.get("numeric_admission") == {k: v for k, v in plan.items() if k != "selection"},
        "private_admission_records_differ",
    )
    _require(record.get("numeric_policy") == "spec_int8", "private_policy_not_spec_int8")
    _require(
        not plan.get("host_int8_narrowing") and not record.get("numeric_admission", {}).get("host_int8_narrowing"),
        "native_narrowing_undeclared",
    )
    count = plan.get("quantized_contractions")
    _require(
        type(count) is int and count > 0 and not plan.get("native_contractions"), "prepared_quantized_rows_required"
    )
    sites = plan.get("sites")
    _require(
        isinstance(sites, list) and len(sites) == count and all(s.get("status") == "admitted" for s in sites),
        "private_site_roster_invalid",
    )
    _require(plan.get("quantization_sites"), "private_quantizer_roster_empty")
    _require(json.loads((bundle / "spec-int8-admission.json").read_text()) == plan, "runtime_admission_changed")
    if selection["spec"] != plan["selection"]["spec"] or selection["host"] != plan["selection"]["host"]:
        # Retain diagnostic inputs only in the operator-private runtime.
        (bundle / "selected-contract-recheck.json").write_text(
            json.dumps({"actual": selection, "frozen": plan["selection"]}, indent=2) + "\n"
        )
        raise _Unqualified("selected_contract_changed")
    reference = plan.get("integer_reference", {})
    expected = reference.get("expected_output_bytes")
    _require(
        reference.get("status") == "prepared" and isinstance(expected, dict) and expected, "private_reference_empty"
    )
    _require(len(verdict["cases"]) == 1, "graded_case_roster_invalid")
    row = next(iter(verdict["cases"].values()))
    _require(
        row.get("status") == "pass" and row.get("semantic_scope") in {"full", "values+mutation+observed-metadata"},
        "full_call_did_not_pass",
    )
    _require(
        row.get("numeric_execution_policy") == "spec_int8" and row.get("integer_reference", {}).get("status") == "pass",
        "exact_spec_reference_did_not_pass",
    )
    comparisons = row["integer_reference"].get("results")
    _require(
        isinstance(comparisons, list) and comparisons and all(r.get("status") == "pass" for r in comparisons),
        "exact_comparisons_empty",
    )
    _require({str(r["output_index"]) for r in comparisons} == set(expected), "exact_comparison_roster_differs")
    source = (Path(capsule["__dir__"]) / "capsule.interface.mlir").read_text()
    bound = C.caller_input_storage(source, report["cases"])
    bound = C._legacy_view_storage(capsule, golden, source, bound)
    _require((bundle / "model.mlir").read_text() == bound, "runtime_source_changed")
    module = parse_mlir_file(bundle / "model.mlir")
    function = next(op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
    returned = next(op for op in function.body.block.ops if op.name == "func.return")
    outputs = tuple(returned.operands)
    _require(len(outputs) == report["output_count"] == len(execution["output_bytes"]), "output_roster_differs")
    indices = []
    for site in sites:
        for key, dtype in (("accumulator_output_index", "i32"), ("readout_output_index", "f32")):
            index = site.get(key)
            _require(type(index) is int and 0 <= index < len(outputs), "observation_index_invalid")
            _require(
                str(outputs[index].type.element_type) == dtype and str(index) in expected,
                "observation_precision_or_reference_missing",
            )
            _require(
                isinstance(expected[str(index)], str) and bool(expected[str(index)]), "observation_reference_empty"
            )
            indices.append(index)
    _require(len(set(indices)) == 2 * count, "observation_indices_ambiguous")
    # Additional grading observations are removed only from this private parsed copy.
    # The actual emitted module and its immutable public commitment remain untouched.
    returned.operands = [v for i, v in enumerate(outputs) if i not in indices]
    function.properties["function_type"] = FunctionType.from_lists(
        [arg.type for arg in function.body.block.args], [v.type for v in returned.operands]
    )
    if ATTRIBUTE in module.attributes:
        layout = json.loads(module.attributes[ATTRIBUTE].data)
        layout["results"] = [v for i, v in enumerate(layout["results"]) if i not in indices]
        module.attributes[ATTRIBUTE] = StringAttr(json.dumps(layout, sort_keys=True))
    module.verify()
    fresh = admission_plan(module, selection, captured=True)
    _require(
        len(fresh["sites"]) == count and all(s["status"] == "admitted" for s in fresh["sites"]),
        "captured_admission_did_not_pass",
    )
    _require(
        fresh["quantization_sites"] and len(fresh["quantization_sites"]) == len(plan["quantization_sites"]),
        "captured_quantizer_roster_differs",
    )
    initial_steps = plan.get("source_declaration_continuity", {}).get("initial_source_admissions")
    _require(isinstance(initial_steps, list) and initial_steps, "source_continuity_missing")
    continuity = source_declaration_continuity({"sites": [{"steps": initial_steps}]}, fresh)
    _require(continuity["status"] == "preserved", "exact_source_obligation_lost")
    contractions = list(observe_contractions(module))
    for site, (operation, _) in zip(sites, contractions, strict=True):
        _, restored, _ = host_readout_chain(operation, with_bridge=True)
        _require(
            outputs[site["accumulator_output_index"]] is operation.results[0]
            and outputs[site["readout_output_index"]] is restored,
            "observation_not_linked_to_mac",
        )
    return plan


def _qualify_full_call(
    capsule, *, bundle, execution, verdict, package_dir, target, execution_provider, rtl_facts, timeout
):
    """Issue credit from trusted private grading and independently checked actual artifacts."""
    flow = capsule.get("operation", {}).get("attributes", {}).get("numeric_flow")
    policy = flow if isinstance(flow, str) else "unknown"
    receipt = unqualified_spec_device("qualification_not_run", policy=policy)
    try:
        from merlin.llvmlower import toolchain
        from merlin.targetgen.core_aten_capsules import load_private
        from merlin.targetgen.core_aten_device import load_execution_provider, selected_facts
        from merlin.targetgen.core_aten_spec_int8 import select_spec

        _require(policy == "spec_int8", "explicit_prepared_spec_int8_required")
        _require(
            execution.get("execution_error") is None and execution.get("lane") == "device", "verified_device_required"
        )
        identity = full_call_execution_identity(capsule, bundle, execution)
        _require(set(identity) == set(SPEC_EXECUTION_IDENTITY), "execution_identity_missing")
        _require(execution_provider is not None and package_dir is not None, "trusted_provider_required")
        _require(
            execution.get("provenance", {}).get("execution_provider_sha256")
            == hashlib.sha256(execution_provider.read_bytes()).hexdigest(),
            "execution_provider_changed",
        )
        plan = load_private(capsule)["batch_map"].get("numeric_preparation", {})
        selected = plan.get("selection", {})
        identities = selected.get("identities", {})
        paths = {}
        for key in ("software_spec", "host_capabilities"):
            record = identities.get(key, {})
            path = Path(record["path"])
            _require(
                _digest(record.get("sha256")) and hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"],
                "selected_contract_binding_changed",
            )
            paths[key] = path
        provider = load_execution_provider(target, execution_provider)
        verifier = getattr(provider, "verify_numeric_execution_artifacts", None)
        _require(callable(verifier), "actual_numeric_artifact_verifier_unavailable")
        with selected_facts(target, rtl_facts) as facts:
            selection = select_spec(target, facts, paths["software_spec"], paths["host_capabilities"])
            _private_numeric_boundary(capsule, bundle, execution, verdict, selection)
            proof = verifier(
                bundle,
                target=target,
                facts=facts,
                numeric_policy=selection["spec"],
                llvm_bin=toolchain.llvm_install() / "bin",
                workdir=Path(tempfile.mkdtemp(prefix="spec-qualification-", dir=bundle)) / "proof",
                execution_evidence=execution["routing"]["execution_evidence"],
                timeout=timeout,
            )
        _require(
            isinstance(proof, dict)
            and proof.get("schema") == "merlin.numeric_execution_artifacts.v1"
            and proof.get("status") == "verified",
            "actual_numeric_artifacts_not_verified",
        )
        bindings = proof.get("bindings", {})
        keys = {"source_sha256", "catalog_sha256", "target_ir_sha256", "object_sha256", "elf_sha256", "trace_sha256"}
        _require(set(bindings) == keys and all(_digest(v) for v in bindings.values()), "artifact_bindings_invalid")
        _join_catalog_source(bundle, bindings["source_sha256"], identity["source_sha256"])
        _require(
            all(bindings[k] == identity[k] for k in ("elf_sha256", "trace_sha256")),
            "artifact_execution_identity_differs",
        )
        _require(
            set(proof.get("checks", {})) == {"schedule", "partial_sums", "device_binding"}
            and all(v is True for v in proof["checks"].values()),
            "artifact_checks_not_verified",
        )
        receipt.update(
            status="qualified",
            reasons=[],
            bindings={
                **bindings,
                "source_sha256": identity["source_sha256"],
                "case_sha256": identity["case_sha256"],
                "software_spec_sha256": identities["software_spec"]["sha256"],
                "host_capabilities_sha256": identities["host_capabilities"]["sha256"],
            },
            checks=dict.fromkeys(SPEC_QUALIFICATION_CHECKS, True),
        )
    except _Unqualified as exc:
        receipt["reasons"] = [str(exc)]
    except Exception:  # noqa: BLE001 -- missing/malformed qualification never earns device credit
        receipt["reasons"] = ["qualification_evidence_invalid"]
    return receipt


def _translate_candidate(llvm_mlir: str):
    """A ``host_llvm_transform`` that returns the candidate, translated to LLVM IR."""

    def transform(_trusted: Path, workdir: Path) -> Path:
        from merlin.llvmlower import toolchain

        source = Path(workdir) / "candidate.mlir"
        source.write_text(llvm_mlir, encoding="utf-8")
        out = Path(workdir) / "candidate.ll"
        done = subprocess.run(
            [str(toolchain.mlir_translate()), "--mlir-to-llvmir", str(source), "-o", str(out)],
            capture_output=True,
            text=True,
        )
        if done.returncode != 0 or not out.is_file():
            raise ValueError(f"candidate emission is not translatable LLVM-dialect MLIR: {done.stderr[-800:]}")
        return out

    return transform


def full_call_adapter(
    target: str,
    *,
    isa: str | None,
    backend: str = "scalar",
    runner_options: Mapping[str, Any] | None = None,
    engine: str = "spike",
    execution_provider: Path | None = None,
    rtl_facts: Path | None = None,
    candidate_transport=None,
    board: Mapping[str, int] | None = None,
):
    """Build an oracle callable exposing ``run_full_call`` for one plain Spike target.

    ``isa`` is the Spike ISA string (derived from the target contract by the caller), or None to read
    it from each linked image (:func:`spike_model.declared_isa`).
    ``runner_options`` may select ``spike_binary``, ``extension``, ``extlib`` and ``path_prepend``.

    With ``board`` (``dram_base``/``dram_bytes``/``code_reserve`` of an elaborated-RTL board), ``engine``
    names that board's RTL simulator: the image is linked inside the board's DRAM, Spike traces it for
    device evidence, and the SAME bytes then run on the RTL, whose console is what is graded.
    """
    if board is not None and engine == "spike":
        raise ValueError("a board-linked full-call oracle needs an elaborated-RTL engine")
    options = dict(runner_options or {})
    if candidate_transport is not None:
        from merlin_experiments.phase1.candidate_products import CandidateProductTransport

        if type(candidate_transport) is not CandidateProductTransport:
            raise TypeError("full-call compilation requires the selected candidate product transport")

    def run_full_call(
        *,
        bundle,
        llvm_mlir,
        command_buffer=None,
        target=target,
        timeout=600,
        package_dir=None,
        capsule=None,
        **_ignored,
    ):
        from merlin.runtime.backends import spike_model
        from merlin.targetgen.core_aten_provenance import batch_provenance

        bundle = Path(bundle)
        result: dict[str, Any] = {"engine": engine, "derived_from_rtl": False}
        build_dir = bundle / ("spike-build" if board is None else f"{engine}-build")
        trace_name = "spike-trace.log" if board is None else f"{engine}-spike-trace.log"
        layout = {key: int(value) for key, value in (board or {}).items() if value is not None}
        # On a board the arena must share the DRAM with the image: half of what the code reserve leaves.
        free_mb = (layout.get("dram_bytes", 0) - layout.get("code_reserve", 64 << 20)) >> 21
        arena_mb = DEFAULT_ARENA_MB if board is None else max(1, min(DEFAULT_ARENA_MB, free_mb))
        routing = routed = None
        result.update(lane="host", executed_instructions=0)
        numeric_flow = (capsule or {}).get("operation", {}).get("attributes", {}).get("numeric_flow")
        # Submitted code (the provider's routing imports it) never runs in this process: a capsule that
        # may route to the device is built in the isolated sandbox unless a candidate product transport
        # already isolates the submitted compiler.
        isolated = (
            execution_provider is not None
            and candidate_transport is None
            and ((capsule or {}).get("scored", True) or (capsule or {}).get("lane_expectation") == "any")
        )
        try:
            if (capsule or {}).get("lane_expectation") == "any" and (capsule or {}).get("scored") is False:
                raise ValueError("optional-device capsules must retain scored full-call correctness")
            if isolated:
                build, routed, reason = _isolated_build(
                    target=target,
                    bundle=bundle,
                    build_dir=build_dir,
                    capsule=capsule or {},
                    package_dir=package_dir,
                    execution_provider=execution_provider,
                    rtl_facts=rtl_facts,
                    request=dict(backend=backend, arena_mb=arena_mb, layout=layout, prepare=not numeric_flow),
                    timeout=timeout,
                )
                routing = routed if routed is not None else None
                if routing is None:
                    result["routing"] = {"lane": "host", "routed": False, "reason": reason}
            elif execution_provider is not None:
                from merlin.targetgen.core_aten_device import (
                    load_execution_provider,
                    select_source_routing,
                    selected_facts,
                )

                provider = load_execution_provider(target, execution_provider)
                with selected_facts(target, rtl_facts) as facts:
                    prepare = getattr(provider, "prepare_bundle", None)
                    if prepare is not None and not numeric_flow:
                        prepare(bundle, target=target, facts=facts)
                    lane = (capsule or {}).get("lane_expectation")
                    compile_products = None
                    if candidate_transport is not None:
                        from merlin_experiments.phase1.candidate_products import ProductSpec

                        tools = candidate_transport.tool_substitutions()

                        def compile_products(prepared, destination, *, inventory_only):
                            names = provider.candidate_catalog_products(inventory_only=inventory_only)
                            return candidate_transport.compile(
                                Path(package_dir),
                                Path(prepared),
                                argv=provider.candidate_catalog_command(
                                    python=tools["python"], inventory_only=inventory_only
                                ),
                                products=tuple(ProductSpec(name, 32 * 1024 * 1024) for name in names),
                                destination=Path(destination),
                                timeout_s=timeout,
                                substitutions=tools,
                                private_paths=(bundle,),
                            )

                    if lane == "any":
                        validate = getattr(
                            provider,
                            "validate_package_candidate_products"
                            if compile_products is not None
                            else "validate_package",
                            None,
                        )
                        if not callable(validate) or package_dir is None:
                            raise ValueError(
                                "optional-device fallback requires independent provider package validation"
                            )
                        validation = dict(target=target, package=Path(package_dir), facts=facts)
                        if compile_products is not None:
                            validation["compile_products"] = compile_products
                        if validate(bundle, **validation) is not None:
                            raise ValueError("provider package validation must return None or raise")
                    if (capsule or {}).get("scored", True) or lane == "any":
                        if package_dir is None:
                            raise ValueError("source-bound execution requires the submitted package")
                        from merlin.targetgen.core_aten_capsules import load_private

                        # The Phase 0 lane's own route: admitted exact non-contraction sites go to the
                        # provider's region offload, contractions to its catalog.
                        report = load_private(capsule)["batch_map"]
                        plan = (
                            report.get("numeric_preparation") if report.get("numeric_policy") == "spec_int8" else None
                        )
                        routing, reason = select_source_routing(
                            bundle,
                            target,
                            Path(package_dir),
                            provider,
                            facts,
                            report,
                            plan,
                            compile_products=compile_products,
                        )
                        # Only an ordinary structural/provider decline may fall
                        # back. Exceptions from source/catalog/binding checks
                        # remain execution errors; never catch and mask them.
                        if routing is None and lane != "any":
                            raise ValueError("submitted backend did not route this capsule: " + reason)
                        if routing is None:
                            result["routing"] = {"lane": "host", "routed": False, "reason": reason}
            if not isolated:
                build = spike_model.build(
                    bundle,
                    build_dir,
                    inputs_npz=bundle / "inputs.npz",
                    arena_mb=arena_mb,
                    dump_all_outputs=True,
                    backend=backend,
                    device=routing,
                    host_llvm_transform=_translate_candidate(llvm_mlir) if execution_provider is None else None,
                    **layout,
                )
            spike_isa = isa or spike_model.declared_isa(build["elf"])
            if not spike_isa:
                raise ValueError("linked image declares no ISA for the functional simulator")
            trace = bundle / trace_name if routing is not None else None
            # The guest's HTIF syscalls reach the simulator's filesystem: run it jailed, never beside answers.
            private = [bundle, *([Path(capsule["__dir__"]).parents[1]] if (capsule or {}).get("__dir__") else [])]
            jail, image = _spike_jail(build["elf"], bundle.parent / f".oracle-{build_dir.name}", private, options)
            run = spike_model.run(
                image,
                command_prefix=jail,
                mem_bytes=build["mem_bytes"],
                timeout=timeout,
                vlen=build.get("vlen"),
                isa=spike_isa,
                trace_path=trace,
                **options,
            )
            if build.get("build_hash") is not None and run.get("metrics", {}).get("build_hash") != build["build_hash"]:
                raise RuntimeError("console build hash does not match the linked executable")
            if routing is not None:
                from merlin.llvmlower.device_build import load_region_offload
                from merlin.llvmlower.device_offload import load_sidecar
                from merlin.targetgen.core_aten_device import executed_device_instructions, selected_facts

                # A provider routes through a contraction catalog or a region offload; each build
                # records its bindings in its own sidecar.
                if not isolated:
                    routed = (load_region_offload(build_dir) or load_sidecar(build_dir)).get("routed")
                if not routed:
                    raise ValueError("submitted backend emitted no source-bound device calls")
                with selected_facts(target, rtl_facts):
                    evidence = executed_device_instructions(trace, Path(build["elf"]), target)
                result.update(
                    lane="device", executed_instructions=evidence["executed_instructions"], execution_evidence=evidence
                )
                from merlin.targetgen.core_aten_capsules import load_private

                record = load_private(capsule)["batch_map"]["cases"][0]
                evidence["case_id"] = str(record.get("case_id") or record["overload"])
                result["routing"] = dict(
                    lane="device",
                    routed=True,
                    target=target,
                    routed_operations=routed,
                    execution_evidence=evidence,
                )
                if evidence["executed_instructions"] < 1:
                    raise ValueError("no target instruction executed for submitted device calls")
            (bundle / "spike-console.txt" if board is None else build_dir / "spike-console.txt").write_text(
                run["console"]
            )
            if board is not None:
                work = bundle.parent / f".oracle-{engine}-run"
                run = _run_on_rtl(target, engine, build, timeout, result, work=work, private=private)
                (bundle / f"{engine}-console.txt").write_text(run["console"])
            result.update(
                output_bytes=run["output_bytes"],
                output_shapes=run.get("output_shapes") or None,
                semantic_readback=run.get("semantic_readback"),
                execution_error=None,
            )
        except Exception as exc:  # noqa: BLE001 -- execution refusal is a verdict input
            from merlin.targetgen.full_call_build import sanitized

            # Simulator errors carry the guest's console (candidate output); feedback gets one path-free line.
            result.update(
                output_bytes=None,
                output_shapes=None,
                semantic_readback=None,
                execution_error=sanitized(f"{type(exc).__name__}: {exc}"),
            )
        provenance = batch_provenance(
            bundle,
            target=target if execution_provider else None,
            package=Path(package_dir) if package_dir else None,
            runner_options=options,
            rtl_facts=rtl_facts,
        )
        provenance["isa"] = isa or "declared_by_linked_image"
        provenance["target"] = target
        if board is not None:
            provenance["rtl_engine"] = engine
            provenance["board_layout"] = layout
            provenance["simulator"] = result.pop("simulator", None)
        if execution_provider is None:
            provenance["candidate_llvm_sha256"] = hashlib.sha256(llvm_mlir.encode()).hexdigest()
        else:
            provenance["compilation_mode"] = "source_bound_submitted_device_catalog"
            provenance["execution_provider_sha256"] = hashlib.sha256(execution_provider.read_bytes()).hexdigest()
        result["provenance"] = provenance
        return result

    def run(cb, llvm_text, workdir, timeout):
        from merlin.targetgen.capsule_runner import OracleUnavailable

        raise OracleUnavailable("this oracle executes the full-call boundary only (run_full_call)")

    run.run_full_call = run_full_call

    def qualify_full_call(capsule, *, bundle, execution, verdict, package_dir, timeout):
        return _qualify_full_call(
            capsule,
            bundle=Path(bundle),
            execution=execution,
            verdict=verdict,
            package_dir=package_dir,
            target=target,
            execution_provider=execution_provider,
            rtl_facts=rtl_facts,
            timeout=timeout,
        )

    def rebound(**changes):
        selected = dict(isa=isa, backend=backend, runner_options=options, engine=engine, rtl_facts=rtl_facts)
        selected.update(execution_provider=execution_provider, candidate_transport=candidate_transport, board=board)
        return full_call_adapter(target, **{**selected, **changes})

    if board is None:  # qualification binds the functional tier's own image and trace
        run.qualify_full_call = qualify_full_call
    run.compiles_source_bundle = execution_provider is not None
    if candidate_transport is not None:
        run._merlin_candidate_transport = candidate_transport
    if execution_provider is not None:
        run._merlin_with_candidate_transport = lambda transport: rebound(candidate_transport=transport)
    run._merlin_on_board = lambda engine, board: rebound(engine=engine, board=board)
    run._merlin_simulator_engine = engine
    return run


def _isolated_build(
    *, target, bundle, build_dir, capsule, package_dir, execution_provider, rtl_facts, request, timeout
):
    """``(build, routed, reason)``: decide the route here from trusted inputs, build in the sandbox.

    The returned image is copied into the private ``build_dir`` before anything executes it, so the
    grader runs and grades exactly the bytes the sandbox produced and nothing the sandbox can still edit.
    """
    from merlin.targetgen.core_aten_capsules import load_private
    from merlin.targetgen.core_aten_device import load_execution_provider, source_route
    from merlin.targetgen.full_call_build import build_in_sandbox, sanitized, sanitized_record
    from merlin.targetgen.rtl.facts import ensure_facts

    if package_dir is None:
        raise ValueError("source-bound execution requires the submitted package")
    lane = capsule.get("lane_expectation")
    report = load_private(capsule)["batch_map"]
    plan = report.get("numeric_preparation") if report.get("numeric_policy") == "spec_int8" else None
    provider = load_execution_provider(target, execution_provider)  # support code only; no submitted module
    region_capable = callable(getattr(provider, "core_aten_offload", None))
    route, reason = source_route(Path(bundle), report, plan, region_capable=region_capable)
    work = Path(bundle).parent / f".isolated-{build_dir.name}"
    reported = build_in_sandbox(
        {**request, "target": target, "route": route, "reason": reason, "validate": lane == "any"}
        | {"require_device": lane != "any"},
        bundle=Path(bundle),
        work=work,
        package=Path(package_dir),
        provider=Path(execution_provider),
        facts=ensure_facts(target, explicit=rtl_facts),
        denied=(Path(capsule["__dir__"]).parents[1],) if capsule.get("__dir__") else (),
        timeout=timeout,
    )
    if reported.get("error"):
        raise ValueError("isolated build refused: " + sanitized(reported["error"]))
    produced = work / "out" / "build"
    elf = Path(reported["build"]["elf"])
    if not elf.is_relative_to(produced) or any(p.is_symlink() for p in produced.rglob("*")):
        raise ValueError("isolated build reported an image outside its own build directory")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    shutil.copytree(produced, build_dir)
    build = {**reported["build"], "elf": str(build_dir / elf.relative_to(produced))}
    routed = reported.get("routed")
    reason = sanitized(reported["reason"]) if reported.get("reason") else reason
    return build, (sanitized_record(routed) if routed is not None else None), reason


def _spike_jail(elf, work: Path, private, options: Mapping) -> tuple[list[str], Path]:
    """The simulator jail for Spike: its binary, extension library directory and tool path only."""
    from merlin.runtime.backends.spike import spike_path
    from merlin.targetgen.full_call_build import oracle_jail

    binary = Path(options.get("spike_binary") or spike_path())
    extension = [Path(options["extlib"]).parent] if options.get("extlib") else []
    path = [Path(p) for p in options.get("path_prepend") or ()]
    return oracle_jail(Path(elf), work, private=private, executables=[binary], roots=extension, path=path)


def _run_on_rtl(target: str, engine: str, build: Mapping, timeout, result: dict, *, work: Path, private) -> dict:
    """Execute a board-linked image on the selected elaborated-RTL engine; parse its console.

    The engine's identity is re-verified around the run (``run_verified``) and recorded in
    ``result["simulator"]``; the console must name the same build as the linked image.
    """
    from merlin.runtime.backends import base as backends
    from merlin.runtime.backends import spike_model
    from merlin.targetgen.capsule_runner import OracleUnavailable
    from merlin.targetgen.full_call_build import run_elf_jailed
    from merlin.targetgen.rtl_simulator_execution import SimulatorExecution, run_verified

    backend = backends.get_backend(target)
    if not backend.available(engine):
        raise OracleUnavailable(f"{engine} not available")
    observed = run_verified(
        SimulatorExecution(backend, engine, target, timeout, unavailable=OracleUnavailable),
        lambda remaining: {
            "console": run_elf_jailed(target, engine, Path(build["elf"]), work, private=private, timeout=remaining),
            "oracle": {},
        },
    )
    # Only a console the RTL actually produced makes this an RTL verdict; anything earlier is not run.
    result["derived_from_rtl"] = True
    run = spike_model.parse_console(observed["console"])
    if build.get("build_hash") is not None and run.get("metrics", {}).get("build_hash") != build["build_hash"]:
        raise RuntimeError(f"{engine} console build hash does not match the linked executable")
    result["simulator"] = observed["oracle"].get("provenance")
    result["cycles"] = run.get("metrics", {}).get("cycles")
    return run


def _contract_isa(target: str) -> str:
    """The Spike ISA string is a declared target fact (``isa.march``); never defaulted here."""
    from merlin.targetgen.target_experiment import load_capability_manifest

    isa = (load_capability_manifest(target).contract.get("isa") or {}).get("march")
    if not isinstance(isa, str) or not isa.strip():
        raise ValueError(f"target contract declares no isa.march for {target!r}")
    return isa.strip()


def sim_adapters(target: str) -> dict:
    """Oracle factory for a plain-ISA Spike target: the functional (L2) tier only."""
    return {"L2": full_call_adapter(target, isa=_contract_isa(target))}


def declared(target: str):
    """Load an explicitly selected support-owned complete-call oracle.

    Capability overlays cannot select executable code. Missing declarations keep
    the generic adapter; malformed declared factories refuse rather than fallback.
    """
    import inspect

    from merlin.common.source_membership import file_members
    from merlin.targetgen.plugins import load_module, resolve_support
    from merlin.targetgen.providers import support_source_dependencies
    from merlin.targetgen.target_registry import TargetContractMissing, resolve

    try:
        info = resolve(target)
        contract = info._load_provider_contract()
    except TargetContractMissing:
        return None
    if not isinstance(contract, dict):
        raise ValueError("full-call support contract is malformed")
    runner = contract.get("runner")
    if runner is None:
        return None
    if not isinstance(runner, dict):
        raise ValueError("full-call runner declaration is malformed")
    if "full_call_provider" not in runner:
        return None
    selected = resolve_support(target)
    if (
        selected.base.resolve() != info.base.resolve()
        or selected.contract_path.resolve() != info.contract_path.resolve()
    ):
        raise ValueError("full-call support selection changed during resolution")
    reference = runner["full_call_provider"]
    if type(reference) is not str or not reference or "\\" in reference:
        raise ValueError("full-call provider must name an ordinary owned Python file")
    root = selected.base.resolve()
    relative = Path(reference)
    provider_path = relative if relative.is_absolute() else root / relative
    if (
        ".." in relative.parts
        or provider_path.suffix != ".py"
        or not provider_path.is_file()
        or not provider_path.is_relative_to(root)
        or any(path.is_symlink() for path in (provider_path, *provider_path.parents))
        or provider_path.resolve(strict=True) != provider_path
    ):
        raise ValueError("full-call provider must name an ordinary contained Python file")
    owners = (root, *support_source_dependencies(root).values())

    def source_identities():
        try:
            return {
                owner: hashlib.sha256(
                    json.dumps(
                        [
                            (name, hashlib.sha256(path.read_bytes()).hexdigest())
                            for name, path in file_members(owner).items()
                        ]
                    ).encode()
                ).hexdigest()
                for owner in owners
            }
        except (OSError, ValueError):
            raise ValueError("full-call provider source inventory is unavailable") from None

    identities = source_identities()
    if any(_FULL_CALL_SOURCE_IDENTITIES.get(owner, digest) != digest for owner, digest in identities.items()):
        raise ValueError("full-call provider source changed after process-local loading")
    # Pin before importing: even a failed factory can leave imported helpers cached.
    _FULL_CALL_SOURCE_IDENTITIES.update(identities)

    def owned_function(value):
        if not inspect.isfunction(value):
            return False
        filename = inspect.getsourcefile(value)
        if filename is None:
            return False
        source = Path(filename)
        return (
            source.is_file()
            and source.resolve() == source
            and not any(path.is_symlink() for path in (source, *source.parents))
            and (source == Path(__file__).resolve() or any(source.is_relative_to(owner) for owner in owners))
        )

    namespace = "full_call_execution_" + hashlib.sha256((str(root) + ":" + str(provider_path)).encode()).hexdigest()
    try:
        provider = load_module(root, provider_path.relative_to(root).as_posix(), package_name=namespace)
        factory = getattr(provider, "full_call_oracle", None)
        if not owned_function(factory) or Path(inspect.getsourcefile(factory)).resolve() != provider_path:
            raise ValueError("invalid factory owner")
        adapter = factory(target)
    except Exception:  # noqa: BLE001 -- declared failures never become fallback or private diagnostic payloads
        raise ValueError("declared full-call oracle could not be constructed") from None
    if not owned_function(adapter) or not all(
        owned_function(getattr(adapter, name, None)) for name in ("run_full_call", "qualify_full_call")
    ):
        raise ValueError("declared full-call oracle lacks owned execution and qualification callbacks")
    if source_identities() != identities:
        raise ValueError("full-call provider source changed during factory construction")
    return adapter


def selected(target: str):
    """The operator-selected Core ATen execution provider as the full-call oracle, when no support
    contract declares one.

    ``MERLIN_CORE_ATEN_EXECUTION_PROVIDER`` already selects the provider that routes submitted device
    catalogs; it must be an ordinary file inside the selected support root. Its ``runner_options``
    selects the Spike extension; the Spike ISA is read from each linked image.
    """
    import os

    from merlin.targetgen.core_aten_device import load_execution_provider
    from merlin.targetgen.plugins import resolve_support
    from merlin.targetgen.rtl.facts import ensure_facts

    explicit = os.environ.get("MERLIN_CORE_ATEN_EXECUTION_PROVIDER")
    if not explicit:
        return None
    path = Path(explicit)
    root = resolve_support(target).base.resolve()
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("selected execution provider must be an ordinary file inside the selected support")
    options = getattr(load_execution_provider(target, path.resolve()), "runner_options", None)
    return full_call_adapter(
        target,
        isa=None,
        runner_options=options() if callable(options) else None,
        execution_provider=path.resolve(),
        rtl_facts=ensure_facts(target),
    )


def on_rtl_board(adapter, engine: str, target: str):
    """The full-call oracle ``adapter`` relinked for the descriptor's elaborated-RTL host board, or None.

    The board (``host: {board, catalog}``) supplies the DRAM the image must fit; a board that names no
    RTL simulator config, or a non-full-call adapter, yields None and the tier stays not-run.
    """
    rebind = getattr(adapter, "_merlin_on_board", None)
    if rebind is None:
        return None
    from merlin.runtime.boards import load_boards
    from merlin.targetgen.corpora import descriptor_path
    from merlin.targetgen.target_experiment import load_target_experiment

    experiment = load_target_experiment(descriptor_path(target))
    catalog = experiment.selected_board_catalog()
    board = load_boards(catalog).get(experiment.host_board) if catalog and experiment.host_board else None
    if board is None or board.target != target or not board.rtl_sim_config:
        return None
    return rebind(engine, dict(dram_base=board.dram_base, dram_bytes=board.dram_bytes, code_reserve=board.code_reserve))


def sim_available(target: str) -> tuple[bool, str]:
    from merlin.runtime.backends.spike import spike_path

    binary = Path(spike_path())
    if not binary.is_file():
        return False, f"{target!r}: spike binary is absent: {binary}"
    try:
        isa = _contract_isa(target)
    except Exception as exc:  # noqa: BLE001 -- unreadable contract means not gradeable
        return False, f"{target!r}: {exc}"
    return True, f"{target!r}: spike functional oracle available (isa {isa})"


# The module that owns the plain-ISA adapters binds them; late lookup keeps monkeypatches effective.
_oracle_policy.bind_sim_oracle_adapters("spike", lambda target: sim_adapters(target))
