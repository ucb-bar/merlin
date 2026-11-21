"""Protected original scale-emission resource gate; no numerical qualification."""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from pathlib import Path

from merlin.compiler.lower.device.selection import DeviceIRBudget
from merlin.development.backend.optimization.contracts import exact_tree_record, sha256_file
from merlin.development.targetgen.corpus import capsule_io as capsule_capsule_common
from merlin.development.targetgen.execution.host.common_transforms import CommonTransformService
from merlin.development.targetgen.execution.host.public_codegen_budget import (
    policy_from_profile,
    resources_complete,
)

# ---- public_scalability_gate -------------------------------------------------------------------


class PublicResourceGateError(RuntimeError):
    """The selected resource prerequisite failed before numerical grading."""


def _exception_record(error):
    """Protected diagnostics only; public feedback uses a fixed projection."""
    return {
        "type": type(error).__name__,
        "module": type(error).__module__,
        "message": str(error)[:16384],
        "traceback": "".join(traceback.format_exception(error))[-65536:],
    }


def require_complete(result):
    """Stop before expensive grading, without changing the selected policy."""
    if result.get("required") is False:
        return
    if (
        result.get("required") is True
        and result.get("resource_complete") is True
        and result.get("status") == "accepted"
    ):
        return
    raise PublicResourceGateError(
        "Required public codegen resource gate incomplete; numerical grading was not started. "
        "Inspect public_codegen_resources.json and the selected public --compile-only --scale-probe results."
    )


def selected_policy(owner):
    if type(owner) is not CommonTransformService:
        raise ValueError("public scalability policy requires its exact original profile owner")
    owner._verify()
    return policy_from_profile(owner.profile)


def selected_budget(owner):
    policy = selected_policy(owner)
    return DeviceIRBudget(**policy["device_ir_budget"]) if policy is not None else None


def run_gate(*, run_dir, candidate, context, capsules_root, contract, facts_path):
    """Run the complete selected public ladder under its archived profile owner."""
    from merlin.development.backend.feedback.execution.snapshot_evaluator import selected
    from merlin.experiments.corpus.source_staging import private_json

    if selected():
        result = {
            "status": "not_selected",
            "required": False,
            "resource_complete": False,
            "scope": "legacy whole-corpus proof ladder is not selected; independent per-emission bounds remain",
        }
        private_json(Path(run_dir) / "public_codegen_resources.json", result)
        return result

    from merlin.development.backend.feedback.execution import profile_runtime as compiler_profile_runtime

    try:
        owner = compiler_profile_runtime.functional_service(run_dir)
        result, _ = _observe(
            owner=owner,
            candidate=candidate,
            context=context,
            capsules_root=capsules_root,
            contract=contract,
            facts_path=facts_path,
            work_root=Path(run_dir).absolute() / "grading_public_scalability",
        )
    except Exception as error:  # noqa: BLE001 -- preparation failure cannot skip a selected prerequisite
        result = {
            "status": "unavailable",
            "required": True,
            "resource_complete": False,
            "reason": type(error).__name__,
            "error": _exception_record(error),
        }
    private_json(Path(run_dir) / "public_codegen_resources.json", result)
    return result


def _observe(*, owner, candidate, context, capsules_root, contract, facts_path, work_root):
    from merlin.development.backend.feedback.execution import compile_only
    from merlin.target.registry.descriptor_loading import load_target_experiment

    policy = selected_policy(owner) if owner is not None else None
    if policy is None:
        return {"status": "not_selected", "required": False, "resource_complete": False}, None
    result = {
        "status": "unavailable",
        "required": policy["required"],
        "resource_complete": False,
        "selection": policy,
        "compiler_tool_profile_sha256": owner.profile["sha256"],
        "scope": "original public emission resource acceptance only; no numerical or RTL credit",
    }
    before = exact_tree_record(Path(candidate))["sha256"]
    result["candidate_sha256"] = before
    actual = None
    try:
        actual = compile_only.run(
            submission=Path(candidate),
            target_experiment=load_target_experiment(context.descriptor),
            capsules_root=capsules_root,
            contract=contract,
            facts_path=facts_path,
            common_service=owner,
            work_root=work_root,
            timeout_s=policy["timeout_s"],
            selected=",".join(policy["capsules"]),
            scale_probe=True,
            ir_budget=DeviceIRBudget(**policy["device_ir_budget"]),
        )
        owner._verify()
        if (
            exact_tree_record(Path(candidate))["sha256"] != before
            or actual.get("candidate_sha256") != before
            or actual.get("compiler_tool_profile_sha256") != owner.profile["sha256"]
        ):
            raise ValueError("public scalability source or original profile changed")
        report = work_root / "public_compile_only.json"
        if json.loads(report.read_bytes()) != actual:
            raise ValueError("public scalability lost its actual protected observation")
        accepted = resources_complete(actual, policy)
        result.update(
            status="accepted" if accepted else "incomplete",
            resource_complete=accepted,
            candidate_sha256=before,
            report={"path": str(report), "sha256": sha256_file(report)},
        )
    except Exception as error:  # noqa: BLE001 -- resource refusal cannot become a numerical verdict
        result["reason"] = type(error).__name__
        result["error"] = _exception_record(error)
        actual = None  # Failed custody cannot supply attributed public samples.
    return result, actual


def observe_authoring(*, owner, candidate, context, capsules_root, contract, facts_path, work_root):
    """Run only an original selected public ladder and return a path-free projection.

    This is independent of numerical capsule grading. The caller publishes it
    before expensive simulations, without exposing protected owner, tool or
    artifact paths. A missing original selection performs no compiler work.
    """
    from merlin.development.backend.feedback.disclosure import restricted

    if restricted():
        return {
            "status": "UNKNOWN",
            "ran": False,
            "all_pass": False,
            "reason": "author disclosure policy withholds fact-derived probe inputs",
        }
    from merlin.development.backend.feedback.execution.snapshot_evaluator import selected

    if selected():
        return None
    result, actual = _observe(
        owner=owner,
        candidate=candidate,
        context=context,
        capsules_root=capsules_root,
        contract=contract,
        facts_path=facts_path,
        work_root=work_root,
    )
    if result["status"] == "not_selected":
        return None
    public = {
        "schema": "merlin.phase1.public_codegen_resources.v1",
        "status": result["status"],
        "required": result["required"],
        "resource_complete": result["resource_complete"],
        "selection": result["selection"],
        "compiler_tool_profile_sha256": result["compiler_tool_profile_sha256"],
        "candidate_sha256": result["candidate_sha256"],
        "samples": [],
        "scope": "selected original public emission resources only; no numerical, offload or execution credit",
        "next_action": "Inspect the named public member with --compile-only --scale-probe before expensive simulation.",
    }
    if result.get("reason"):
        public["reason"] = "original public scale observation unavailable"
    if actual is not None:
        for row in actual["cases"]:
            sample = {name: row.get(name) for name in ("member", "scale", "variant", "lowering_status", "wall_seconds")}
            growth = row.get("code_growth") or {}
            sample["code_growth"] = {
                name: growth[name]
                for name in (
                    "bytes",
                    "function_code_bytes",
                    "lines",
                    "inline_asm_sites",
                    "call_sites",
                    "functions",
                    "basic_blocks",
                )
                if name in growth
            }
            sample["resource_budget"] = row.get("resource_budget")
            if row.get("reason"):
                sample["reason"] = "emission unavailable or original aggregate deadline exhausted"
            public["samples"].append(sample)
    return public


def _read_observation(path):
    from merlin.development.backend.feedback.execution.compiler_profile_child import _pin

    pin = _pin(path)
    if Path(pin["path"]).stat().st_size > 32 << 20:
        raise ValueError("retained resource observation exceeds its metadata bound")
    return json.loads(Path(pin["path"]).read_bytes()), pin


def _producer_sources(selection):
    """Reopen the recorded producer revision, without relabeling it current."""
    import hashlib
    import subprocess

    from merlin.development.backend.feedback.execution.compiler_profile_child import _plain

    revision = selection["implementation_revision"]
    records = selection["implementation_sources"]
    names = {"public_scalability_gate.py", "compiler_profile_runtime.py"}
    if (
        len(revision) != 40
        or any(c not in "0123456789abcdef" for c in revision)
        or len(records) != 2
        or {Path(row["path"]).name for row in records} != names
    ):
        raise ValueError("retained resource producer source selection is incomplete")
    sources = []
    for row in records:
        path = _plain(row["path"])
        repo = next((parent for parent in path.parents if (parent / ".git").exists()), None)
        if repo is None:
            raise ValueError("retained resource producer is not its selected repository source")
        relative = path.relative_to(repo)
        admitted = (
            Path("src/merlin/development/backend/feedback") / path.name,
            Path("packages/merlin-experiments/src/merlin/experiments/phase1/feedback") / path.name,
        )
        if relative not in admitted:
            raise ValueError("retained resource producer is not its selected repository source")
        raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{relative}"], timeout=10)
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("retained resource producer differs from its recorded revision")
        sources.append({"source": row, "repository": str(repo), "revision": revision, "relative_path": str(relative)})
    return sources


def verify_retained_resource_observation(run_dir, *, selection_path, verdict_path):
    """Reopen actual resource products; this issues no new execution observation.

    A separate qualification may join this closure with older numerical evidence
    for the exact same compiler. Its producer revision and actual invocations
    stay independently attributed; no historical verdict is rewritten here.
    """
    import yaml

    from merlin.common.digest import hash_tree
    from merlin.common.process import invocation_records as invocation_record
    from merlin.development.backend.feedback.execution import (
        compile_only,
        scale_probe_child,
    )
    from merlin.development.backend.feedback.execution import (
        profile_runtime as compiler_profile_runtime,
    )
    from merlin.development.backend.feedback.execution.compiler_profile_child import _pin, _plain
    from merlin.development.backend.optimization.search.retained_revisions import source_inventory
    from merlin.development.targetgen.sandbox import grants as bwrap
    from merlin.development.testgen.public_sources import public_scale_sources

    run_dir = _plain(run_dir)
    selection, selection_pin = _read_observation(selection_path)
    verdict, verdict_pin = _read_observation(verdict_path)
    if (
        selection.get("schema") != "merlin.operator.resource_only_probe.v1"
        or selection.get("original_run") != str(run_dir)
        or verdict.get("selection") != selection_pin
        or verdict.get("candidate_unchanged") is not True
    ):
        raise ValueError("retained resources lost their original run or selected observation")
    for role, path in (
        ("original_environment", run_dir / "environment.yaml"),
        ("original_archive", run_dir / "input_bundle_manifest.yaml"),
    ):
        if selection[role] != _pin(path):
            raise ValueError("retained resource original inputs changed")
    config_path = selection["original_config"]["path"]
    if _pin(config_path) != selection["original_config"]:
        raise ValueError("retained resource launch configuration changed")
    config = yaml.safe_load(Path(config_path).read_bytes())
    producer_sources = _producer_sources(selection)
    owner = compiler_profile_runtime.functional_service(run_dir)
    with compile_only._snapshot_operation(owner):
        policy = selected_policy(owner)
        if policy is None:
            raise ValueError("retained resource observation has no selected original policy")
        candidate = _plain(selection["candidate"]["path"])
        if (
            hash_tree(candidate) != selection["candidate"]["hash_tree"]
            or hash_tree(run_dir / "submission")["sha256"] != selection["candidate"]["hash_tree"]["sha256"]
        ):
            raise ValueError("retained resource compiler differs from the original graded source")
        result = verdict["result"]
        require_complete(result)
        if result.get("status") != "accepted" or result.get("resource_complete") is not True:
            raise ValueError("retained resource observation is incomplete")
        report, report_pin = _read_observation(result["report"]["path"])
        facts = bwrap.frozen_selected_rtl_facts(owner.workspace, owner.bundle, repo=owner.repo)
        if (
            result["report"] != report_pin
            or result.get("selection") != policy
            or selection.get("policy") != policy
            or result.get("candidate_sha256") != exact_tree_record(candidate)["sha256"]
            or report.get("candidate_sha256") != result["candidate_sha256"]
            or result.get("compiler_tool_profile_sha256") != owner.profile["sha256"]
            or report.get("compiler_tool_profile_sha256") != owner.profile["sha256"]
            or selection.get("facts") != _pin(facts)
            or report.get("facts_sha256") != _pin(facts)["sha256"]
            or not resources_complete(report, policy)
        ):
            raise ValueError("retained resource report changed its selected policy/compiler/facts")
        public, contract = run_dir / "private_corpus_input/public", run_dir / "private_corpus_input/contract"
        if selection.get("public") != str(public) or selection.get("contract") != str(contract):
            raise ValueError("retained resources selected another original public corpus")
        descriptor = _pin(selection["descriptor"]["path"])
        environment = yaml.safe_load((run_dir / "environment.yaml").read_bytes())
        argv = environment["run_config"]["launcher_argv"]
        if (
            descriptor != selection["descriptor"]
            or argv.count("--descriptor") != 1
            or descriptor["path"] != argv[argv.index("--descriptor") + 1]
        ):
            raise ValueError("retained resource descriptor changed")
        facts_document = json.loads(facts.read_bytes())
        edges = {
            row["cols"]
            for row in facts_document.get("facts", {}).get("arrays", [])
            if type(row.get("cols")) is int and row["cols"] > 0
        }
        capsules = capsule_capsule_common.discover_capsules(str(public), labels={"public", "dev"}, contract=contract)
        cases, manifest = public_scale_sources.derive(
            capsules,
            selected=",".join(policy["capsules"]),
            index_bits=owner.profile["host_index_bits"],
            tile_edge=next(iter(edges)) if len(edges) == 1 else None,
        )
        if manifest != report["source_manifest"]:
            raise ValueError("retained resource cases differ from the original public producer")
        fields = (
            "lowering_status",
            "compilation_status",
            "code_growth",
            "resource_budget",
            "emission_invocation_sha256",
            "translation_invocation_sha256",
            "static_obligations",
        )
        invocations, products = {}, {}
        candidate_sources = [
            _pin(candidate / name)
            for name, kind, _mode in source_inventory(
                candidate,
                max_bytes=32 << 20,
                max_files=8192,
                exclude_generated=True,
            )["entries"]
            if kind == "file"
        ]
        for index, (case, row) in enumerate(zip(cases, report["cases"], strict=True)):
            root = Path(report_pin["path"]).parent
            source = root / "sources" / f"case-{index:03d}.mlir"
            output = root / f"case-{index:03d}"
            if source.read_text() != case.source_ir or _pin(source)["sha256"] != row["source_sha256"]:
                raise ValueError("retained scale source differs from its original producer")
            analysis = scale_probe_child._verify_products(output, {name: row[name] for name in fields})
            if DeviceIRBudget(**policy["device_ir_budget"]).evaluate(analysis) != row["resource_budget"]:
                raise ValueError("retained scale cost differs from its actual emitted LLVM")
            records = [(_pin(path), invocation_record.verify(path)) for path in sorted(output.rglob("invocation.json"))]
            emission = [record for _, record in records if record["stage"] == "public_scale_source_lowering"]
            translation = [record for _, record in records if record["stage"] == "public_scale_llvm_translation"]
            custody = [record for _, record in records if record["stage"] == "ordinary_source_emission"]
            if (
                len(emission) != 1
                or len(translation) != 1
                or emission[0]["inputs"] != [_pin(source)]
                or translation[0]["inputs"] != [_pin(output / "generated/lowered.llvm.mlir")]
                or translation[0]["outputs"] != [_pin(output / "generated/lowered.ll")]
                or translation[0]["argv"]
                != [
                    translation[0]["executable"]["path"],
                    "--mlir-to-llvmir",
                    str(output / "generated/lowered.llvm.mlir"),
                    "-o",
                    str(output / "generated/lowered.ll"),
                ]
                or translation[0]["executable"]["sha256"] != config["runtime_pins"]["mlir_translate"]["sha256"]
                or len(custody) != 1
                or any(pin not in custody[0]["dependencies"] for pin in candidate_sources)
            ):
                raise ValueError("retained scale invocations do not bind the original source and measured LLVM")
            products[str(source)] = _pin(source)
            for pin, record in records:
                invocations[pin["path"]] = pin
                for product in record["outputs"]:
                    products[product["path"]] = product
        for pin in (
            selection_pin,
            verdict_pin,
            report_pin,
            descriptor,
            selection["original_config"],
            selection["original_environment"],
            selection["original_archive"],
            selection["facts"],
            *invocations.values(),
            *products.values(),
        ):
            if _pin(pin["path"]) != pin:
                raise ValueError("retained resource evidence changed during verification")
        owner._verify()
    return {
        "selection": selection_pin,
        "verdict": verdict_pin,
        "report": report_pin,
        "result": result,
        "products": sorted(products.values(), key=lambda pin: pin["path"]),
        "invocations": sorted(invocations.values(), key=lambda pin: pin["path"]),
        "producer_sources": producer_sources,
        "scope": "reopened original resource observations; no new execution or historical grade promotion",
    }


# ---- codegen_scalability -----------------------------------------------------------------------
# Public, fact-derived emitted-text and optional object-build observations.
#
# No model capture, validation shape, numerical answer or simulator is read here.
# Text/object size and compile time are observations, not executed work or a
# correctness theorem. A runtime loop may correctly produce smaller code.

_MAX_BUILD_PROBE_BYTES = 16 * 1024 * 1024


def selected_build_service(target: str):
    """Select the exact trusted host capability before authoring, never in the agent sandbox."""
    from merlin.development.targetgen.execution.native.native_model_execution import _build_service_for

    service = _build_service_for(target)
    return service, _selected_build_inputs(service, target)


def _selected_build_inputs(service, target: str) -> dict:
    """Pin only the direct tools and trusted build capability this observation uses.

    This is not a transitive toolchain, sysroot or compiler-correctness attestation.
    """
    from merlin.compiler.lower.model import codegen, toolchain
    from merlin.development.targetgen.execution.runner import compile as contract_compile
    from merlin.target.contract.build.services import BuildOnlyService, file_digest
    from merlin.target.contract.harness.readback_policy import compile_sources

    if type(service) is not BuildOnlyService:
        raise ValueError("public object telemetry requires an exact pure build service")
    service.verify(target)
    recipe = service.recipe.with_effective_abi()

    def pin(path) -> dict:
        selected = Path(path)
        if not selected.is_absolute() or not selected.is_file():
            raise ValueError("public object telemetry requires explicitly selected regular tools")
        resolved = selected.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError("selected build tool is not a regular file")
        return {"selected": str(selected), "resolved": str(resolved), "sha256": file_digest(resolved)}

    pins = tuple(sorted(service.source_pins))
    return {
        "clang": pin(toolchain.clang()),
        "mlir_translate": pin(toolchain.mlir_translate()),
        "recipe_compiler": pin(recipe.compiler),
        "recipe_cflags": list(recipe.cflags),
        "recipe_march": recipe.march(),
        "recipe_mabi": recipe.mabi(),
        "object_compile_flags": [*codegen.RISCV_FLAGS, recipe.march(), recipe.mabi(), "-fstack-usage"],
        "stack_frame_policy": recipe.require_kernel_stack_frame().record(),
        "producer_modules": {
            "codegen": pin(codegen.__file__),
            "contract_compile": pin(contract_compile.__file__),
            **{f"contract_{path.stem}": pin(path) for path in compile_sources()[1:]},
            "toolchain": pin(toolchain.__file__),
            "observer": pin(__file__),
        },
        "trusted_source_pins_sha256": hashlib.sha256(
            json.dumps(pins, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "trusted_source_pin_count": len(pins),
        "scope": "direct selected tools and trusted source pins, not transitive toolchain closure",
    }


def _compile_observation(artifact: str, work: Path, *, target: str, service, budget_s: int) -> dict:
    """Compile the exact emitted LLVM MLIR once, without linking or executing it."""
    from merlin.development.targetgen.execution.runner.compile import llvm_mlir_to_object

    source = artifact.encode("utf-8")
    result = {
        "status": "compile_error",
        "artifact_sha256": hashlib.sha256(source).hexdigest(),
        "artifact_bytes": len(source),
        "budget_s": budget_s,
        "scope": "public_build_only; not execution, numerical equivalence or certification",
    }
    try:
        selected = _selected_build_inputs(service, target)
        result["build_selection"] = selected
        # Parsing an unbounded generated module would evade the subprocess
        # budget. Report its exact size/hash without claiming a compile.
        if len(source) > _MAX_BUILD_PROBE_BYTES:
            result["status"] = "source_exceeds_probe_budget"
            return result
        started = time.monotonic()
        try:
            obj = llvm_mlir_to_object(
                artifact,
                work / "scalability_object",
                target=target,
                _build_service=service,
                build_timeout_s=budget_s,
            )
        finally:
            result["compile_wall_s"] = time.monotonic() - started
        verified = _selected_build_inputs(service, target)
        if verified != selected:
            result["status"] = "selection_changed"
            raise ValueError("selected build inputs changed during public object compilation")
        if not obj.is_file() or obj.is_symlink():
            raise ValueError("public object build produced no regular object")
        from merlin.target.contract.build.services import file_digest

        object_sha = file_digest(obj)
        build_root = work / "scalability_object"
        stack_path, abi_path = build_root / "kernel.stack_frame.json", build_root / "kernel.abi.json"
        stack, abi = json.loads(stack_path.read_text()), json.loads(abi_path.read_text())
        compiled_source = build_root / ("kernel.arena.ll" if stack.get("repair") is not None else "kernel.ll")
        if (
            stack.get("status") != "passed"
            or stack.get("object_sha256") != object_sha
            or stack.get("llvm_ir_sha256") != file_digest(compiled_source)
            or abi.get("object_sha256") != object_sha
            or abi.get("abi") != selected["recipe_mabi"].partition("=")[2]
        ):
            raise ValueError("public object build receipts do not match the measured object")
        result.update(
            {
                "status": "compiled",
                "object_bytes": obj.stat().st_size,
                "object_sha256": object_sha,
                "emitted_llvm_ir_sha256": file_digest(build_root / "kernel.ll"),
                "compiled_llvm_ir_sha256": stack["llvm_ir_sha256"],
                "stack_frame_receipt_sha256": file_digest(stack_path),
                "abi_receipt_sha256": file_digest(abi_path),
                "stack_repair": stack.get("repair") is not None,
            }
        )
    except Exception as exc:  # noqa: BLE001 -- optional diagnostics cannot become a grade
        result["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    return result


def run(
    package: str | Path,
    *,
    target: str,
    contract: str | Path | None = None,
    timeout: int = 30,
    additional_forbidden: tuple[str, ...] = (),
    build_service=None,
    build_timeout_s: int = 30,
) -> dict:
    """Observe the same public contraction at geometric multiples of its tile.

    The experiment ratios are fixed, but geometry and formats come from the
    selected target inputs. The default result remains emit-only. An explicit
    pure build service enables separately bounded object compilation of those
    same emitted bytes; it does not link or execute them. Neither a size
    threshold nor a compiler strategy follows from the measurements.
    """
    from merlin.development.targetgen.conformance.lowering_coverage import probe_shape
    from merlin.development.targetgen.corpus.binding import derive_binding
    from merlin.development.targetgen.policies.form_regimes import tile_edge
    from merlin.target.registry.descriptor_loading import experiment_for

    if build_service is not None:
        if type(build_timeout_s) is not int or not 0 < build_timeout_s <= 120:
            raise ValueError("public object-build budget must be 1..120 seconds per sample")
        _selected_build_inputs(build_service, target)

    experiment = experiment_for(target)
    if experiment is None:
        raise ValueError("public scalability probe requires an explicitly selected target experiment")
    binding = derive_binding(experiment, {})
    tile = tile_edge(target, operand=binding.operand_dtype)
    samples = []
    baseline = None
    for scale in (1, 2, 4, 8):
        shape = (tile * scale,) * 3
        started = time.monotonic()
        build_record = None
        build_callback_wall = 0.0

        def observe(artifact: str, work: Path) -> None:
            nonlocal build_record, build_callback_wall
            began = time.monotonic()
            try:
                build_record = _compile_observation(
                    artifact, work, target=target, service=build_service, budget_s=build_timeout_s
                )
            finally:
                build_callback_wall = time.monotonic() - began

        probe_kwargs = {
            "target": target,
            "m": shape[0],
            "k": shape[1],
            "n": shape[2],
            "operand_mlir": binding.mlir_dtype(binding.operand_dtype),
            "accum_mlir": binding.mlir_dtype(binding.accum_dtype),
            "contract": contract,
            "timeout": timeout,
            "additional_forbidden": additional_forbidden,
        }
        if build_service is not None:
            probe_kwargs["on_lowered"] = observe
        outcome, detail, lines = probe_shape(package, **probe_kwargs)
        elapsed = time.monotonic() - started
        if scale == 1 and outcome == "lowered" and lines > 0:
            baseline = lines
        sample = {
            "extent_scale": scale,
            "shape_mkn": list(shape),
            "contraction_volume_ratio": scale**3,
            "outcome": outcome,
            "artifact_nonempty_lines": lines,
            "line_ratio_vs_baseline": lines / baseline if baseline and outcome == "lowered" else None,
            "emit_pipeline_wall_s": max(0.0, elapsed - build_callback_wall),
        }
        if build_service is not None:
            sample["build_only"] = build_record or {
                "status": "not_emitted",
                "scope": "public_build_only; no object compiled",
            }
        if detail:
            sample["detail"] = detail
        samples.append(sample)
    result = {
        "schema": (
            "merlin.codegen_scalability_probe.v1" if build_service is None else "merlin.codegen_scalability_probe.v2"
        ),
        "scope": "public_emit_only" if build_service is None else "public_emit_and_build_only",
        "tile_edge": tile,
        "samples": samples,
        "observations_complete": all(
            row["outcome"] == "lowered" and row["artifact_nonempty_lines"] > 0 for row in samples
        ),
        "note": (
            "Public fact-derived emit-only observations: not compilation, executed work, numerical "
            "equivalence or certification. Constant-sized or smaller looped code can be correct. "
            "Use the existing compiler/build and native numerical checks before claiming success."
            if build_service is None
            else "Public fact-derived emitted-text and optional object-build observations only: no link, "
            "execution, numerical equivalence or certification. Object size, compile time and failure "
            "are advisory; neither loop syntax nor a size trend is a correctness condition."
        ),
    }
    return result
