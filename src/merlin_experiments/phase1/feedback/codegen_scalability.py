"""Public, fact-derived emitted-text and optional object-build observations.

No model capture, validation shape, numerical answer or simulator is read here.
Text/object size and compile time are observations, not executed work or a
correctness theorem. A runtime loop may correctly produce smaller code.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

_MAX_BUILD_PROBE_BYTES = 16 * 1024 * 1024


def selected_build_service(target: str):
    """Select the exact trusted host capability before authoring, never in the agent sandbox."""
    from merlin.targetgen.native_model_execution import _build_service_for

    service = _build_service_for(target)
    return service, _selected_build_inputs(service, target)


def _selected_build_inputs(service, target: str) -> dict:
    """Pin only the direct tools and trusted build capability this observation uses.

    This is not a transitive toolchain, sysroot or compiler-correctness attestation.
    """
    from merlin.llvmlower import codegen, toolchain
    from merlin.targetgen.contract import compile as contract_compile
    from merlin.targetgen.contract.build_service import BuildOnlyService, file_digest

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
    from merlin.targetgen.contract.compile import llvm_mlir_to_object

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
        from merlin.targetgen.contract.build_service import file_digest

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
    from merlin.targetgen.capability_probes import tile_edge
    from merlin.targetgen.corpora import experiment_for
    from merlin.targetgen.corpus_spec import derive_binding
    from merlin.targetgen.lowering_coverage import probe_shape

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
