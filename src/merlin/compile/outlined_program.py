"""Build a strict deployable outlined program without compiling its parent driver.

Outputs are ordinary kernel objects and a controller/worker C driver. A selected
platform supplies startup, pinned persistent workers, clocks, link/runtime input
closure and a separate complete-output gate. No hardware is launched here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.common.strict_json import loads
from merlin.runtime.staged_dispatch import StagedRoles


def build_outlined_program(
    source: str | Path,
    output: str | Path,
    *,
    roles: StagedRoles,
    worker_slots: tuple[int, ...],
    workspace_budget_bytes: int,
    alignment_bytes: int,
    compute_cflags: tuple[str, ...],
    vectorize: bool = False,
    codegen_target: str = "rvv",
    prefix: str = "merlin_chain",
    prepared_staged_kernels=None,
    data_preparation: bool = False,
    packing_tile_shape: tuple[int, int] | None = None,
    packing_tile_budget_bytes: int | None = None,
    frozen_inputs=None,
    analysis_workspace_budget_bytes: int | None = None,
) -> dict:
    """Compile isolated kernels and emit a real C staged chain, default-off.

    Exact canonical IR dedup normalizes only function/SSA spelling. Unknown,
    provenance, numerical, indexing and effect attributes remain in the key.
    Signature/shape similarity alone never grants implementation reuse.
    """
    from xdsl.dialects.builtin import StringAttr

    from merlin.common.paths import runtime_dir
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower import deployable_ir, deployable_outline, kernel_backend, toolchain
    from merlin.llvmlower.portable_math import cache_token
    from merlin.runtime import outlined_chain, prepared_staged_kernel
    from merlin.xdsl_dialects.lowering import outline as outline_owner

    if type(vectorize) is not bool:
        raise ValueError("explicit boolean vectorization selection required")
    if not isinstance(prefix, str) or not prefix.isascii() or not prefix.isidentifier():
        raise ValueError("explicit C-safe program prefix required")
    if analysis_workspace_budget_bytes is not None and (
        frozen_inputs is None
        or type(analysis_workspace_budget_bytes) is not int
        or not 0 < analysis_workspace_budget_bytes < 1 << 63
    ):
        raise ValueError("analysis workspace bound requires explicit frozen inputs and positive LP64 byte limit")
    source, output = Path(source).resolve(strict=True), Path(output)
    raw = source.read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    (output / "source.mlir").write_bytes(raw)
    report = {
        "schema": "merlin.deployable_outlined_build.v1",
        "status": "preparing",
        "source": str(source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "roles": roles.to_dict(),
        "hardware_contacted": False,
        "numerical_gate_performed": False,
        "physical_busy": None,
        "full_model_admission": False,
        "parent_forward_compiled": False,
    }
    try:
        module = parse_mlir_text(raw.decode())
        plan = deployable_outline.plan_deployable_chain(
            module,
            worker_slots=worker_slots,
            worker_count=len(roles.compute_harts),
            workspace_budget_bytes=(
                workspace_budget_bytes if analysis_workspace_budget_bytes is None else analysis_workspace_budget_bytes
            ),
            alignment_bytes=alignment_bytes,
            data_preparation=data_preparation,
            packing_tile_shape=packing_tile_shape,
            packing_tile_budget_bytes=packing_tile_budget_bytes,
        )
        prepared_staged_kernels = {} if prepared_staged_kernels is None else dict(prepared_staged_kernels)
        frozen_blob_objects = []
        frozen_blob_paths = []
        analysis_workspace_bytes = plan.workspace_bytes
        if frozen_inputs is not None:
            import subprocess

            from merlin.compile import frozen_inputs as frozen_module
            from merlin.llvmlower import codegen
            from merlin.targetgen.contract import harness_blobs

            plan, blob = frozen_module.specialize(plan, frozen_inputs, output / "offline")
            remap = {
                row["source_dispatch_index"]: row["runtime_dispatch_index"]
                for row in plan.frozen_inputs["dispatch_mapping"]
                if not row["folded"]
            }
            if set(prepared_staged_kernels) - set(remap):
                raise ValueError("explicit prepared selection cannot be silently folded offline")
            prepared_staged_kernels = {remap[i]: kernel for i, kernel in prepared_staged_kernels.items()}
            sources = harness_blobs.stage_harness_blobs(
                output, {prefix + "_frozen_blob": {"bytes": blob, "align": plan.alignment_bytes, "elems": len(blob)}}
            )
            for assembly in sources:
                obj = assembly.with_suffix(".o")
                subprocess.run(
                    [
                        str(toolchain.clang()),
                        *codegen.RISCV_FLAGS,
                        *compute_cflags,
                        "-c",
                        assembly.name,
                        "-o",
                        obj.name,
                    ],
                    cwd=output,
                    check=True,
                    capture_output=True,
                )
                frozen_blob_objects.append(str(obj))
                frozen_blob_paths.extend([assembly, obj, assembly.with_suffix(".bin")])
            frozen_blob_paths.extend(
                [output / "harness_blobs.json", Path(frozen_module.__file__), Path(harness_blobs.__file__)]
            )
            (output / "frozen-inputs.json").write_text(json.dumps(plan.frozen_inputs, indent=2) + "\n")
            frozen_blob_paths.append(output / "frozen-inputs.json")
        plan, state_layout = prepared_staged_kernel.reserve_states(
            plan,
            prepared_staged_kernels,
            budget_bytes=workspace_budget_bytes - (0 if frozen_inputs is None else len(blob)),
        )
        prepared_staged_kernel.verify_selection(plan, prepared_staged_kernels, state_layout)
        (output / "outlined.mlir").write_text(deployable_ir.checked_generic_text(plan.outlined.module))
        plan_record = {**plan.to_dict(), "provider_state_layout": state_layout}
        plan_record["effective_worker_slots"] = {
            str(index): list(prepared_staged_kernels[index].worker_slots)
            if index in prepared_staged_kernels
            else [step["worker_slot"]]
            for index, step in enumerate(plan.dispatches)
        }
        if prepared_staged_kernels:
            plan_record["arithmetic_policy"] = (
                "explicit selected worker callbacks; provider owns source-equivalent output partition"
            )
        (output / "plan.json").write_text(json.dumps(plan_record, indent=2) + "\n")
        identity = {
            "compiler": str(toolchain.clang()),
            "compiler_sha256": sha256_file(toolchain.clang()),
            "compute_cflags": list(compute_cflags),
            "vectorize": vectorize,
            "codegen_target": codegen_target,
            "portable_math_policy": cache_token(),
            "compiler_owner_sha256": {
                str(Path(module.__file__)): sha256_file(module.__file__)
                for module in (deployable_ir, deployable_outline, kernel_backend, prepared_staged_kernel, outline_owner)
            },
        }
        if data_preparation:
            from merlin.llvmlower import deployable_packing

            identity["data_preparation"] = True
            if packing_tile_shape is not None:
                identity["controller_packing_policy"] = plan.data_preparation["controller_packing_policy"]
            identity["compiler_owner_sha256"][str(Path(deployable_packing.__file__))] = sha256_file(
                deployable_packing.__file__
            )
        cache, builds, mappings, entries, prepared_builds = {}, [], [], [], {}
        for index, step in enumerate(plan.dispatches):
            kernel = deployable_outline.canonical_kernel(plan.outlined.module, step["symbol"])
            key = deployable_outline.kernel_code_key(kernel, identity)
            if index in prepared_staged_kernels:
                selected = prepared_staged_kernels[index]
                work = output / ("prepared_" + str(index))
                work.mkdir()
                (work / "source.mlir").write_text(deployable_ir.checked_generic_text(kernel))
                record = {**selected.record(), "canonical_code_key": key, "state_layout": state_layout[index]}
                (work / "prepared.json").write_text(json.dumps(record, indent=2) + "\n")
                prepared_builds[index] = record
                entries.append(selected.compute)
                mappings.append(
                    {
                        "source_symbol": step["symbol"],
                        "provenance": step["provenance"],
                        "entry": selected.compute,
                        "entry_kind": "prepared_staged_callbacks",
                        "canonical_code_key": key,
                        "worker_slots": list(selected.worker_slots),
                        "prepared_record": str(work / "prepared.json"),
                    }
                )
                continue
            if key not in cache:
                entry = prefix + "_kernel_" + str(len(cache))
                function = next(op for op in kernel.body.block.ops if op.name == "func.func")
                function.properties["sym_name"] = StringAttr(entry)
                work = output / ("kernel_" + str(len(cache)))
                built = kernel_backend.compile_deployable_kernel(
                    kernel,
                    work,
                    entry=entry,
                    extra_cflags=compute_cflags,
                    vectorize=vectorize,
                    codegen_target=codegen_target,
                )
                (work / "source.mlir").write_text(deployable_ir.checked_generic_text(kernel))
                built["canonical_code_key"] = key
                builds.append(built)
                cache[key] = built
            built = cache[key]
            entries.append(built["entry"])
            mappings.append(
                {
                    "source_symbol": step["symbol"],
                    "provenance": step["provenance"],
                    "entry": built["entry"],
                    "canonical_code_key": key,
                    "worker_slot": step["worker_slot"],
                    "object_sha256": built["object_sha256"],
                }
            )
        emitted = outlined_chain.render_outlined_chain(
            plan,
            tuple(entries),
            roles=roles,
            prefix=prefix,
            prepared_kernels=prepared_staged_kernels,
            provider_state_layout=state_layout,
        )
        header, driver = output / (prefix + ".h"), output / (prefix + ".c")
        header.write_text(emitted["header"])
        driver.write_text(emitted["source"])
        runtime = runtime_dir() / "c"
        support = [
            runtime / name
            for name in (
                "merlin_staged.c",
                "merlin_staged.h",
                "merlin_model.h",
                "merlin_descriptors.c",
                "merlin_descriptor_internal.h",
            )
        ]
        if prepared_staged_kernels:
            support.append(runtime / "merlin_staged_kernel.h")
        source_pins = {
            str(path): sha256_file(path)
            for path in (
                source,
                header,
                driver,
                *support,
                Path(deployable_outline.__file__),
                Path(outline_owner.__file__),
                Path(deployable_ir.__file__),
                Path(kernel_backend.__file__),
                Path(outlined_chain.__file__),
                Path(prepared_staged_kernel.__file__),
                Path(__file__),
            )
        }
        for built in builds:
            if sha256_file(built["object"]) != built["object_sha256"]:
                raise ValueError("ordinary kernel object changed before build closure")
            source_pins[built["object"]] = built["object_sha256"]
        if frozen_inputs is not None:
            for path, digest in plan.frozen_inputs["input_sha256"].items():
                if sha256_file(path) != digest:
                    raise ValueError("frozen input changed before build closure")
            source_pins.update(plan.frozen_inputs["input_sha256"])
            source_pins.update({str(path): sha256_file(path) for path in frozen_blob_paths})
            # Published host evidence is an explicit manifest. The retained .so
            # must equal the actually loaded private-image digest; NFS loader
            # temporaries are neither dependencies nor publication candidates.
            for row in plan.frozen_inputs["dispatch_mapping"]:
                proof = row.get("proof", {})
                if proof and proof["input_sha256"].get(proof["retained_host_image"]) != proof["host_image_sha256"]:
                    raise ValueError("retained host image manifest differs from actually loaded image")
                for path, digest in proof.get("input_sha256", {}).items():
                    if sha256_file(path) != digest:
                        raise ValueError("offline evaluation dependency changed before build closure")
                    source_pins[path] = digest
        if data_preparation:
            source_pins[str(Path(deployable_packing.__file__))] = sha256_file(deployable_packing.__file__)
        for selected in prepared_staged_kernels.values():
            selected.verify(worker_count=plan.worker_count)
            source_pins.update(selected.input_sha256)
        for work in (output / ("prepared_" + str(index)) for index in prepared_staged_kernels):
            for path in work.iterdir():
                source_pins[str(path)] = sha256_file(path)
        if source.read_bytes() != raw:
            raise ValueError("source graph changed during outlined compilation")
        report.update(
            {
                "status": "built",
                "dispatches": len(plan.dispatches),
                "unique_kernels": len(builds) + len(prepared_builds),
                "plan": str(output / "plan.json"),
                "driver": str(driver),
                "header": str(header),
                "kernel_builds": builds,
                "prepared_kernel_builds": prepared_builds,
                "link_objects": list(
                    dict.fromkeys(
                        [v["object"] for v in builds]
                        + [str(p) for v in prepared_staged_kernels.values() for p in v.objects]
                        + frozen_blob_objects
                    )
                ),
                "source_to_kernel": mappings,
                "runtime_sources": [str(runtime / "merlin_staged.c"), str(runtime / "merlin_descriptors.c")],
                "runtime_include_directory": str(runtime),
                "workspace_bytes": plan.workspace_bytes,
                "workspace_budget_bytes": workspace_budget_bytes,
                "provider_state_layout": state_layout,
                "driver_metadata": emitted["metadata"],
                "input_sha256": source_pins,
                "target_link_and_execution": "explicit caller platform recipe required",
                "implicit_compiler_import_closure_complete": False,
                "source_serialization": "checked_generic_operation_form",
                "kernel_cache_scope": (
                    "within this build only; function/SSA spelling normalized, all attributes retained"
                ),
            }
        )
        if frozen_inputs is not None:
            report["frozen_inputs"] = plan.frozen_inputs
            report["live_storage_bytes"] = plan.workspace_bytes + plan.frozen_inputs["published_blob_bytes"]
            report["storage_accounting"] = {
                "arena_and_live_blob_bytes": report["live_storage_bytes"],
                "arena_and_live_blob_budget_bytes": workspace_budget_bytes,
                "caller_input_bytes": sum(
                    b.bytes for b in plan.buffers if b.arg_index in plan.frozen_inputs["external_input_source_indices"]
                ),
                "caller_output_bytes": sum(
                    next(b.bytes for b in plan.buffers if b.id == bid) for bid in plan.program.results
                ),
                "context_stack_code_and_platform_memory": "caller must admit actual linked image separately",
            }
            if analysis_workspace_budget_bytes is not None:
                report["analysis_plan"] = {
                    "metadata_only": True,
                    "pre_specialization_workspace_bytes": analysis_workspace_bytes,
                    "analysis_workspace_budget_bytes": analysis_workspace_budget_bytes,
                    "final_target_storage_budget_bytes": workspace_budget_bytes,
                }
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = loads(args.config.read_bytes(), max_bytes=1 << 20)
    if not isinstance(config, dict) or config.pop("schema", None) != "merlin.outlined_program.v1":
        raise ValueError("explicit outlined-program config required")
    role_data = config.pop("roles")
    for key in ("cpu_hart_ids", "compute_harts", "profile_harts"):
        if not isinstance(role_data.get(key), list):
            raise ValueError("role policy requires JSON hart lists")
        role_data[key] = tuple(role_data[key])
    for key in ("worker_slots", "compute_cflags"):
        if not isinstance(config.get(key), list):
            raise ValueError("worker assignments and target compiler flags require JSON lists")
        config[key] = tuple(config[key])
    if config.get("packing_tile_shape") is not None:
        if not isinstance(config["packing_tile_shape"], list):
            raise ValueError("packing_tile_shape requires a JSON list")
        config["packing_tile_shape"] = tuple(config["packing_tile_shape"])
    built = build_outlined_program(args.source, args.output, roles=StagedRoles(**role_data), **config)
    print(json.dumps({"status": built["status"], "build": str(args.output / "build.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
