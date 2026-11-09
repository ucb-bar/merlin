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
) -> dict:
    """Compile isolated kernels and emit a real C staged chain, default-off.

    Exact canonical IR dedup normalizes only function/SSA spelling. Unknown,
    provenance, numerical, indexing and effect attributes remain in the key.
    Signature/shape similarity alone never grants implementation reuse.
    """
    from xdsl.dialects.builtin import StringAttr

    from merlin.common.paths import runtime_dir
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower import deployable_outline, kernel_backend, toolchain
    from merlin.llvmlower.portable_math import cache_token
    from merlin.runtime import outlined_chain
    from merlin.xdsl_dialects._common import text

    if type(vectorize) is not bool:
        raise ValueError("explicit boolean vectorization selection required")
    if not isinstance(prefix, str) or not prefix.isascii() or not prefix.isidentifier():
        raise ValueError("explicit C-safe program prefix required")
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
            workspace_budget_bytes=workspace_budget_bytes,
            alignment_bytes=alignment_bytes,
        )
        (output / "outlined.mlir").write_text(text(plan.outlined.module))
        (output / "plan.json").write_text(json.dumps(plan.to_dict(), indent=2) + "\n")
        identity = {
            "compiler": str(toolchain.clang()),
            "compiler_sha256": sha256_file(toolchain.clang()),
            "compute_cflags": list(compute_cflags),
            "vectorize": vectorize,
            "codegen_target": codegen_target,
            "portable_math_policy": cache_token(),
            "compiler_owner_sha256": {
                str(Path(module.__file__)): sha256_file(module.__file__)
                for module in (deployable_outline, kernel_backend)
            },
        }
        cache, builds, mappings, entries = {}, [], [], []
        for step in plan.dispatches:
            kernel = deployable_outline.canonical_kernel(plan.outlined.module, step["symbol"])
            key = deployable_outline.kernel_code_key(kernel, identity)
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
                (work / "source.mlir").write_text(text(kernel))
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
        emitted = outlined_chain.render_outlined_chain(plan, tuple(entries), roles=roles, prefix=prefix)
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
        source_pins = {
            str(path): sha256_file(path)
            for path in (
                source,
                header,
                driver,
                *support,
                Path(deployable_outline.__file__),
                Path(kernel_backend.__file__),
                Path(outlined_chain.__file__),
                Path(__file__),
            )
        }
        if source.read_bytes() != raw:
            raise ValueError("source graph changed during outlined compilation")
        report.update(
            {
                "status": "built",
                "dispatches": len(plan.dispatches),
                "unique_kernels": len(builds),
                "plan": str(output / "plan.json"),
                "driver": str(driver),
                "header": str(header),
                "kernel_builds": builds,
                "source_to_kernel": mappings,
                "runtime_sources": [str(runtime / "merlin_staged.c"), str(runtime / "merlin_descriptors.c")],
                "runtime_include_directory": str(runtime),
                "workspace_bytes": plan.workspace_bytes,
                "driver_metadata": emitted["metadata"],
                "input_sha256": source_pins,
                "target_link_and_execution": "explicit caller platform recipe required",
                "implicit_compiler_import_closure_complete": False,
                "kernel_cache_scope": (
                    "within this build only; function/SSA spelling normalized, all attributes retained"
                ),
            }
        )
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
    built = build_outlined_program(args.source, args.output, roles=StagedRoles(**role_data), **config)
    print(json.dumps({"status": built["status"], "build": str(args.output / "build.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
