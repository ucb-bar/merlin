"""Build a saved whole model as one bare-metal ELF and optionally execute it.

The board, host package, capture and (optional) device routing are explicit inputs.
This is a compiler/execution receipt, not a Phase 0 release or a claim that a
statically routed accelerator call executed.  Native RTL execution is delegated
to the selected target support backend's public ``run_elf`` implementation.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from merlin.common.paths import out_dir
from merlin.compile import model_execution_inputs as MI
from merlin.compile.host_lane import require_host_isa_dts
from merlin.mining import registry
from merlin.runtime.backends import spike as spike_backend
from merlin.runtime.backends import spike_model
from merlin.runtime.boards import CONSOLE_HTIF, FLOW_BAREMETAL, load_boards
from merlin.targetgen.application_inventory import verify_capture_receipt


class BaremetalModelError(ValueError):
    """Selected bytes cannot support the requested whole-model claim."""


def _sha(path: Path) -> str:
    return MI.file_sha256(path)


def _retain_simulator_failure_output(
    exc: subprocess.TimeoutExpired | subprocess.CalledProcessError,
    output: Path,
    receipt: dict[str, Any],
) -> None:
    """Keep incomplete process streams as byte evidence, never as model OUT."""
    artifacts: dict[str, Any] = {"scope": "diagnostic_partial_simulator_output_only"}
    for stream in ("stdout", "stderr"):
        value = getattr(exc, stream, None)
        if stream == "stdout" and value is None:
            value = getattr(exc, "output", None)
        if value is None:
            continue
        data = value if isinstance(value, bytes) else value.encode("utf-8", errors="surrogateescape")
        path = output / f"simulator_{stream}.partial.bin"
        path.write_bytes(data)
        artifacts[stream] = {"path": str(path), "sha256": _sha(path), "bytes": len(data)}
    receipt["failure_artifacts"] = artifacts


def _output_root(path: str | Path) -> Path:
    lexical = Path(path)
    if lexical.is_symlink():
        raise BaremetalModelError(f"output must not be a symlink: {lexical}")
    output = lexical.resolve()
    root = out_dir().resolve()
    if output == root or not output.is_relative_to(root) or output.exists():
        raise BaremetalModelError(f"output must be a fresh directory below {root}: {output}")
    return output


def _saved_reference(capture: Path, name: str | None, *, execution: bool) -> np.ndarray | None:
    verification = verify_capture_receipt(capture / "model.mlir")
    if verification.get("status") != "verified_materialized":
        raise BaremetalModelError(f"saved capture receipt is not byte-verified: {verification.get('errors')}")
    for required in ("inputs.npz", "input_order.json", "golden.npy"):
        path = capture / required
        if not path.is_file() or path.is_symlink():
            raise BaremetalModelError(f"saved capture lacks a safe {required}")
    if not execution:
        return None
    if name is None or Path(name).name != name or not name.endswith(".npy"):
        raise BaremetalModelError("execution requires one explicit in-bundle .npy reference_file")
    if (capture / "goldens.npz").exists() or (capture / "output_order.json").exists():
        raise BaremetalModelError("multi-output captures have no complete bare-metal OUT gate")
    reference = capture / name
    recorded = json.loads((capture / "capture_receipt.json").read_text()).get("artifacts") or {}
    if not reference.is_file() or reference.is_symlink() or (recorded.get(name) or {}).get("sha256") != _sha(reference):
        raise BaremetalModelError(f"reference {name!r} is absent or not bound by the capture receipt")
    golden = np.load(reference, allow_pickle=False)
    if golden.dtype != np.float32 or golden.size < 1 or not np.isfinite(golden).all():
        raise BaremetalModelError("execution reference must be a nonempty finite float32 array")
    if golden.size > 1_048_576:
        raise BaremetalModelError("complete chunked OUT observability is limited to 1048576 elements")
    return golden.reshape(-1)


def _native_engine(target: str, run: str, facts: dict[str, str]):
    from merlin.targetgen.oracle_policy import selected_l3_engine_report

    selection = selected_l3_engine_report(target)
    if not selection.get("available") or selection.get("engine") != run:
        raise BaremetalModelError(f"requested {run} is not the selected elaborated-RTL engine: {selection}")
    backend, citation, revalidate, prepare = MI.native_engine(target, run, facts)
    if not callable(getattr(backend, "run_elf", None)):
        raise BaremetalModelError(f"selected backend has no public run_elf for {target}")
    return backend, {"selection": selection, "citation": citation}, revalidate, prepare


def compile_saved_model(
    *,
    capture: str | Path,
    package: str | Path,
    board_catalog: str | Path,
    board: str,
    dts: str | Path,
    output: str | Path,
    target: str,
    run: str,
    arena_mb: int,
    timeout_s: int = 300,
    reference_file: str | None = None,
    rtl_facts: str | Path | None = None,
    device: Any | None = None,
    math_archive_symbols: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compile one saved model; ``none`` makes no execution or numerical claim.

    ``reference_file`` is required for execution because a weight-only capture's
    ``golden.npy`` is not necessarily the reference for a W8A8 execution.  This
    executor supports one complete float32 OUT, using chunked Spike console
    records above 4096 values (at most 1048576 values).
    """
    if run not in {"none", "spike", "gsim", "verilator"}:
        raise BaremetalModelError(f"unsupported bare-metal run {run!r}")
    if type(arena_mb) is not int or arena_mb < 1 or type(timeout_s) is not int or timeout_s < 1:
        raise BaremetalModelError("arena_mb and timeout_s must be positive integers")
    paths = [Path(capture), Path(package), Path(board_catalog), Path(dts)]
    if rtl_facts is not None:
        paths.append(Path(rtl_facts))
    if any(path.is_symlink() for path in paths):
        raise BaremetalModelError("explicit inputs must not be symlinks")
    capture_path, package_path, catalog_path, dts_path = (path.resolve(strict=True) for path in paths[:4])
    output_path = _output_root(output)
    if output_path.is_relative_to(capture_path) or output_path.is_relative_to(package_path):
        raise BaremetalModelError("output must not be nested inside a saved capture or package")
    device_path = None
    if device is not None:
        from merlin.llvmlower.device_build import DeviceRouting

        if not isinstance(device, DeviceRouting):
            raise BaremetalModelError("device must be an explicit DeviceRouting")
        device_path = Path(device.package_dir)
        if device_path.is_symlink():
            raise BaremetalModelError("selected device package must not be a symlink")
        device_path = device_path.resolve(strict=True)
        if output_path.is_relative_to(device_path):
            raise BaremetalModelError("output must not be nested inside the selected device package")
    output_path.mkdir(parents=True)
    receipt: dict[str, Any] = {
        "schema": "merlin.baremetal-saved-model.v1",
        "status": "failed",
        "scope": "one saved whole-model ELF; execution is not a Phase 0 release or static-routing proof",
        "execution_route": "host_baseline" if device is None else "device_requested_dispatch_unverified",
        "inputs": {
            "capture": str(capture_path),
            "package": str(package_path),
            "board_catalog": str(catalog_path),
            "board": board,
            "dts": str(dts_path),
            "target": target,
            "run": run,
            "reference_file": reference_file,
            "rtl_facts": str(rtl_facts) if rtl_facts else None,
            "arena_mb": arena_mb,
            "timeout_s": timeout_s,
        },
    }
    try:
        capture_tree = MI.strict_tree_sha256(capture_path)
        package_tree = MI.strict_tree_sha256(package_path)
        catalog_sha, dts_sha = _sha(catalog_path), _sha(dts_path)
        golden = _saved_reference(capture_path, reference_file, execution=run != "none")
        boards = load_boards(catalog_path)
        selected = boards.get(board)
        if selected is None or selected.target != target:
            raise BaremetalModelError("selected board is absent or names a different target")
        if selected.flow != FLOW_BAREMETAL or selected.console != CONSOLE_HTIF:
            raise BaremetalModelError("selected board is not a bare-metal HTIF board")
        if selected.harts != 1 or selected.code_reserve is None or selected.host_dts_sha256 is None:
            raise BaremetalModelError("complete saved-model executor needs one hart, code reserve and pinned host DTS")
        if selected.dram_base != spike_model.DRAM_BASE:
            raise BaremetalModelError("selected board DRAM differs from the current bare-metal runner base")
        pkg = registry.load_rvv_package(package_path)
        if pkg.backend not in {"scalar", "rvv"}:
            raise BaremetalModelError(f"unsupported whole-model host package backend {pkg.backend!r}")
        isas = require_host_isa_dts(pkg.cflags, dts_path, expected_sha256=selected.host_dts_sha256)
        if len(isas) != selected.harts or len(set(isas)) != 1:
            raise BaremetalModelError("DTS CPU ISA roster does not match selected board")
        marches = [flag.removeprefix("-march=") for flag in pkg.cflags if flag.startswith("-march=")]
        if len(marches) != 1:
            raise BaremetalModelError("host package must declare exactly one -march")
        if device is not None:
            device_tree = MI.strict_tree_sha256(device_path)
            receipt["inputs"]["device"] = {
                "name": device.device,
                "package": str(device.package_dir),
                "package_tree": device_tree,
            }
        else:
            device_tree = None
        native = run in {"gsim", "verilator"}
        if native:
            if rtl_facts is None or not selected.rtl_sim_config:
                raise BaremetalModelError("native RTL execution requires explicit facts and board RTL config")
            facts = MI.selected_firrtl(rtl_facts, target=target, config=selected.rtl_sim_config)
            ambient = os.environ.get("MERLIN_RTL_FACTS", "").strip()
            if ambient and _sha(Path(ambient)) != facts["sha256"]:
                raise BaremetalModelError("ambient RTL facts differ from explicitly selected native facts")
            backend, selection, revalidate, prepare = _native_engine(target, run, facts)
            receipt["inputs"]["rtl_facts_identity"] = facts
        else:
            backend, selection, revalidate, prepare = None, None, None, None
        if run == "spike" and not spike_backend.available():
            raise BaremetalModelError("Spike simulator or bare-metal cross-toolchain is unavailable")
        receipt["inputs"].update(
            {
                "capture_tree": capture_tree,
                "package_tree": package_tree,
                "board_catalog_sha256": catalog_sha,
                "dts_sha256": dts_sha,
                "host_isa": isas[0],
                "simulator_isa": marches[0],
                "golden_sha256": _sha(capture_path / reference_file) if golden is not None else None,
            }
        )
        if selection is not None:
            receipt["engine_selection"] = selection
        build_options = {"math_archive_symbols": math_archive_symbols} if math_archive_symbols is not None else {}
        if golden is not None and golden.size > 4096:
            build_options["output_dump_cap"] = int(golden.size)
        built = spike_model.build(
            capture_path,
            output_path / "build",
            arena_mb=arena_mb,
            dram_base=selected.dram_base,
            dram_bytes=selected.dram_bytes,
            code_reserve=selected.code_reserve,
            int8_compute=bool(pkg.is_int8),
            backend=pkg.backend,
            rvv_schedule=pkg.schedule_text if pkg.backend == "rvv" else None,
            cflags_override=list(pkg.cflags),
            vlen=selected.vlen if pkg.backend == "rvv" else None,
            console=selected.console,
            device=device,
            **build_options,
        )
        if not isinstance(built.get("build_hash"), str) or not built["build_hash"]:
            raise BaremetalModelError("bare-metal build returned no citable build hash")
        if (
            not isinstance(built.get("index_lowering"), dict)
            or built["index_lowering"].get("schema") != "merlin.selected-index-lowering.v1"
        ):
            raise BaremetalModelError("bare-metal build returned no selected index-lowering record")
        elf = Path(built["elf"])
        if not elf.is_file() or elf.is_symlink() or not elf.resolve().is_relative_to(output_path):
            raise BaremetalModelError("bare-metal build returned no safe ELF in its output")
        elf_sha = _sha(elf)
        arch = spike_model.arch_extensions(elf)
        MI.require_elf_isa_supported(arch, isas[0], require_scalar=False)
        receipt["output"] = {
            "elf": str(elf),
            "elf_sha256": elf_sha,
            "elf_arch_extensions": arch,
            "build_hash": built.get("build_hash"),
            "index_lowering": built["index_lowering"],
            "matrix_routing": built.get("matrix_routing"),
        }
        # Bind the producer's record, rather than minting a new command history.
        # Its consumer independently verifies completed commands and input bytes.
        from merlin.llvmlower.compilation_recipe import FILENAME as RECIPE_FILENAME

        recipe = elf.parent / RECIPE_FILENAME
        if recipe.exists():
            if recipe.is_symlink() or not recipe.is_file():
                raise BaremetalModelError("bare-metal compilation recipe is absent or indirect")
            receipt["output"]["compilation_recipe"] = {"path": str(recipe), "sha256": _sha(recipe)}
        from merlin.llvmlower.device_offload import SIDECAR_NAME

        sidecar = output_path / "build" / SIDECAR_NAME
        if sidecar.is_file() and not sidecar.is_symlink():
            receipt["output"]["device_sidecar"] = {"path": str(sidecar), "sha256": _sha(sidecar)}
        if device is not None and "device_sidecar" not in receipt["output"]:
            raise BaremetalModelError("device was selected but the build emitted no device dispatch sidecar")
        if device is not None:
            receipt["output"]["device_dispatch_evidence"] = "static_sidecar_only; execution not established"
        if run != "none":
            if backend is None:
                try:
                    result = spike_model.run(
                        elf,
                        harts=selected.harts,
                        mem_bytes=built["mem_bytes"],
                        isa=marches[0],
                        timeout=timeout_s,
                        vlen=built.get("vlen"),
                    )
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
                    _retain_simulator_failure_output(exc, output_path, receipt)
                    raise
                console = str(result.get("console", ""))
            else:
                revalidate()
                command = None
                if run == "gsim":
                    if not callable(prepare):
                        raise BaremetalModelError("GSIM backend has no byte-bound command preparer")
                    command = prepare(
                        elf, expected_elf_sha256=elf_sha, expected_engine_provenance=selection["citation"]
                    )
                    command_check = command.revalidate()
                    command_evidence = command.to_evidence()
                    receipt["output"]["native_command"] = {
                        "schema": command_evidence["schema"],
                        "command_sha256": command_check["command_sha256"],
                        "emulator_argv": command_evidence["emulator_argv"],
                        "max_cycles": command_evidence["max_cycles"],
                    }
                if run == "gsim":
                    from merlin.targetgen.rtl_engine_policy import gsim_runtime_slot

                    slot = gsim_runtime_slot(wait_timeout_s=timeout_s)
                else:
                    slot = nullcontext()
                with slot:
                    try:
                        console = backend.run_elf(elf, simulator=run, timeout=timeout_s)
                    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
                        _retain_simulator_failure_output(exc, output_path, receipt)
                        raise
                if command is not None:
                    command.revalidate()
                revalidate()
            console_path = output_path / "console.log"
            console_path.write_text(console, encoding="utf-8")
            receipt["output"].update({"console": str(console_path), "console_sha256": _sha(console_path)})
            from merlin.runtime.backends.spike_model import parse_console

            observed = parse_console(console)
            if not isinstance(observed, dict):
                raise BaremetalModelError("whole-model console parser returned no structured result")
            if observed.get("metrics", {}).get("build_hash") != built.get("build_hash"):
                raise BaremetalModelError("whole-model console does not identify the linked build")
            if observed.get("metrics", {}).get("memref_rank_mismatch") != 0:
                raise BaremetalModelError("whole-model console lacks a clean memref-rank diagnostic")
            values = np.asarray(observed["outputs"], dtype=np.float32).reshape(-1)
            if values.size != golden.size or not np.isfinite(values).all():
                raise BaremetalModelError("whole-model OUT is partial or nonfinite")
            if not np.array_equal(values.view(np.uint32), golden.view(np.uint32)):
                mismatched = int(np.count_nonzero(values.view(np.uint32) != golden.view(np.uint32)))
                raise BaremetalModelError(
                    f"whole-model output differs from declared reference in {mismatched} elements"
                )
            receipt["output"].update(
                {"elements": int(golden.size), "mismatched_elements": 0, "metrics": observed.get("metrics") or {}}
            )
        if (
            MI.strict_tree_sha256(capture_path) != capture_tree
            or MI.strict_tree_sha256(package_path) != package_tree
            or _sha(catalog_path) != catalog_sha
            or _sha(dts_path) != dts_sha
            or _sha(elf) != elf_sha
        ):
            raise BaremetalModelError("an input or linked ELF changed during compilation/execution")
        if device is not None and MI.strict_tree_sha256(device_path) != device_tree:
            raise BaremetalModelError("selected device package changed during compilation/execution")
        if sidecar.is_file() and _sha(sidecar) != receipt["output"]["device_sidecar"]["sha256"]:
            raise BaremetalModelError("device dispatch sidecar changed during compilation/execution")
        if selection is not None and _native_engine(target, run, facts)[1] != selection:
            raise BaremetalModelError("selected native RTL engine changed during execution")
        if native and MI.selected_firrtl(rtl_facts, target=target, config=selected.rtl_sim_config) != facts:
            raise BaremetalModelError("selected RTL facts or source FIRRTL changed during execution")
        receipt["status"] = "compiled" if run == "none" else "verified_complete_output"
    except Exception as exc:
        receipt["failure"] = f"{type(exc).__name__}: {exc}"
        (output_path / "baremetal_model.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        raise
    (output_path / "baremetal_model.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
