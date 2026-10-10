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
    if golden.size > 4096:
        raise BaremetalModelError("complete OUT observability is limited to 4096 elements; use run='none'")
    return golden.reshape(-1)


def _full_references(capture: Path, reference_file: str | None) -> list[tuple[str, np.ndarray | None]]:
    """``[(name, reference or None)]`` in forward-result order for a full-readback execution.

    A capture that states every result (``output_order.json`` with one array each in ``goldens.npz``)
    references them all. Otherwise result zero is held to the one explicit ``reference_file`` and any
    further result is recorded as having no capture reference (a caller may hold it to an independent
    host execution of the same program). Every file must be bound by the capture receipt or lie in the
    capture tree whose digest this compilation records."""
    from merlin.llvmlower.c_runtime import _out_specs

    recorded = json.loads((capture / "capture_receipt.json").read_text()).get("artifacts") or {}

    def present(name: str, *, receipt_bound: bool) -> Path:
        path = capture / name
        if not path.is_file() or path.is_symlink():
            raise BaremetalModelError(f"reference {name!r} is absent from the capture")
        if receipt_bound and (recorded.get(name) or {}).get("sha256") != _sha(path):
            raise BaremetalModelError(f"reference {name!r} is not bound by the capture receipt")
        return path

    results = len(_out_specs(capture / "model.mlir"))
    if (capture / "output_order.json").exists() or (capture / "goldens.npz").exists():
        names = [str(x) for x in json.loads(present("output_order.json", receipt_bound=False).read_text())]
        archive = np.load(present("goldens.npz", receipt_bound=False), allow_pickle=False)
        if len(names) != results or len(set(names)) != len(names) or set(names) != set(archive.files):
            raise BaremetalModelError("output_order.json and goldens.npz disagree with the forward's results")
        return [(name, np.asarray(archive[name])) for name in names]
    if reference_file is None or Path(reference_file).name != reference_file or not reference_file.endswith(".npy"):
        raise BaremetalModelError("execution needs one explicit in-bundle .npy reference_file for result zero")
    first = np.load(present(reference_file, receipt_bound=True), allow_pickle=False)
    return [(Path(reference_file).stem, first), *((f"out{i}", None) for i in range(1, results))]


def _agreement(value: np.ndarray, reference: np.ndarray, tolerance: dict[str, float] | None) -> dict[str, Any]:
    """Exact (bit for bit, or integer equality) without a tolerance; element-wise atol/rtol with one."""
    reference = np.asarray(reference)
    if value.size != reference.size:
        return {"passed": False, "note": f"printed {value.size} of {reference.size} elements"}
    flat, ref = value.reshape(-1), reference.reshape(-1)
    if tolerance is None:
        if flat.dtype == np.float32 and ref.dtype == np.float32:
            differ = int(np.count_nonzero(flat.view(np.uint32) != ref.view(np.uint32)))
        elif flat.dtype.kind in "iub" and ref.dtype.kind in "iub":
            differ = int(np.count_nonzero(flat.astype(np.int64) != ref.astype(np.int64)))
        else:
            return {"passed": False, "note": f"{flat.dtype} cannot be exact against {ref.dtype}"}
        return {"passed": differ == 0, "mismatched_elements": differ, "of": int(ref.size)}
    from merlin.perf.float_accuracy import compare

    agreement = compare(flat.astype(np.float64), ref.astype(np.float64), tolerance["atol"], tolerance["rtol"])
    return {**agreement, "passed": agreement["within"] == agreement["of"]}


def _spike_extension_backend(target: str):
    """The target backend that runs ELFs on the functional simulator WITH its accelerator extension.

    ``None`` for a target whose backend declares no extension (a host-only image then runs on the
    plain simulator); an extension that is declared but cannot be resolved refuses the run."""
    from merlin.runtime.backends import base

    try:
        backend = base.get_backend(target)
    except KeyError:
        return None
    if not callable(getattr(backend, "spike_extension", None)) or not callable(getattr(backend, "run_elf", None)):
        return None
    return backend


def _judge_prefix(console: str, built: dict, golden: np.ndarray, output: Path, receipt: dict) -> None:
    """The historical one-output protocol: the printed prefix, bit for bit, against the reference."""
    from merlin.runtime.backends.spike_model import parse_console

    if isinstance(console, bytes):
        console = console.decode("utf-8", errors="replace")
    console_path = output / "console.log"
    console_path.write_text(console, encoding="utf-8")
    receipt["output"].update({"console": str(console_path), "console_sha256": _sha(console_path)})
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
        raise BaremetalModelError(f"whole-model output differs from declared reference in {mismatched} elements")
    receipt["output"].update(
        {"elements": int(golden.size), "mismatched_elements": 0, "metrics": observed.get("metrics") or {}}
    )


def console_text_metrics(text: str) -> dict[str, str]:
    """``METRIC <name> <value>`` lines as strings (a build hash is hex, not an integer)."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "METRIC":
            out[parts[1]] = parts[2]
    return out


def read_full_outputs(console: bytes, capture: Path, built_hash: str | None) -> tuple[list[np.ndarray], dict]:
    """Every forward result decoded from a full-readback console, plus its text metrics.

    Refuses a console that does not identify ``built_hash`` or reports a memref-rank refusal."""
    from merlin.llvmlower.c_runtime import _out_specs
    from merlin.runtime.out_bin import binary_console_diagnostics, parse_binary_console_details
    from merlin.runtime.whole_model_readback import decode_outputs

    if not isinstance(console, bytes):
        raise BaremetalModelError("a full-readback console must be read as raw bytes")
    outputs, _numeric, frames = parse_binary_console_details(console)
    metrics = console_text_metrics(binary_console_diagnostics(console).decode("utf-8", errors="replace"))
    if built_hash is not None and metrics.get("build_hash") != built_hash:
        raise BaremetalModelError("whole-model console does not identify the linked build")
    if metrics.get("memref_rank_mismatch") != "0":
        raise BaremetalModelError("whole-model console lacks a clean memref-rank diagnostic")
    values = decode_outputs(outputs, frames, _out_specs(capture / "model.mlir"))
    return values, metrics


def _judge_full_readback(console, built, capture, references, tolerance, output: Path) -> dict[str, Any]:
    console_path = output / "console.bin"
    console_path.write_bytes(console if isinstance(console, bytes) else console.encode("utf-8"))
    values, metrics = read_full_outputs(console, capture, built.get("build_hash"))
    np.savez(output / "outputs.npz", **{f"out{i}": value for i, value in enumerate(values)})
    record: dict[str, Any] = {
        "console": str(console_path),
        "console_sha256": _sha(console_path),
        "outputs_npz": str(output / "outputs.npz"),
        "metrics": metrics,
    }
    if references is None:
        return record
    if len(references) != len(values):
        raise BaremetalModelError(f"the forward has {len(values)} results and the capture states {len(references)}")
    per_output = []
    for (name, reference), value in zip(references, values, strict=True):
        if reference is None:
            per_output.append({"name": name, "passed": True, "status": "no_capture_reference"})
            continue
        per_output.append({"name": name, "status": "compared", **_agreement(value, reference, tolerance)})
    record["per_output"] = per_output
    failed = [row["name"] for row in per_output if not row["passed"]]
    if failed:
        raise BaremetalModelError(f"whole-model results differ from the capture's references: {failed}")
    return record


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
    readback: str = "prefix",
    group_profile: bool = False,
    tolerance: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compile one saved model; ``none`` makes no execution or numerical claim.

    ``readback="full"`` builds the image to print every forward result complete
    (:mod:`merlin.runtime.whole_model_readback`); execution then holds each result to the capture's
    own reference (``goldens.npz`` + ``output_order.json``, or the one ``reference_file``), exactly or
    within ``tolerance`` (``{atol, rtol}``). ``group_profile`` brackets every routed device-group call
    with cycle counts and records them. A ``spike`` run goes through the target backend with its
    accelerator extension when the backend declares one.

    ``reference_file`` is required for execution because a weight-only capture's
    ``golden.npy`` is not necessarily the reference for a W8A8 execution.  This
    initial executor supports one complete float32 OUT (at most 4096 values).
    """
    if run not in {"none", "spike", "gsim", "verilator"}:
        raise BaremetalModelError(f"unsupported bare-metal run {run!r}")
    if readback not in {"prefix", "full"}:
        raise BaremetalModelError(f"unsupported readback {readback!r}; choose prefix or full")
    if tolerance is not None and (
        not isinstance(tolerance, dict)
        or set(tolerance) != {"atol", "rtol"}
        or any(isinstance(v, bool) or not isinstance(v, int | float) or v < 0 for v in tolerance.values())
    ):
        raise BaremetalModelError("tolerance declares exactly non-negative atol and rtol")
    full = readback == "full"
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
            "readback": readback,
            "group_profile": bool(group_profile),
            "tolerance": tolerance,
        },
    }
    try:
        capture_tree = MI.strict_tree_sha256(capture_path)
        package_tree = MI.strict_tree_sha256(package_path)
        catalog_sha, dts_sha = _sha(catalog_path), _sha(dts_path)
        golden = _saved_reference(capture_path, reference_file, execution=run != "none" and not full)
        references = _full_references(capture_path, reference_file) if full and run != "none" else None
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
                "golden_sha256": _sha(capture_path / reference_file)
                if golden is not None or (references is not None and reference_file)
                else None,
            }
        )
        if selection is not None:
            receipt["engine_selection"] = selection
        build_options = {"math_archive_symbols": math_archive_symbols} if math_archive_symbols is not None else {}
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
            full_readback=full,
            group_profile=bool(group_profile),
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
        receipt["output"]["readback"] = readback
        if run != "none":
            extension = _spike_extension_backend(target) if run == "spike" else None
            if backend is None and extension is not None:
                # The functional model WITH the target's accelerator extension: a device-routed image
                # issues accelerator instructions the plain simulator traps on.
                flags, libdir = extension.spike_extension()
                receipt["output"]["spike_extension"] = {"flags": list(flags), "library_dir": str(libdir)}
                try:
                    console = extension.run_elf(elf, simulator="spike", timeout=timeout_s, capture_bytes=full)
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
                    _retain_simulator_failure_output(exc, output_path, receipt)
                    raise
            elif backend is None and full:
                if device is not None:
                    raise BaremetalModelError("a device-routed image needs the target's spike extension")
                try:
                    console = spike_model.run_raw(
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
            elif backend is None:
                if device is not None:
                    raise BaremetalModelError("a device-routed image needs the target's spike extension")
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
                        console = backend.run_elf(elf, simulator=run, timeout=timeout_s, capture_bytes=full)
                    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
                        _retain_simulator_failure_output(exc, output_path, receipt)
                        raise
                if command is not None:
                    command.revalidate()
                revalidate()
            if full:
                receipt["output"].update(
                    _judge_full_readback(console, built, capture_path, references, tolerance, output_path)
                )
            else:
                _judge_prefix(console, built, golden, output_path, receipt)
            if group_profile:
                from merlin.runtime.whole_model_readback import parse_group_profile

                text = console.decode("utf-8", errors="replace") if isinstance(console, bytes) else console
                if full:
                    from merlin.runtime.out_bin import binary_console_diagnostics

                    text = binary_console_diagnostics(console).decode("utf-8", errors="replace")
                profile = parse_group_profile(text)
                if profile is None:
                    raise BaremetalModelError("the group-profiled image printed no group profile")
                (output_path / "group_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
                receipt["output"]["group_profile"] = {
                    "path": str(output_path / "group_profile.json"),
                    "calls": profile["calls"],
                    "dropped": profile["dropped"],
                    "group_cycles": sum(row["cycles"] for row in profile["groups"]),
                    "gap_cycles": sum(row["gap_before"] for row in profile["groups"]) + profile["tail_gap"],
                }
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
