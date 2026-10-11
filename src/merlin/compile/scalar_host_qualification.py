"""Qualify one saved scalar host capture on a selected simulator, without promoting host support.

This is intentionally narrower than a target capability declaration: one pinned
package, one capture, one board/DTS, one complete single-output numerical run.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from merlin.common.paths import out_dir
from merlin.compile.host_lane import dts_string_values, require_host_isa_dts
from merlin.compile.model_execution_inputs import (
    ModelExecutionInputError as ScalarHostQualificationError,
)
from merlin.compile.model_execution_inputs import (
    file_sha256 as _file_sha256,
)
from merlin.compile.model_execution_inputs import (
    native_engine as _native_engine,
)
from merlin.compile.model_execution_inputs import (
    require_elf_isa_supported as _require_elf_isa_supported,
)
from merlin.compile.model_execution_inputs import (
    selected_firrtl as _selected_firrtl,
)
from merlin.compile.model_execution_inputs import (
    strict_tree_sha256 as _strict_tree_sha256,
)
from merlin.mining.registry import load_rvv_package
from merlin.runtime.backends import spike_model
from merlin.runtime.boards import CONSOLE_HTIF, FLOW_BAREMETAL, load_boards

_OUTPUT_DTYPES = {
    np.dtype(np.float32): ("f32", np.uint32),
    np.dtype(np.int64): ("i64", np.uint64),
    np.dtype(np.bool_): ("i1", np.uint8),
}


def _single_output_golden(capture: Path) -> np.ndarray:
    required = (
        "model.mlir",
        "weights.safetensors",
        "weights.safetensors.manifest.json",
        "inputs.npz",
        "input_order.json",
        "golden.npy",
        "capture_receipt.json",
    )
    missing = [name for name in required if not (capture / name).is_file()]
    if missing:
        raise ScalarHostQualificationError(f"saved capture is incomplete: missing {missing}")
    if (capture / "goldens.npz").exists() or (capture / "output_order.json").exists():
        raise ScalarHostQualificationError("multi-output captures are unsupported by the bare-metal OUT parser")
    capture_receipt = json.loads((capture / "capture_receipt.json").read_text(encoding="utf-8"))
    if capture_receipt.get("schema") != "m2m.capture-receipt.v1" or not (
        capture_receipt.get("materialized_abi") or {}
    ).get("complete"):
        raise ScalarHostQualificationError("saved capture has no complete materialized ABI receipt")
    recorded = capture_receipt.get("artifacts") or {}
    if any(
        (recorded.get(name) or {}).get("sha256") != _file_sha256(capture / name)
        for name in required
        if name != "capture_receipt.json"
    ):
        raise ScalarHostQualificationError("saved capture artifacts disagree with its capture receipt")
    golden = np.load(capture / "golden.npy", allow_pickle=False)
    if golden.dtype not in _OUTPUT_DTYPES or not np.isfinite(golden).all():
        raise ScalarHostQualificationError("single-output golden must be finite float32, int64 or bool")
    if golden.dtype == np.bool_ and np.any(golden.view(np.uint8) > 1):
        raise ScalarHostQualificationError("Boolean golden must contain canonical zero/one bytes")
    from merlin.common.mlir_query import forward_signature

    _arguments, outputs = forward_signature(capture / "model.mlir")
    expected_dtype = _OUTPUT_DTYPES[golden.dtype][0]
    if len(outputs) != 1 or outputs[0] != (list(golden.shape), expected_dtype):
        raise ScalarHostQualificationError("golden shape/dtype differs from the single parsed model output ABI")
    if golden.size > 4096:
        raise ScalarHostQualificationError("bare-metal OUT prints at most 4096 elements; full output is required")
    return golden.reshape(-1)


def _output_root(path: str | Path) -> Path:
    output = Path(path).resolve()
    generated = out_dir().resolve()
    if not output.is_relative_to(generated) or output == generated:
        raise ScalarHostQualificationError(f"qualification output must be a new directory below {generated}")
    if output.exists():
        raise ScalarHostQualificationError(f"qualification output already exists: {output}")
    return output


def _one_out_bits(console: str) -> np.ndarray:
    try:
        return spike_model.parse_console(console)["raw_output_bits"]
    except spike_model.SpikeModelError as exc:
        raise ScalarHostQualificationError(str(exc)) from exc


def qualify(
    *,
    capture: str | Path,
    package: str | Path,
    board_catalog: str | Path,
    board: str,
    dts: str | Path,
    output: str | Path,
    arena_mb: int,
    timeout_s: int = 300,
    simulator: str = "spike",
    rtl_facts: str | Path | None = None,
) -> dict[str, Any]:
    """Build, run and gate one complete saved-capture result; emit a bounded receipt.

    The receipt qualifies only this finite run and these bytes. It does not mutate or
    rebind a reviewed target descriptor, capability declaration, or package manifest.
    """
    if arena_mb < 1 or timeout_s < 1:
        raise ScalarHostQualificationError("arena_mb and timeout_s must be positive")
    if simulator not in {"spike", "gsim", "verilator"}:
        raise ScalarHostQualificationError(f"unsupported simulator {simulator!r}")
    lexical_inputs = [Path(capture), Path(package), Path(board_catalog), Path(dts)]
    if any(path.is_symlink() for path in lexical_inputs):
        raise ScalarHostQualificationError("qualification inputs must not be symlinks")
    capture_path, package_path = lexical_inputs[0].resolve(), lexical_inputs[1].resolve()
    catalog_path, dts_path = lexical_inputs[2].resolve(), lexical_inputs[3].resolve()
    output_path = _output_root(output)
    if output_path.is_relative_to(capture_path) or output_path.is_relative_to(package_path):
        raise ScalarHostQualificationError("qualification output is nested inside an input tree")
    capture_tree = _strict_tree_sha256(capture_path)
    package_tree = _strict_tree_sha256(package_path)
    golden = _single_output_golden(capture_path)
    output_dtype, unsigned = _OUTPUT_DTYPES[golden.dtype]
    if not catalog_path.is_file() or catalog_path.is_symlink():
        raise ScalarHostQualificationError(f"board catalog is missing or a symlink: {catalog_path}")
    catalog_sha256 = _file_sha256(catalog_path)
    boards = load_boards(catalog_path)
    selected = boards.get(board)
    if selected is None:
        raise ScalarHostQualificationError(f"board {board!r} is not in {catalog_path}")
    if selected.flow != FLOW_BAREMETAL or selected.console != CONSOLE_HTIF:
        raise ScalarHostQualificationError("host qualification supports only bare-metal HTIF boards")
    if selected.harts != 1 or selected.dram_base != spike_model.DRAM_BASE:
        raise ScalarHostQualificationError("host qualification requires one hart and DRAM at 0x80000000")
    if selected.code_reserve is None or selected.host_dts_sha256 is None:
        raise ScalarHostQualificationError("board lacks code reserve or a pinned elaborated host DTS")
    loaded = load_rvv_package(package_path)
    if loaded.backend != "scalar" or loaded.dtype_strategy != "int8_w8a8":
        raise ScalarHostQualificationError("only the generic scalar int8 package is supported")
    marches = [flag.removeprefix("-march=") for flag in loaded.cflags if flag.startswith("-march=")]
    if len(marches) != 1:
        raise ScalarHostQualificationError("scalar package must declare exactly one -march")
    simulator_isa = marches[0]
    isas = require_host_isa_dts(loaded.cflags, dts_path, expected_sha256=selected.host_dts_sha256)
    cpu_nodes = dts_string_values(dts_path.read_text(encoding="utf-8"), "device_type").count("cpu")
    if cpu_nodes != selected.harts or len(isas) != cpu_nodes or len(set(isas)) != 1:
        raise ScalarHostQualificationError("DTS CPU ISA roster does not match the board hart count")
    if dts_path.is_symlink():
        raise ScalarHostQualificationError("selected host DTS must not be a symlink")
    dts_sha256 = _file_sha256(dts_path)
    if dts_sha256 != selected.host_dts_sha256:
        raise ScalarHostQualificationError("selected DTS changed after ISA validation")
    native = None
    selected_firrtl = None
    if simulator != "spike":
        if rtl_facts is None or not selected.target or not selected.rtl_sim_config:
            raise ScalarHostQualificationError("native RTL qualification requires board target/config and --rtl-facts")
        selected_firrtl = _selected_firrtl(rtl_facts, target=selected.target, config=selected.rtl_sim_config)
        native = _native_engine(selected.target, simulator, selected_firrtl)

    receipt: dict[str, Any] = {
        "schema": (
            "merlin.scalar_host_spike_qualification.v1"
            if simulator == "spike"
            else "merlin.scalar_host_native_qualification.v1"
        ),
        "status": "failed",
        "scope": (
            "one saved capture on Spike; not elaborated-RTL execution or an operation-support declaration"
            if simulator == "spike"
            else "one saved capture on one selected elaborated-RTL engine; not blanket host-operation approval"
        ),
        "inputs": {
            "capture": str(capture_path),
            "capture_tree": capture_tree,
            "package": str(package_path),
            "package_tree": package_tree,
            "board_catalog": str(catalog_path),
            "board_catalog_sha256": catalog_sha256,
            "board": board,
            "board_target": selected.target,
            "board_dram_base": selected.dram_base,
            "board_dram_bytes": selected.dram_bytes,
            "board_harts": selected.harts,
            "board_code_reserve": selected.code_reserve,
            "dts": str(dts_path),
            "dts_sha256": dts_sha256,
            "host_isa": isas[0],
            "simulator_isa": simulator_isa,
            "golden_sha256": _file_sha256(capture_path / "golden.npy"),
            "output_dtype": output_dtype,
            "arena_mb": arena_mb,
            "timeout_s": timeout_s,
            "code_reserve_policy": "spike_model default: fixed base plus generated static IO",
            "simulator": simulator,
        },
    }
    if selected_firrtl is not None and native is not None:
        receipt["inputs"]["selected_rtl_facts"] = selected_firrtl
        receipt["engine"] = {"simulator": simulator, "fidelity": "elaborated_rtl", "citation": native[1]}
    output_path.mkdir(parents=True)
    try:
        from merlin.llvmlower.quant_scope import FEATURE as captured_quantization_scope

        built = spike_model.build(
            capture_path,
            output_path / "build",
            arena_mb=arena_mb,
            dram_base=selected.dram_base,
            dram_bytes=selected.dram_bytes,
            int8_compute=True,
            features=frozenset({captured_quantization_scope}),
            backend="scalar",
            rvv_schedule=None,
            cflags_override=loaded.cflags,
            console=selected.console,
        )
        elf = Path(built["elf"])
        if not elf.is_file():
            raise ScalarHostQualificationError("bare-metal build returned no ELF")
        elf_sha256 = _file_sha256(elf)
        elf_arch = spike_model.arch_extensions(elf)
        _require_elf_isa_supported(elf_arch, isas[0])
        if native is None:
            result = spike_model.run(
                elf,
                harts=selected.harts,
                mem_bytes=built["mem_bytes"],
                isa=simulator_isa,
                timeout=timeout_s,
            )
            console = str(result.get("console", ""))
            metrics = result.get("metrics") or {}
            observed = np.asarray(result["outputs"]).reshape(-1)
        else:
            backend, citation, revalidate, prepare = native
            revalidate()
            prepared = None
            if prepare is not None:
                prepared = prepare(elf, expected_elf_sha256=elf_sha256, expected_engine_provenance=citation)
                prepared.revalidate()
            from merlin.runtime.backends.chipyard_rocc_bound import BoundRoCCBackend

            if simulator == "gsim" and not isinstance(backend, BoundRoCCBackend):
                from merlin.targetgen.rtl_engine_policy import gsim_runtime_slot

                slot = gsim_runtime_slot(wait_timeout_s=timeout_s)
            else:
                slot = nullcontext()
            with slot:
                console = backend.run_elf(elf, simulator=simulator, timeout=timeout_s)
            if not isinstance(console, str) or sum(line.split() == ["DONE"] for line in console.splitlines()) != 1:
                raise ScalarHostQualificationError("native backend did not return one completed DONE protocol")
            if prepared is not None:
                prepared.revalidate()
            revalidate()
            parsed = spike_model.parse_console(console)
            metrics = parsed["metrics"]
            observed = np.asarray(parsed["outputs"]).reshape(-1)
        console_path = output_path / ("spike-console.log" if native is None else "rtl-console.log")
        console_path.write_text(console, encoding="utf-8")
        out_bits = _one_out_bits(console)
        if observed.dtype != golden.dtype:
            raise ScalarHostQualificationError(f"output dtype {observed.dtype} differs from golden {golden.dtype}")
        if observed.size != golden.size or not np.isfinite(observed).all():
            raise ScalarHostQualificationError(
                f"partial or nonfinite output: observed {observed.size}, golden {golden.size} elements"
            )
        if (
            out_bits.dtype != unsigned
            or out_bits.size != observed.size
            or not np.array_equal(out_bits, observed.view(unsigned))
        ):
            raise ScalarHostQualificationError("parsed output disagrees with its raw OUT bits")
        mismatch = int(np.count_nonzero(observed.view(unsigned) != golden.view(unsigned)))
        max_abs = (
            max(
                (abs(int(actual) - int(expected)) for actual, expected in zip(observed, golden, strict=True)), default=0
            )
            if golden.dtype == np.int64
            else float(np.max(np.abs(observed.astype(np.float64) - golden.astype(np.float64)), initial=0))
        )
        receipt["numerical_comparison"] = {
            "comparison": "exact_output_bits",
            "elements": int(golden.size),
            "element_dtype": output_dtype,
            "mismatched_elements": mismatch,
            "max_absolute_error": max_abs,
            "console": str(console_path),
            "console_sha256": _file_sha256(console_path),
        }
        if metrics.get("memref_rank_mismatch") != 0:
            raise ScalarHostQualificationError("missing or nonzero memref_rank_mismatch diagnostic")
        if mismatch:
            raise ScalarHostQualificationError(f"exact golden mismatch in {mismatch}/{golden.size} elements")
        if _strict_tree_sha256(capture_path) != capture_tree or _strict_tree_sha256(package_path) != package_tree:
            raise ScalarHostQualificationError("capture or package changed during qualification")
        if _file_sha256(catalog_path) != catalog_sha256 or _file_sha256(dts_path) != dts_sha256:
            raise ScalarHostQualificationError("board catalog or DTS changed during qualification")
        if _file_sha256(elf) != elf_sha256:
            raise ScalarHostQualificationError("ELF changed after execution")
        if selected_firrtl is not None and native is not None:
            if (
                _selected_firrtl(
                    selected_firrtl["path"], target=selected_firrtl["target"], config=selected_firrtl["config"]
                )
                != selected_firrtl
            ):
                raise ScalarHostQualificationError("selected RTL facts or FIRRTL changed during qualification")
            native[2]()
        receipt.update(
            {
                "status": "passed_saved_capture_spike" if simulator == "spike" else "passed_saved_capture_native_rtl",
                "output": {
                    "elf": str(elf),
                    "elf_sha256": elf_sha256,
                    "elf_arch_extensions": elf_arch,
                    **(
                        {"spike_console": str(console_path), "spike_console_sha256": _file_sha256(console_path)}
                        if native is None
                        else {"rtl_console": str(console_path), "rtl_console_sha256": _file_sha256(console_path)}
                    ),
                    "elements": int(golden.size),
                    "element_dtype": output_dtype,
                    "numerical_output_checked": bool(golden.size),
                    "comparison": "exact_output_bits"
                    if golden.size
                    else "empty_output_ABI_and_completed_execution_only",
                    "mismatched_elements": mismatch,
                    "max_absolute_error": max_abs,
                    "metrics": metrics,
                    "build_hash": built.get("build_hash"),
                },
            }
        )
    except Exception as exc:
        receipt["failure"] = f"{type(exc).__name__}: {exc}"
        (output_path / "qualification.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        raise
    (output_path / "qualification.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--board-catalog", required=True, type=Path)
    parser.add_argument("--board", required=True)
    parser.add_argument("--dts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arena-mb", required=True, type=int)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--simulator", choices=("spike", "gsim", "verilator"), default="spike")
    parser.add_argument("--rtl-facts", type=Path, help="selected FIRRTL facts, required for native RTL engines")
    args = parser.parse_args()
    result = qualify(
        capture=args.capture,
        package=args.package,
        board_catalog=args.board_catalog,
        board=args.board,
        dts=args.dts,
        output=args.output,
        arena_mb=args.arena_mb,
        timeout_s=args.timeout_s,
        simulator=args.simulator,
        rtl_facts=args.rtl_facts,
    )
    print(json.dumps({"status": result["status"], "receipt": str(Path(args.output) / "qualification.json")}))


if __name__ == "__main__":
    main()
