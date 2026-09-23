#!/usr/bin/env python3
"""Bench OUR generated schedules against an EXTERNAL kernel bundle, on the same design, layer by layer.

The question: at the level of one kernel, how does a schedule this repo generates compare with a
hand- or auto-tuned kernel somebody else shipped for this accelerator? To be an answer rather than a
number, every row has to survive three failure modes that were found inside the external bundle itself:

1. **Isolated flatters embedded.** A kernel with a one-time host-side setup cost looks fast when it is
   measured twice and slow when it is called once. So EVERY row is measured under BOTH protocols --
   ``cold_single`` (the first invocation in a fresh program, which pays every one-time cost) and
   ``warm_then_measured`` (the steady-state invocation an isolated benchmark reports) -- and the row
   also carries the per-call-site figure those two imply for the call-site count the bundle declares.
2. **Silent fallback.** Each external kernel is wrapped in ``#if !<NAME>_FITS`` with a stock-library
   call in the false branch, so a design it does not fit measures the LIBRARY under the kernel's name.
   The rendered program carries a ``_Static_assert`` on that same macro, read off the kernel's own
   source (:mod:`external_kernel_bundle`), so the compiler refuses instead of substituting; the fit is
   independently cross-checked against the descriptor and the design's parameter header.
3. **Host work counted as the kernel.** The external kernels zero-pad on the HOST; the library and our
   routes pad in hardware. Nothing makes those windows equal, so each row DECLARES what is inside its
   measured window instead of implying they are the same.

Shapes are never typed here: the bundle's descriptor SELECTS which layers to bench, and the layer's
full contract (batch, padding, activation, requant scale, multiplicity) comes from the model capture,
so a row is the same layer the library/package/schedule tables measure.

    MERLIN_OUT_ROOT="$PWD/out" PYTHONPATH=merlin/python \\
      .venv/bin/python merlin/experiments/gemmini_perf_bench/scripts/external_kernel_table.py \\
      --bundle <external kernel bundle> --scales out/artifacts/.../spec_scales.json \\
      --target gemmini --design-pin gemmini_gsim_model_serialclk --slots 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from external_kernel_bundle import (  # noqa: E402
    GEOMETRY_MACROS,
    ExternalKernel,
    find_header,
    fit_evidence,
    geometry_agreement,
    load_bundle,
)
from layer_library_table import unique_layers  # noqa: E402

#: The symbol the external kernel is called through. The harness calls one function on the four
#: operand pointers it declares, so the bundle's own signature is adapted by a wrapper, not by
#: reshaping the harness (which would stop the arms from sharing a window).
EXTERNAL_SYMBOL = "xk_external_layer"

#: What is inside a measured window, per arm. Declared, never implied: the external kernels pad on the
#: host and the library/compiler routes pad in hardware, so these windows are NOT the same work.
WINDOW = {
    "external": ("host_pad_buffer_zero_once", "host_zero_pad_copy", "accelerator_conv"),
    "library": ("accelerator_conv",),
    "package": ("accelerator_conv", "package_declared_host_buffers"),
    # Measured on another engine, in-model: the window is the whole group, host sequencing included.
    # Wider than the library window, so the figure is conservative against it rather than flattering.
    "compute_group": ("accelerator_conv", "host_group_sequencing"),
}

#: The one-time member of a window: present in a ``cold_single`` measurement, absent from a warm one.
ONE_TIME_IN_WINDOW = {"external": ("host_pad_buffer_zero_once",)}

PROTOCOLS = ("cold_single", "warm_then_measured")

#: Comparisons this bench will NOT make, and why. They are stated in the product because the numbers
#: they involve are sitting right beside these rows in the same bundle, and a reader who does not find
#: the refusal written down will make the comparison themselves.
NOT_COMPARED = (
    {
        "comparison": "our measured cycles vs the cycle figures the bundle itself quotes for these kernels",
        "why": "the bundle's figures come from its own search harness at that harness's batch and with "
        "an identity requantization, on a different engine. Every row here is measured at the batch "
        "and the fused epilogue the model capture declares, on this design's pinned engine. The two "
        "are not the same measurement and putting them in one column would be a category error -- "
        "which is why the external kernel is BUILT AND RUN here rather than cited.",
    },
    {
        "comparison": "a per-kernel row vs the bundle's whole-model cycle count",
        "why": "the bundle's whole-model figure covers the entire network, of which these kernels are "
        "a few call sites; a kernel row cannot be scaled up to it or read as a share of it.",
    },
    {
        "comparison": "the package arm's cycles vs the external/library arms', operand for operand",
        "why": "the package arm's interface convolution declares no bias, so it is measured with a zero "
        "bias while the other arms carry the capture's. Same shape and same schedule question, but it "
        "is a contract difference and it is stated rather than absorbed.",
    },
    {
        "comparison": "the compute-group arm's cycles as if they were measured on this bench's device",
        "why": "that arm ran in-model on another engine entirely, so it is kept in its own block with "
        "its engine, job and status on the row. Its comparison against the external kernel is TWO HOPS "
        "-- through the stock-library control measured on both engines -- because the external kernel "
        "was never run on that engine. Its comparison against the library is one hop and needs no "
        "calibration, since both sides of that one came from the same pair of runs.",
    },
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def c_float32(value: float) -> str:
    """The float32 nearest ``value`` as an exact C99 hex literal (never a decimal that re-rounds)."""
    import numpy as np

    return float(np.float32(value)).hex() + "f"


def host_pad_copy_bytes(spec: dict) -> int:
    """Bytes the host copies per invocation to build the kernel's zero-padded input image.

    The whole int8 input tensor, since only the border of the padded buffer is reused between calls.
    Derived from the layer's own extents -- nothing about the bundle's buffer sizes is assumed.
    """
    return int(spec["batch"]) * int(spec["in_dim"]) ** 2 * int(spec["in_channels"])


def design_geometry(target: str) -> tuple[Path, dict]:
    """The design's geometry macros, read from the header the target's own build recipe compiles with."""
    from merlin.runtime.backends import base as backends

    recipe = backends.harness_build_recipe(target)
    header, defines = find_header(recipe.include_roots, GEOMETRY_MACROS)
    return header, {name: defines[name] for name in GEOMETRY_MACROS}


def external_kernel_c(kernel: ExternalKernel, spec: dict) -> str:
    """C placed ahead of the harness ``main``: the bundle's header, the fit assertion, and a wrapper.

    The wrapper exists because the harness calls one function on ``(input, weights, bias, output)``;
    the bundle's entry also takes a batch, an activation and a requant scale. Those three come from the
    model capture's own contract for this layer, so the external kernel computes the same function the
    library and package arms do -- which is what makes the digest check meaningful for all of them.
    """
    act = "RELU" if spec["relu"] else "NO_ACTIVATION"
    return (
        f'#include "{kernel.header.name}"\n\n'
        f"/* Refuse rather than fall back: if this design did not fit the kernel, `#if !{kernel.guard.macro}`\n"
        f"   would silently compile `{kernel.guard.fallback_symbol}` under the external kernel's name. */\n"
        f'_Static_assert({kernel.guard.macro}, "{kernel.name} does not fit this design; '
        f'the bundle would have measured {kernel.guard.fallback_symbol} instead");\n\n'
        f"static void {EXTERNAL_SYMBOL}(elem_t *input, elem_t *weights, acc_t *bias, elem_t *output) {{\n"
        f"    {kernel.name}({int(spec['batch'])}, input, weights, bias, output,\n"
        f"        {act}, {c_float32(float(spec['scale']))});\n"
        f"}}\n"
    )


def _stage_bundle_headers(kernel: ExternalKernel, wd: Path) -> list[str]:
    """Copy the bundle headers the kernel needs beside the program, preserving their include shape.

    The kernel header includes its companions by a path relative to itself, so the bundle's ``src``
    layout is reproduced under the work directory rather than flattened; the target's own headers stay
    on the include path and win for everything the bundle does not ship.
    """
    src = kernel.header.parent
    staged: list[str] = []
    pending = [(kernel.header, kernel.header.name)]
    while pending:
        source, rel = pending.pop()
        if rel in staged:
            continue
        dest = wd / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        staged.append(rel)
        for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 2 or parts[0] != "#include" or not parts[1].startswith('"'):
                continue
            want = parts[1].strip('"')
            # The bundle may ship a header FLAT that its own sources include through a subdirectory
            # (the bundle's README says so). Resolve by the path as written first, then by base name,
            # and stage it at the path the including file asks for -- never at where it was found.
            found = src / want
            if not found.is_file():
                found = src / Path(want).name
            if found.is_file():
                pending.append((found, want))
    return staged


def measure_external(
    kernel: ExternalKernel,
    sig: str,
    row: dict,
    protocol: str,
    *,
    target: str,
    design_pin: str,
    workroot: Path,
    cache,
    contract_obj,
    max_cycles: int,
    timeout_s: float,
) -> dict:
    """One external-kernel row: build, run on the pinned engine, and check the digest off-device."""
    from merlin.perf.layer_bench import LayerKey, build_program, run_on_gsim
    from merlin.perf.layer_bench.reference import pack_operands
    from merlin.runtime.backends import base
    from merlin.targetgen import gsim_emulator

    backend = base.get_backend(target)
    label = "X" + _sha((kernel.name + protocol).encode())[:12]
    spec = {**row["spec"], "label": label, "scale": row["scale"], "seed": 1, "protocol": protocol}
    blob, offsets = pack_operands(spec, accumulator_dtype=contract_obj.accumulator_dtype)
    kernel_c = external_kernel_c(kernel, spec)
    source = backend.render_schedule_layer(
        spec,
        offsets=offsets,
        kernel_c=kernel_c,
        symbol=EXTERNAL_SYMBOL,
        arg_names=["input", "weights", "bias", "output"],
    )
    engine = gsim_emulator.citation(target, env_var=getattr(backend, "GSIM_EMU_ENV", None))
    key = LayerKey(
        target=target,
        design_pin=design_pin,
        engine_sha256=engine["binary_sha256"],
        group_signature=sig,
        contract_digest=contract_obj.digest(),
        schedule_digest=_sha((_sha(source.encode()) + _sha(blob)).encode()),
        emitter_digest=_sha(kernel.header.read_bytes()),
        harness_version=backend.LIBRARY_LAYER_HARNESS_VERSION + "+external",
        protocol=protocol,
    )
    cached = cache.get(key)
    if cached is not None:
        return {
            "arm": "external",
            "kernel": kernel.name,
            "protocol": protocol,
            # What the host moves on EVERY call so the accelerator can read a pre-padded image.
            # It is inside the measured window and is not work the other arms do at all, so the
            # figure sits beside the cycles instead of being left for the reader to reconstruct.
            "host_bytes_copied_per_call": host_pad_copy_bytes(spec),
            # The program itself, so a row can be read back to the C that produced it even
            # after the build tree is gone. It is rendered before the cache is consulted, so
            # a cached row carries the same text the measured one was built from.
            "source": source,
            "cached": True,
            **cached,
        }
    wd = workroot / key.digest()[:16]
    wd.mkdir(parents=True, exist_ok=True)
    staged = _stage_bundle_headers(kernel, wd)
    (wd / backend.LIBRARY_LAYER_OPERAND_BLOB).write_bytes(blob)
    (wd / "layer.c").write_text(source, encoding="utf-8")
    built = build_program([wd / "layer.c"], wd, target=target, max_loaded_bytes=None)
    run = run_on_gsim(built.elf, target=target, max_cycles=max_cycles, timeout_s=timeout_s, backdoor=True)
    recs = [r for r in run.records if r.label == label]
    payload = _payload(
        spec,
        row,
        run,
        recs,
        built,
        engine,
        contract_obj,
        extra={
            "kernel_symbol": kernel.name,
            "staged_headers": staged,
            "fit_macro": kernel.guard.macro,
            "fallback_symbol_refused": kernel.guard.fallback_symbol,
            "call_sites": kernel.call_sites,
            "window": window_for("external", protocol),
        },
    )
    if payload["status"] != "ok":
        payload["stderr_tail"] = run.stderr_tail
        payload["stdout_tail"] = run.stdout_tail
        return {
            "arm": "external",
            "kernel": kernel.name,
            "protocol": protocol,
            # What the host moves on EVERY call so the accelerator can read a pre-padded image.
            # It is inside the measured window and is not work the other arms do at all, so the
            # figure sits beside the cycles instead of being left for the reader to reconstruct.
            "host_bytes_copied_per_call": host_pad_copy_bytes(spec),
            # The program itself, so a row can be read back to the C that produced it even
            # after the build tree is gone. It is rendered before the cache is consulted, so
            # a cached row carries the same text the measured one was built from.
            "source": source,
            "cached": False,
            **payload,
        }
    (wd / backend.LIBRARY_LAYER_OPERAND_BLOB).unlink(missing_ok=True)
    return {
        "arm": "external",
        "kernel": kernel.name,
        "protocol": protocol,
        # What the host moves on EVERY call so the accelerator can read a pre-padded image.
        # It is inside the measured window and is not work the other arms do at all, so the
        # figure sits beside the cycles instead of being left for the reader to reconstruct.
        "host_bytes_copied_per_call": host_pad_copy_bytes(spec),
        # The program itself, so a row can be read back to the C that produced it even
        # after the build tree is gone. It is rendered before the cache is consulted, so
        # a cached row carries the same text the measured one was built from.
        "source": source,
        "cached": False,
        **cache.put(key, payload),
    }


def measure_library(
    kernel: ExternalKernel,
    sig: str,
    row: dict,
    protocol: str,
    *,
    target: str,
    design_pin: str,
    workroot: Path,
    cache,
    contract_obj,
    max_cycles: int,
    timeout_s: float,
) -> dict:
    """The same layer through the target's own library kernel -- which is also what the external
    kernel's guard would have fallen back to, so this row doubles as the fallback's cost."""
    from merlin.perf.layer_bench import LayerKey, build_program, run_on_gsim
    from merlin.perf.layer_bench.reference import pack_operands
    from merlin.runtime.backends import base
    from merlin.targetgen import gsim_emulator

    backend = base.get_backend(target)
    label = "L" + _sha((kernel.name + protocol).encode())[:12]
    spec = {**row["spec"], "label": label, "scale": row["scale"], "seed": 1, "protocol": protocol}
    blob, offsets = pack_operands(spec, accumulator_dtype=contract_obj.accumulator_dtype)
    source = backend.render_library_layer(spec, offsets=offsets)
    engine = gsim_emulator.citation(target, env_var=getattr(backend, "GSIM_EMU_ENV", None))
    key = LayerKey(
        target=target,
        design_pin=design_pin,
        engine_sha256=engine["binary_sha256"],
        group_signature=sig,
        contract_digest=contract_obj.digest(),
        schedule_digest=_sha(source.encode() + b"\0" + _sha(blob).encode()),
        emitter_digest=backend.library_layer_emitter_digest(),
        harness_version=backend.LIBRARY_LAYER_HARNESS_VERSION,
        protocol=protocol,
    )
    cached = cache.get(key)
    if cached is not None:
        return {
            "arm": "library",
            "kernel": kernel.name,
            "protocol": protocol,
            "source": source,
            "cached": True,
            **cached,
        }
    wd = workroot / key.digest()[:16]
    wd.mkdir(parents=True, exist_ok=True)
    (wd / backend.LIBRARY_LAYER_OPERAND_BLOB).write_bytes(blob)
    (wd / "layer.c").write_text(source, encoding="utf-8")
    built = build_program([wd / "layer.c"], wd, target=target, max_loaded_bytes=None)
    run = run_on_gsim(built.elf, target=target, max_cycles=max_cycles, timeout_s=timeout_s, backdoor=True)
    recs = [r for r in run.records if r.label == label]
    payload = _payload(
        spec,
        row,
        run,
        recs,
        built,
        engine,
        contract_obj,
        # The library symbol is not named here: it is the symbol the external kernel's OWN guard falls
        # back to, so this row is measurably the fallback the `_Static_assert` refused.
        extra={
            "kernel_symbol": kernel.guard.fallback_symbol,
            "call_sites": kernel.call_sites,
            "window": window_for("library", protocol),
        },
    )
    if payload["status"] != "ok":
        payload["stderr_tail"] = run.stderr_tail
        payload["stdout_tail"] = run.stdout_tail
        return {
            "arm": "library",
            "kernel": kernel.name,
            "protocol": protocol,
            "source": source,
            "cached": False,
            **payload,
        }
    (wd / backend.LIBRARY_LAYER_OPERAND_BLOB).unlink(missing_ok=True)
    return {
        "arm": "library",
        "kernel": kernel.name,
        "protocol": protocol,
        "source": source,
        "cached": False,
        **cache.put(key, payload),
    }


def window_for(arm: str, protocol: str) -> list[str]:
    """What the measured window of this arm contains under this protocol.

    A one-time member (an external kernel zeroing its host padding buffer on first use) is inside a
    ``cold_single`` window and outside a warm one. Reporting the window is the whole point: the arms do
    NOT measure the same work, and a table that printed only cycles would imply they did.
    """
    one_time = set(ONE_TIME_IN_WINDOW.get(arm, ()))
    return [w for w in WINDOW.get(arm, ()) if protocol == "cold_single" or w not in one_time]


def _payload(spec, row, run, recs, built, engine, contract_obj, *, extra: dict) -> dict:
    """The common receipt: what ran, how long it took, and whether it computed the declared function."""
    from merlin.perf.layer_bench.reference import expected_digest

    payload = {
        "spec": spec,
        "count": row["count"],
        "macs": row["macs"],
        "names": row["names"],
        "elf_sha256": built.elf_sha256,
        "loaded_bytes": built.loaded_bytes,
        "completed": run.completed,
        "wall_seconds": round(run.wall_seconds, 1),
        "load_path": run.load_path,
        "engine_cycles": run.finish.cycles if run.finish else None,
        "cycles": recs[0].cycles if len(recs) == 1 else None,
        "digest": recs[0].fields.get("digest") if len(recs) == 1 else None,
        "digest_expected": expected_digest(spec, contract_obj),
        "engine": engine,
        **extra,
    }
    payload["numerics"] = (
        "exact" if payload["digest"] is not None and payload["digest"] == payload["digest_expected"] else "mismatch"
    )
    if not run.completed:
        payload["status"] = "engine_did_not_finish"
    elif len(recs) != 1:
        payload["status"] = f"expected 1 record, got {len(recs)}"
    elif payload["numerics"] != "exact":
        payload["status"] = "numerics_mismatch"
    else:
        payload["status"] = "ok"
    return payload


def select_layers(bundle, scales: dict) -> tuple[list[tuple[ExternalKernel, str, dict]], list[dict]]:
    """Pair each LIVE external kernel with the capture's layer of that shape; refuse the rest, with why.

    The descriptor says which shapes the bundle claims; the capture says what the layer actually is
    (padding, activation, requant scale, how many times the model runs it). A kernel whose shape the
    capture does not contain cannot be compared at all -- there is no contract to hold both sides to.
    """
    layers = unique_layers(scales)
    paired, refused = [], []
    for kernel in bundle.kernels:
        if not kernel.live:
            refused.append(
                {
                    "kernel": kernel.name,
                    "why": "declared but never called by the bundle's own driver",
                    "call_sites": kernel.call_sites,
                    "status_in_descriptor": kernel.status,
                }
            )
            continue
        if kernel.guard is None:
            refused.append(
                {
                    "kernel": kernel.name,
                    "why": "carries no compile-time fit guard, so nothing would stop a stock-library "
                    "fallback from being measured under its name",
                    "detail": kernel.guard_error,
                }
            )
            continue
        want = {
            "op": "conv2d" if kernel.op == "conv2d" else kernel.op,
            "kernel": _dim(kernel.row.get("filter")),
            "in_channels": _int(kernel.row.get("in_ch")),
            "out_channels": _int(kernel.row.get("out_ch")),
            "stride": _int(kernel.row.get("stride")),
            "in_dim": _dim(kernel.row.get("spatial")),
        }
        hits = [(sig, row) for sig, row in layers.items() if all(row["spec"].get(k) == v for k, v in want.items())]
        if len(hits) != 1:
            refused.append(
                {
                    "kernel": kernel.name,
                    "why": f"the model capture holds {len(hits)} layers of this shape; exactly 1 is needed "
                    "to fix the contract (padding, activation, requant scale) both sides are held to",
                    "descriptor_shape": want,
                }
            )
            continue
        sig, row = hits[0]
        if row["count"] != kernel.call_sites:
            refused.append(
                {
                    "kernel": kernel.name,
                    "why": "the descriptor's call-site count and the capture's multiplicity disagree, so "
                    "the per-call-site figure cannot be formed",
                    "descriptor_call_sites": kernel.call_sites,
                    "capture_multiplicity": row["count"],
                }
            )
            continue
        paired.append((kernel, sig, row))
    return paired, refused


def _int(text) -> int | None:
    text = (text or "").strip()
    return int(text) if text.isdecimal() else None


def _dim(text) -> int | None:
    """The leading extent of a descriptor field like ``3x3`` or ``56x56`` (square shapes only)."""
    text = (text or "").strip()
    head, sep, tail = text.partition("x")
    if not sep or not head.isdecimal() or head != tail:
        return None
    return int(head)


#: The routes this repo can lower a layer through. Whether each one CAN express a given shape is
#: probed, never asserted: the probe calls the route's own admission test (or tries to import it), so
#: a route that grows the ability later stops being refused without anyone editing this list.
OUR_ROUTES = ("package", "schedule", "compute_group")


def route_reachability(spec: dict) -> dict:
    """For each of our routes, whether it can express this layer -- asked of the route itself.

    A route that cannot express the shape is a finding about the compiler, so it is recorded with the
    reason the route itself gives rather than omitted from the table.
    """
    out: dict[str, dict] = {}

    try:
        from layer_schedule_table import matmul_view

        view = matmul_view(spec)
        out["schedule"] = (
            {"reachable": True}
            if view is not None
            else {
                "reachable": False,
                "why": "the schedule route admits only layers with a matmul view; this layer is a "
                "convolution with a spatial window and padding, which it has no reference schedule for",
            }
        )
    except Exception as exc:  # noqa: BLE001 -- an unimportable route is a refusal, not a crash
        out["schedule"] = {"reachable": False, "why": f"{type(exc).__name__}: {exc}"}

    try:  # the compute-group route reaches the device through the package compiler, if it is present
        import importlib

        importlib.import_module("merlin.targetgen.group_capsules")
        out["compute_group"] = {"reachable": True}
    except Exception as exc:  # noqa: BLE001
        out["compute_group"] = {
            "reachable": False,
            "why": f"the compute-group route is not part of this checkout ({type(exc).__name__}: {exc})",
        }

    # The package route ADMITS the shape -- whether it then lowers it in bounded time is a separate
    # question, answered by the measurement and written back over this entry by :func:`apply_package_outcome`.
    out["package"] = {"reachable": True, "why": "admits the shape; whether it lowers is measured"}
    return out


def apply_package_outcome(routes: dict, rows: Sequence[dict]) -> None:
    """Replace the package route's "admits the shape" with what actually happened to it.

    Admitting a layer and producing code for it are different claims, and leaving the optimistic one
    standing beside a failed row is how a table says a route works when the run says otherwise.
    """
    for row in rows:
        verdict = (routes.get(row.get("kernel")) or {}).get("package")
        if verdict is None:
            continue
        if row.get("status") == "ok":
            verdict["why"] = f"lowered and measured: {row.get('cycles'):,} cycles"
        else:
            verdict["reachable"] = False
            verdict["why"] = f"admits the shape but did not produce code for it -- {row.get('status')}"


def package_rows(table_path: Path, sigs: dict) -> list[dict]:
    """Fold a ``layer_package_table.json`` in as the ``package`` arm, for the layers this bench paired.

    The package route is measured by its own harness, so its rows are ADOPTED rather than re-measured:
    the contract digest must match, and a row whose contract differs is dropped with the reason, never
    silently placed beside rows it is not comparable with.
    """
    doc = json.loads(Path(table_path).read_text(encoding="utf-8"))
    out = []
    for row in doc.get("rows", []):
        name = sigs.get(row.get("sig"))
        if name is None:
            continue
        out.append(
            {
                "arm": "package",
                "kernel": name,
                # The package harness measures one window; it is labelled by the protocol it used.
                "protocol": doc.get("protocol", "warm_then_measured"),
                "package": doc.get("package"),
                "package_digest": doc.get("package_digest"),
                "window": window_for("package", "warm_then_measured"),
                "contract_digest": doc.get("contract_digest"),
                # The interface conv carries no bias, so the package arm is measured with ZERO bias
                # while the external and library arms carry the capture's bias. Same shape, same
                # schedule question -- but it is a contract difference and it is stated, not buried.
                "operand_note": "bias is zero (the interface conv declares none)",
                **{k: row.get(k) for k in ("cycles", "numerics", "wall_seconds", "engine_cycles", "error")},
                "kernel_symbol": "(package-lowered)",
                "status": "ok" if row.get("numerics") == "exact" else (row.get("error") or "not exact"),
            }
        )
    return out


def write_programs(into: Path, rows: list[dict]) -> dict[str, str]:
    """Write each row's rendered C beside the table and move the text out of the row.

    Returns the map from row key to the path inside the product, so the JSON says where each row's
    program is without carrying kilobytes of C inline. A row whose source the measurement did not
    carry (an adopted one, say) simply has no entry -- it is not invented.
    """
    written: dict[str, str] = {}
    for row in rows:
        source = row.pop("source", None)
        if not source:
            continue
        name = f"{row.get('arm')}_{row.get('kernel')}_{row.get('protocol')}.c"
        into.mkdir(parents=True, exist_ok=True)
        (into / name).write_text(source, encoding="utf-8")
        relpath = f"{into.name}/{name}"
        row["program"] = relpath
        row["program_sha256"] = _sha(source.encode())
        written[name[:-2]] = relpath
    return written


def cross_engine_arm(args, paired, results: list[dict]) -> tuple[list[dict], list[dict]]:
    """Calibrated rows from another engine, each licensed by a control this bench measured itself.

    The licence is COMPUTED, not accepted: :mod:`cross_engine_rows` compares the foreign run's own
    stock-library control against this bench's library row for the same layer and refuses the row if
    the two engines disagree beyond the declared tolerance. A refusal is carried into the table with
    its reason, because "we could not license this comparison" is a result.
    """
    import cross_engine_rows

    ours = cross_engine_rows.load_run(args.cross_engine_ours)
    control = cross_engine_rows.load_run(args.cross_engine_control)
    capsules = json.loads(Path(args.cross_engine_groups).read_text(encoding="utf-8"))
    own_library = {
        r["kernel"]: r["cycles"]
        for r in results
        if r.get("arm") == "library" and r.get("protocol") == "warm_then_measured" and r.get("cycles")
    }
    rows, refusals = [], []
    for kernel, _, row in paired:
        own = own_library.get(kernel.name)
        if own is None:
            refusals.append(
                {
                    "kernel": kernel.name,
                    "arm": "compute_group",
                    "why": "this bench has no library control for the layer on its own engine, so a "
                    "measurement from another engine cannot be calibrated against it",
                }
            )
            continue
        try:
            calibrated = cross_engine_rows.calibrate(
                ours,
                control,
                group_capsules=capsules,
                layer=row["spec"],
                own_control_cycles=own,
                tolerance=args.cross_engine_tolerance,
            )
        except cross_engine_rows.NotLicensed as exc:
            refusals.append({"kernel": kernel.name, "arm": "compute_group", "why": str(exc)})
            continue
        calibrated.update(
            {
                "kernel": kernel.name,
                "protocol": "in_model",
                "call_sites": kernel.call_sites,
                "kernel_symbol": "(compute-group schedule)",
                "window": list(WINDOW["compute_group"]),
                "count": row["count"],
                "macs": row["macs"],
            }
        )
        calibrated["acc_scale"]["own"] = row["scale"]
        calibrated["acc_scale"]["identical"] = calibrated["acc_scale"]["foreign"] == row["scale"]
        rows.append(calibrated)
    return rows, refusals


def per_call_site(cold: int | None, warm: int | None, sites: int | None) -> float | None:
    """What one call site costs when the kernel is embedded: the first pays the one-time cost, the rest
    do not. Undefined without both measurements and a call-site count, and it is left undefined rather
    than approximated by whichever number happens to be present."""
    if cold is None or warm is None or not sites:
        return None
    return (cold + (sites - 1) * warm) / sites


def _provenance_line(block: dict) -> str:
    """Which device these cycles are about, spelled out where a reader will see it."""
    checks = block.get("artifact_checks") or {}
    if not checks:
        return "No declared artifact was verified for this run; the cycles below name no device revision."
    parts = [
        f"`{name}` {c.get('digest', '?')[:12]} ({'verified' if c.get('ok') else 'OFF-PIN'})"
        for name, c in sorted(checks.items())
    ]
    return "These cycles are about: " + ", ".join(parts) + "."


def _geometry_line(agreement: dict) -> str:
    """One sentence about whether the bundle's own header describes the machine we built against."""
    if not agreement.get("bundle_ships_a_parameter_header"):
        return "The bundle ships no parameter header of its own, so only the descriptor's footprint "
    if agreement.get("geometry_matches"):
        return (
            "The bundle ships its own parameter header and every geometry macro in it matches this "
            "design's, so the kernels' absolute scratchpad addresses were searched against the same "
            "machine. They are nonetheless compiled against the DESIGN's header here, not the bundle's."
        )
    return (
        "WARNING: the bundle's own parameter header disagrees with this design on "
        + ", ".join(f"{k} (bundle {a} vs design {b})" for k, (a, b) in agreement["differing_macros"].items())
        + ". The kernels' absolute scratchpad addresses were searched against a different machine."
    )


def render_markdown(doc: dict) -> str:
    """The table a reader can act on: both windows per row, and what the bench would not compare."""
    by_kernel: dict[str, dict] = {}
    for row in doc["rows"]:
        if row.get("arm") is None:
            continue
        by_kernel.setdefault(row["kernel"], {})[(row["arm"], row["protocol"])] = row
    out = [
        f"# External kernel bench -- {doc['target']} @ `{doc['design_pin']}`",
        "",
        f"Bundle `{doc['bundle']['root']}` declares {doc['bundle']['declared_kernels']} kernels; "
        f"{len(doc['bundle']['live_kernels'])} are live.",
        f"Design geometry read from `{doc['design']['params_header']}`: "
        + ", ".join(f"{k} {v}" for k, v in sorted(doc["design"]["geometry"].items()))
        + ".",
        "",
        _geometry_line(doc["bundle"].get("geometry_agreement") or {}),
        "",
        _provenance_line(doc.get("provenance") or {}),
        "",
        "Every row is measured twice on the same engine: `cold_single` is the FIRST invocation in a",
        "fresh program (it pays every one-time cost), `warm_then_measured` is the steady-state one an",
        "isolated kernel benchmark reports. `per call site` is what the pair implies when the kernel is",
        "embedded at the call-site count the bundle declares.",
        "",
    ]
    for name, arms in by_kernel.items():
        fit = doc["fit_evidence"].get(name, {})
        out += [
            f"## `{name}`",
            "",
            f"Fit asserted at compile time on `{fit.get('fit_macro')}`; the fallback it refuses is "
            f"`{fit.get('fallback_symbol')}`. Footprint {fit.get('spad_rows_declared_by_descriptor')} rows "
            f"(descriptor) = {fit.get('spad_rows_derived_from_source')} rows (source) against "
            f"{fit.get('spad_rows_available')} available.",
            "",
            "| arm | kernel symbol | window (cold) | cold cycles | warm cycles | per call site | numerics |",
            "|---|---|---|---:|---:|---:|---|",
        ]
        for arm in sorted({a for a, _ in arms if a != "compute_group"}):
            cold = arms.get((arm, "cold_single")) or {}
            warm = arms.get((arm, "warm_then_measured")) or {}
            sites = cold.get("call_sites") or warm.get("call_sites")
            amortized = per_call_site(cold.get("cycles"), warm.get("cycles"), sites)
            verdicts = {r.get("numerics") or r.get("status") or r.get("error") for r in (cold, warm) if r}
            out.append(
                f"| {arm} | `{cold.get('kernel_symbol') or warm.get('kernel_symbol')}` | "
                f"{', '.join(cold.get('window') or warm.get('window') or ['-'])} | "
                f"{_cyc(cold.get('cycles'), measured=bool(cold))} | "
                f"{_cyc(warm.get('cycles'), measured=bool(warm))} | "
                f"{f'{amortized:,.0f}' if amortized is not None else '-'} | "
                f"{'/'.join(sorted(str(v)[:90] for v in verdicts))} |"
            )
        out.append("")
        foreign = arms.get(("compute_group", "in_model"))
        if foreign:
            out += _cross_engine_block(foreign, arms)
    routes = doc.get("our_routes") or {}
    if routes:
        out += [
            "## What this bench does not compare",
            "",
            "Each of our routes was asked whether it can express the layer at all; a route that cannot",
            "is a fact about the compiler, so it is listed rather than left out of the table.",
            "",
            "| kernel | route | can express | why not |",
            "|---|---|---|---|",
        ]
        for name, per_route in routes.items():
            for route in OUR_ROUTES:
                verdict = per_route.get(route) or {}
                out.append(
                    # the full reason (a compiler command line, say) stays in the JSON; the cell
                    # carries the part a reader acts on
                    f"| `{name}` | {route} | {'yes' if verdict.get('reachable') else 'NO'} | "
                    f"{_clip(verdict.get('why', ''), 180)} |"
                )
        out.append("")
    if doc.get("not_compared"):
        out += ["", "The bench also declines these comparisons outright:", ""]
        out += [f"- **{n['comparison']}** -- {n['why']}" for n in doc["not_compared"]]
        out.append("")
    if doc["refused"]:
        out += ["## Refused kernels", "", "| kernel | why |", "|---|---|"]
        out += [f"| `{r.get('kernel')}` | {r.get('why')} |" for r in doc["refused"]]
        out.append("")
    return "\n".join(out) + "\n"


def _cross_engine_block(row: dict, arms: dict) -> list[str]:
    """The calibrated arm, kept in its OWN block with its device, status and licence on the row.

    Deliberately not merged into the table above. Those rows are direct measurements on this bench's
    own pinned device; this one is a different device, a different window and a two-hop comparison, and
    a reader who cannot see that distinction will cite it as if it were not.
    """
    cal = row["calibration"]
    ext_warm = (arms.get(("external", "warm_then_measured")) or {}).get("cycles")
    ext_cold = (arms.get(("external", "cold_single")) or {}).get("cycles")
    sites = row.get("call_sites")
    ext_site = per_call_site(ext_cold, ext_warm, sites)
    lines = [
        "**Measured on another engine, calibrated in.** This row is NOT a measurement on this bench's",
        f"device. It ran in-model on `{row['engine']}` (job {row['job_id']}, status `{row['status']}`),",
        f"over compute groups {', '.join(row['groups'])}, and its window is {row['measured_in']} --",
        "wider than the library window above, so it is conservative rather than flattering.",
        "",
        f"The licence is the shared stock-library control on this same layer: {cal['foreign_control_cycles']:,} "
        f"cycles there against {cal['own_control_cycles']:,} here, agreeing to "
        f"{100 * (cal['agreement_ratio'] - 1):.2f}% (declared tolerance {100 * cal['tolerance']:.2f}%).",
        "Every comparison below therefore runs through that control -- it is two hops, not a direct",
        "measurement against the external kernel, which was never run on this engine.",
        "",
        # `vs library` uses the control from the SAME foreign run, not this bench's: both were measured
        # on one engine in one window, so that ratio needs no calibration at all and is the stronger
        # evidence. Only the comparisons against the external kernel have to travel through the licence.
        "| | cycles | vs external, via the control (warm) | vs external, via the control (per call site) "
        "| vs library (same engine, no calibration) |",
        "|---|---:|---:|---:|---:|",
        f"| compute-group schedule | {row['cycles']:,}-{row['cycles_max']:,} | "
        f"{f'{ext_warm / row["cycles"]:.2f}x faster' if ext_warm else '-'} | "
        f"{f'{ext_site / row["cycles"]:.2f}x faster' if ext_site else '-'} | "
        f"{row['cycles'] / cal['foreign_control_cycles']:.3f}x |",
        "",
    ]
    caveats = [
        f"correctness: {row['oracle']}" if row.get("oracle") else "correctness: no oracle recorded for this run",
        "the two foreign runs produced identical per-group output checksums, so the compute-group "
        "schedule and the stock library computed the same function there",
    ]
    if not row.get("sealed"):
        caveats.append(
            f"status is `{row['status']}`, not `sealed` -- the run's bytes could not be bound to its "
            "result by content, which is this repo's own bar for a hardware verdict"
        )
    if not cal.get("control_has_own_oracle"):
        caveats.append(f"the control run (job {cal['control_job_id']}) records no correctness oracle of its own")
    scale = row.get("acc_scale") or {}
    if scale.get("foreign") is not None and not scale.get("identical"):
        caveats.append(
            f"the requant constant differs ({scale['foreign']} there, {scale['own']} here) -- same "
            f"epilogue {row.get('epilogue')}, so the same instructions and the same cycles, but the two "
            "sides are not computing the identical function"
        )
    lines += [f"- {c}" for c in caveats] + [""]
    return lines


def _clip(text: str, width: int) -> str:
    """Shorten for a table cell, and SAY it was shortened, so nobody reads a clipped reason as whole."""
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 4] + " ..."


def _cyc(value, *, measured: bool = True) -> str:
    """A cell that keeps "we ran it and it failed" apart from "this window was never run".

    An arm measured under one protocol only (the package arm is) must not read as a failure under the
    other -- a blank that looks like a failure is how a route gets blamed for a run nobody attempted.
    """
    if isinstance(value, int):
        return f"{value:,}"
    return "FAILED" if measured else "not run"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, type=Path, help="external kernel bundle root")
    ap.add_argument("--scales", required=True, type=Path, help="model capture with per-node requant scales")
    ap.add_argument("--target", required=True)
    ap.add_argument("--design-pin", required=True, help="hardware_pins.yaml artifact the engine models")
    ap.add_argument(
        "--verify-artifacts",
        nargs="*",
        default=(),
        help="further declared artifacts to verify by content before a cycle count is written down "
        "(e.g. the engine binary, whose own citation is also recorded per row)",
    )
    ap.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="write the table even if a declared artifact does not verify (the disagreement is recorded)",
    )
    ap.add_argument("--arms", default="external,library", help="comma-separated: external, library")
    ap.add_argument(
        "--cross-engine-ours", type=Path, default=None, help="measurement dir for OUR run on another engine"
    )
    ap.add_argument(
        "--cross-engine-control",
        type=Path,
        default=None,
        help="measurement dir for the STOCK-LIBRARY run on that same engine -- the control that licenses the row",
    )
    ap.add_argument("--cross-engine-groups", type=Path, default=None, help="group_capsules.json for that run")
    ap.add_argument(
        "--cross-engine-tolerance",
        type=float,
        default=0.02,
        help="how far the shared control may disagree across the two engines before the row is refused",
    )
    ap.add_argument(
        "--package-table",
        type=Path,
        default=None,
        help="a layer_package_table.json whose rows for these layers are folded in as the package arm",
    )
    ap.add_argument("--protocols", default=",".join(PROTOCOLS))
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--max-cycles", type=int, default=200_000_000)
    ap.add_argument("--timeout-s", type=float, default=7200)
    ap.add_argument("--out-dir", type=Path, default=None, help="write here instead of a versioned product")
    ap.add_argument("--version", type=int, default=1)
    args = ap.parse_args(argv)

    from merlin.common.artifacts import new_product
    from merlin.common.paths import artifacts_dir
    from merlin.perf.layer_bench import ReceiptCache
    from merlin.runtime.backends import base
    from merlin.sched.contract import contract

    backend = base.get_backend(args.target)
    contract_obj = contract("per_tensor_readout_v1", backend.readout_facts())
    params_header, geometry = design_geometry(args.target)
    bundle = load_bundle(args.bundle)
    scales = json.loads(args.scales.read_text(encoding="utf-8"))
    paired, refused = select_layers(bundle, scales)

    fits = {}
    for kernel, _, _ in paired:
        evidence = fit_evidence(kernel, geometry)
        fits[kernel.name] = evidence
        if not evidence["sources_agree"]:
            print(f"{kernel.name}: footprint sources disagree: {evidence}", file=sys.stderr)
            return 2
        if not evidence["fits"]:
            print(f"{kernel.name}: does not fit this design ({evidence})", file=sys.stderr)
            return 2

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    unknown = [p for p in protocols if p not in PROTOCOLS]
    if unknown:
        print(f"unknown protocol(s) {unknown}", file=sys.stderr)
        return 2

    if args.out_dir is not None:
        out_dir, product = Path(args.out_dir), None
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        product = new_product(
            "perf-bench",
            version=args.version,
            target=args.target,
            sources=[str(args.bundle), str(args.scales)],
            notes="apples-to-apples per-kernel bench of an external kernel bundle against this repo's routes",
        )
        out_dir = product.path
    cache = ReceiptCache(artifacts_dir() / "perf-bench" / args.target / "layer_cache")
    workroot = out_dir / "work"

    measurers = {"external": measure_external, "library": measure_library}
    jobs = [
        (measurers[arm], kernel, sig, row, protocol)
        for arm in arms
        for kernel, sig, row in paired
        for protocol in protocols
        if arm in measurers
    ]
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.slots)) as pool:
        futs = {
            pool.submit(
                fn,
                kernel,
                sig,
                row,
                protocol,
                target=args.target,
                design_pin=args.design_pin,
                workroot=workroot,
                cache=cache,
                contract_obj=contract_obj,
                max_cycles=args.max_cycles,
                timeout_s=args.timeout_s,
            ): (kernel.name, protocol)
            for fn, kernel, sig, row, protocol in jobs
        }
        for fut in as_completed(futs):
            name, protocol = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001 -- one failed arm is a row, not a crash
                res = {"kernel": name, "protocol": protocol, "error": f"{type(exc).__name__}: {exc}"}
            results.append(res)
            print(
                json.dumps(
                    {k: res.get(k) for k in ("arm", "kernel", "protocol", "cycles", "numerics", "status", "error")}
                ),
                flush=True,
            )

    routes = {kernel.name: route_reachability(row["spec"]) for kernel, _, row in paired}
    if args.cross_engine_ours and args.cross_engine_control and args.cross_engine_groups:
        calibrated, not_licensed = cross_engine_arm(args, paired, results)
        results.extend(calibrated)
        refused.extend(not_licensed)
        for row in calibrated:
            verdict = (routes.get(row["kernel"]) or {}).get("compute_group")
            if verdict is not None:
                verdict["reachable"] = True
                verdict["why"] = (
                    f"not in this checkout, but measured on {row['engine']} (job {row['job_id']}) and "
                    f"calibrated in through the shared library control to "
                    f"{100 * (row['calibration']['agreement_ratio'] - 1):.2f}%"
                )
    if args.package_table is not None:
        adopted = package_rows(args.package_table, {sig: kernel.name for kernel, sig, _ in paired})
        mismatched = [r for r in adopted if r.get("contract_digest") != contract_obj.digest()]
        if mismatched:
            print(
                f"{args.package_table}: measured under a different numerics contract; not comparable",
                file=sys.stderr,
            )
            return 2
        results.extend(adopted)
        apply_package_outcome(routes, adopted)

    # A cycle count is a claim about a device. Verify WHICH device before writing it down: the
    # elaboration the engine models, and the engine binary itself, each checked by content.
    from merlin.common import provenance

    checks = {name: provenance.verify_artifact(name) for name in (args.design_pin, *args.verify_artifacts)}
    off_pin = {name: check.to_dict() for name, check in checks.items() if not check.ok}
    if off_pin and not args.allow_unpinned:
        print(f"the engine's declared artifacts do not verify: {json.dumps(off_pin, default=str)}", file=sys.stderr)
        return 2

    doc = {
        "schema": "external_kernel_table_v1",
        "target": args.target,
        "design_pin": args.design_pin,
        "provenance": provenance.record(
            sources=[args.bundle, args.scales],
            extra={"artifact_checks": {name: check.to_dict() for name, check in checks.items()}},
        ),
        "design": {
            "params_header": str(params_header),
            "params_header_sha256": _sha(params_header.read_bytes()),
            "geometry": geometry,
        },
        "bundle": {
            "root": str(bundle.root),
            "descriptor": str(bundle.descriptor),
            "descriptor_sha256": bundle.descriptor_sha256,
            "manifest": bundle.manifest_verified,
            "headers": bundle.headers,
            "declared_kernels": len(bundle.kernels),
            "live_kernels": [k.name for k in bundle.live_kernels()],
            "geometry_agreement": geometry_agreement(bundle, geometry),
        },
        "capture": {"path": str(args.scales), "sha256": _sha(args.scales.read_bytes())},
        "contract_digest": contract_obj.digest(),
        "fit_evidence": fits,
        "window": {arm: {p: window_for(arm, p) for p in protocols} for arm in arms},
        "our_routes": routes,
        "package_table": str(args.package_table) if args.package_table else None,
        "not_compared": [dict(n) for n in NOT_COMPARED],
        "refused": refused,
        "rows": results,
    }
    # The program that produced each row, kept beside the table. A build tree is scratch and gets
    # pruned; a row whose C nobody can read again is a number with no way back to what it measured.
    programs = write_programs(out_dir / "programs", results)
    doc["programs"] = programs
    (out_dir / "external_kernel_table.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    (out_dir / "external_kernel_table.md").write_text(render_markdown(doc), encoding="utf-8")
    if product is not None:
        product.add_artifact("external_kernel_table.json")
        product.add_artifact("external_kernel_table.md")
        for relpath in programs.values():
            product.add_artifact(relpath)
        product.write_manifest()
    print(f"wrote {out_dir}")
    bad = [r for r in results if r.get("status") != "ok"]
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
