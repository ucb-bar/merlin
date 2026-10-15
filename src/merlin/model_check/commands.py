"""The subcommands behind ``python -m merlin.model_check`` (contract: ``docs/reference/model_check.md``)."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from merlin.model_check import bundle as B
from merlin.model_check import replay as RP
from merlin.model_check.cli import register
from merlin.model_check.models import load as load_models

#: ``fast``: per unique kernel, at most this many calls on at most this many inputs.
FAST_CALLS_PER_SYMBOL = 3
FAST_INPUTS = 2
#: One replay image carries at most this many operand bytes (they sit in the image's high region).
IMAGE_BYTES = 768 << 20


def _target(args) -> str:
    return load_models(args.models).target


def _bundle(args) -> B.Bundle:
    return B.Bundle(args.bundle) if args.bundle else B.Bundle.default(_target(args))


def _programs(model, args) -> list[str]:
    names = [p.name for p in model.programs]
    if getattr(args, "program", None):
        model.program(args.program)
        return [args.program]
    return names


def _bind_engine_inputs(args, manifest: dict[str, Any]) -> None:
    """Host package, contract and facts: explicit arguments, else what the bundle was built with."""
    engine = manifest.get("engine") or {}
    for key in ("host_package", "contract", "rtl_facts"):
        if getattr(args, key, None) is None and engine.get(key):
            setattr(args, key, Path(engine[key]))
    if args.host_package is None:
        raise B.BundleError("no host package: pass --host-package or use a bundle that records one")
    if args.contract is not None:
        os.environ["MERLIN_TARGET_CONTRACT"] = str(args.contract)
    if args.rtl_facts is not None:
        os.environ["MERLIN_RTL_FACTS"] = str(args.rtl_facts)


@register("bundle-build")
def bundle_build(args: argparse.Namespace, report: dict) -> dict:
    models = load_models(args.models)
    model = models.model(args.model)
    bundle = _bundle(args)
    keep = None if args.keep == "all" else (lambda _symbol, ordinal: ordinal < FAST_CALLS_PER_SYMBOL)
    engine = {"host_package": str(args.host_package), "contract": str(args.contract), "rtl_facts": str(args.rtl_facts)}
    record = B.build(
        bundle,
        model,
        target=models.target,
        captures=args.captures,
        programs=_programs(model, args),
        keep_call=keep,
        engine=engine,
    )
    report["bundle"] = {"path": str(bundle.root)}
    report["programs"] = [{"program": name, **row} for name, row in record["programs"].items()]
    report["gates"]["consistency"] = {
        "status": "pass" if all(row["consistent"] for row in record["programs"].values()) else "fail",
        "detail": "device-rewritten program == INT8 reference, bit for bit, on the capture's inputs",
    }
    return report


@register("census")
def census(args: argparse.Namespace, report: dict) -> dict:
    import json

    models = load_models(args.models)
    model = models.model(args.model)
    bundle = _bundle(args)
    for program in _programs(model, args):
        path = bundle.program_dir(model.name, program) / "census.json"
        data = json.loads(path.read_text())
        report["programs"].append(
            {"program": program, "census": {k: v for k, v in data.items() if k != "contractions"}}
        )
    report["gates"]["census"] = {"status": "pass", "detail": "measurement only"}
    return report


def _cases(
    bundle: B.Bundle, model: str, program: str, inputs: Sequence[str], *, group: str | None, all_calls: bool
) -> list[RP.ReplayCase]:
    """Replay cases for ``program``: every call, or (fast) a few per symbol on a few inputs."""
    blobs = bundle.blobs(model)
    catalog = bundle.catalog(model, program)
    region_of = {}
    for row in catalog.sidecar.get("routed") or ():
        region_of.setdefault(str(row.get("symbol")), set()).add(str(row.get("source_region_id")))
    cases: list[RP.ReplayCase] = []
    for input_id in inputs:
        doc = bundle.calls(model, input_id, program)
        for call in doc["calls"]:
            symbol = call["symbol"]
            if group is not None and group != symbol and group not in region_of.get(symbol, ()):
                continue
            if not all_calls and call["ordinal"] >= FAST_CALLS_PER_SYMBOL:
                continue
            cases.append(
                RP.ReplayCase(
                    symbol=symbol,
                    label=f"{input_id}/{program}/{symbol}#{call['ordinal']}",
                    roles=tuple(call["roles"]),
                    operands=tuple(blobs.get(k) for k in call["operands"]),
                    expected=tuple(blobs.get(k) for k in call["expected"]),
                )
            )
    return cases


def _chunks(cases: Sequence[RP.ReplayCase]) -> list[list[RP.ReplayCase]]:
    chunks: list[list[RP.ReplayCase]] = [[]]
    size = 0
    for case in cases:
        nbytes = sum(int(a.nbytes) for a in case.operands)
        if chunks[-1] and size + nbytes > IMAGE_BYTES:
            chunks.append([])
            size = 0
        chunks[-1].append(case)
        size += nbytes
    return [c for c in chunks if c]


def _replay_program(args, bundle, model, program, inputs, *, group, all_calls, work: Path) -> dict[str, Any]:
    catalog = bundle.catalog(model.name, program)
    cases = _cases(bundle, model.name, program, inputs, group=group, all_calls=all_calls)
    symbols = sorted({c.symbol for c in cases})
    toolchain = RP.toolchain_for(args.host_package)
    started = time.monotonic()
    built = RP.build_kernels(
        catalog.target,
        catalog.sidecar,
        symbols,
        package_dir=args.submission,
        workdir=work / "device",
        operand_dtype=catalog.routing["operand_dtype"],
        accum_dtype=catalog.routing["accum_dtype"],
        numeric_policy=catalog.routing["numeric_policy"],
        toolchain=toolchain,
    )
    build_seconds = time.monotonic() - started
    refused = {str(sym): str(why) for sym, why in built.skipped}
    objects = [o for o in built.objects if Path(o).name != "device_shim.o"]
    verdicts: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    runnable = [c for c in cases if c.symbol in built.kernels]
    for index, chunk in enumerate(_chunks(runnable)):
        image_dir = work / f"image_{index}"
        image_dir.mkdir(parents=True, exist_ok=True)
        main_c, data_s, plan = RP._write_image_sources(image_dir, chunk, built.kernels, catalog.interfaces)  # noqa: SLF001
        elf = RP.link_image(image_dir, main_c, data_s, objects, toolchain)
        if args.engine == "spike":
            console, wall = RP.run_image(
                catalog.target, elf, dram_base=args.dram_base, dram_bytes=args.dram_bytes, timeout=args.timeout_s
            )
        else:
            raise RP.ReplayError("replay on gsim is not wired yet in this version")
        verdicts += RP.judge_console(console, plan)
        images.append({"cases": len(chunk), "engine": args.engine, "seconds": round(wall, 1), "elf": str(elf)})
    per_symbol: dict[str, dict[str, Any]] = {}
    for verdict in verdicts:
        row = per_symbol.setdefault(
            verdict["symbol"], {"symbol": verdict["symbol"], "cases": 0, "exact": 0, "mismatches": []}
        )
        row["cases"] += 1
        row["exact"] += int(verdict["exact"])
        if not verdict["exact"]:
            row["mismatches"].append(verdict["label"])
    kernels = []
    for symbol in symbols:
        if symbol in refused:
            kernels.append({"symbol": symbol, "status": "refused", "detail": refused[symbol][:300]})
            continue
        row = per_symbol.get(symbol, {"symbol": symbol, "cases": 0, "exact": 0, "mismatches": []})
        status = "pass" if row["cases"] and row["exact"] == row["cases"] else "fail"
        kernels.append({**row, "status": status, "mismatches": row["mismatches"][:5]})
    return {
        "program": program,
        "build": {"seconds": round(build_seconds, 1), "built": len(built.kernels), "refused": len(refused)},
        "kernels": kernels,
        "images": images,
        "refused": refused,
    }


def _replay(args, report, *, all_calls: bool, group: str | None) -> dict:
    models = load_models(args.models)
    model = models.model(args.model)
    bundle = _bundle(args)
    manifest = bundle.manifest()
    _bind_engine_inputs(args, manifest)
    report["bundle"] = {"path": str(bundle.root), "merlin_commit": manifest.get("merlin_commit")}
    inputs = bundle.inputs(model.name, args.split)
    if not inputs:
        raise B.BundleError(f"bundle has no {args.split} inputs for {model.name}")
    if not all_calls:
        inputs = inputs[:FAST_INPUTS]
    from merlin.common.paths import build_dir

    work = Path(args.work or (build_dir() / "model_check")) / f"{model.name}-{int(time.time())}"
    programs = _programs(model, args)
    for program in programs:
        report["programs"].append(
            _replay_program(args, bundle, model, program, inputs, group=group, all_calls=all_calls, work=work / program)
        )
    kernels = [k for p in report["programs"] for k in p["kernels"]]
    refused = [k for k in kernels if k["status"] == "refused"]
    failed = [k for k in kernels if k["status"] == "fail"]
    report["gates"]["build"] = {
        "status": "pass" if not refused else "fail",
        "detail": f"{len(kernels) - len(refused)}/{len(kernels)} kernels built",
    }
    report["gates"]["kernel_exact"] = {
        "status": "pass" if kernels and not failed and not refused else "fail",
        "partial": not all_calls,
        "detail": f"{sum(k.get('exact', 0) for k in kernels)}/{sum(k.get('cases', 0) for k in kernels)} calls "
        f"bit-exact on {len(inputs)} input(s); {len(failed)} kernel(s) differ",
    }
    for kernel in failed:
        report["failures"].append(
            {
                "kind": "gate",
                "gate": "kernel_exact",
                "symbol": kernel["symbol"],
                "detail": ", ".join(kernel["mismatches"]),
            }
        )
    return report


@register("replay-kernel")
def replay_kernel(args: argparse.Namespace, report: dict) -> dict:
    report = _replay(args, report, all_calls=bool(args.all_groups) or args.split != "visible", group=args.group)
    report["gates"]["kernel_exact"]["partial"] = False
    return report


@register("check")
def check(args: argparse.Namespace, report: dict) -> dict:
    if args.mode != "fast":
        raise RP.ReplayError("check --mode full is not wired yet in this version")
    from merlin.model_check.approximations import declared

    approximations = declared(Path(args.submission))
    report = _replay(args, report, all_calls=False, group=None)
    report["gates"]["kernel_exact"]["partial"] = False
    if approximations:
        report["gates"]["model_numerics"] = {
            "status": "fail",
            "detail": "declared approximate kernels need the model replay, not wired in this version",
        }
    else:
        exact = report["gates"]["kernel_exact"]["status"] == "pass"
        report["gates"]["model_numerics"] = {
            "status": "pass" if exact else "fail",
            "detail": "every kernel exact, so the model computes its INT8 reference"
            if exact
            else "a kernel differs from its reference; the model gate cannot pass",
        }
    return report


@register("inputs-prepare")
def inputs_prepare(args: argparse.Namespace, report: dict) -> dict:
    from merlin.common.paths import build_dir
    from merlin.model_check import inputs as INP

    models = load_models(args.models)
    models.model(args.model)
    sources = args.sources or models.path.with_name("sources.yaml")
    extra: dict[str, Any] = {}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            raise INP.InputError(f"--set expects KEY=VALUE, got {item!r}")
        extra[key] = Path(value) if key.endswith("_python") else value
    record = INP.prepare(
        args.model,
        sources,
        data_root=args.data,
        public=sources.parent,
        private=args.private,
        python=args.python,
        scratch=build_dir() / "model_check" / "inputs",
        **extra,
    )
    report["programs"].append({"inputs": {k: len(v) for k, v in record.items() if isinstance(v, dict)}})
    size = INP.stored_bytes(sources.parent)
    report["gates"]["inputs"] = {
        "status": "pass" if size <= INP.PUBLIC_BUDGET else "fail",
        "detail": f"public example holds {size} bytes (budget {INP.PUBLIC_BUDGET})",
    }
    return report
