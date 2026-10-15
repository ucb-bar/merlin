"""The subcommands behind ``python -m merlin.model_check`` (contract: ``docs/reference/model_check.md``)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from merlin.model_check import bundle as B
from merlin.model_check import metrics as M
from merlin.model_check import replay as RP
from merlin.model_check import variants as VAR
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
    from merlin.model_check import binding as BD

    binding = BD.write(BD.receipt_for(args), bundle.root / "binding.json")
    report["binding"] = {k: binding[k] for k in ("receipt_sha256", "status")}
    record = B.build(
        bundle,
        model,
        target=models.target,
        captures=args.captures,
        programs=_programs(model, args),
        keep_call=keep,
        engine=engine,
        binding=binding,
    )
    report["bundle"] = {"path": str(bundle.root)}
    report["programs"] = [{"program": name, **row} for name, row in record["programs"].items()]
    report["gates"]["consistency"] = {
        "status": "pass" if all(row["consistent"] for row in record["programs"].values()) else "fail",
        "detail": "device-rewritten program == INT8 reference, bit for bit, on the capture's inputs",
    }
    return report


@register("binding")
def binding(args: argparse.Namespace, report: dict) -> dict:
    from merlin.model_check import binding as BD

    receipt = BD.receipt_for(args)
    report["binding"] = {**BD.record(receipt), "receipt": receipt}
    report["gates"]["binding"] = BD.gate(receipt)
    report["failures"] += BD.failures(receipt)
    return report


@register("variants-build")
def variants_build(args: argparse.Namespace, report: dict) -> dict:
    model = load_models(args.models).model(args.model)
    bundle = _bundle(args)
    record = VAR.build(bundle, model.name, _programs(model, args), seed=args.seed)
    report["bundle"] = {"path": str(bundle.root)}
    report["programs"] = [{"program": name, "variants": row} for name, row in record["programs"].items()]
    device = [row for row in record["programs"].values() if not row.get("host_only")]
    report["gates"]["variants"] = {
        "status": "pass" if device and all(row["variants"] for row in device) else "fail",
        "detail": f"{sum(r['variants'] for r in device)} variant(s) of {sum(r['kernels'] for r in device)} "
        f"unique kernel(s); {sum(r['skipped'] for r in device)} class(es) unreachable",
    }
    return report


@register("census")
def census(args: argparse.Namespace, report: dict) -> dict:
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
        path = bundle.root / model / input_id / program / "calls.json"
        if not path.is_file():  # an input whose references are still being built, or a host-only program
            continue
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


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _kernel_identity(catalog, symbol: str) -> dict[str, Any]:
    """Which source groups a kernel serves, and its interface and call-shape identities (two kernels
    with both digests equal are the same kernel, whichever program or group they came from)."""
    regions = sorted(
        {str(r.get("source_region_id")) for r in catalog.sidecar.get("routed") or () if r.get("symbol") == symbol}
    )
    return {
        "source_region_id": regions[0] if regions else None,
        "source_region_ids": regions,
        "interface_sha256": _sha256(catalog.interfaces.get(symbol, "")),
        "shape_sha256": _sha256(json.dumps(catalog.sidecar["call_buffers"].get(symbol), sort_keys=True)),
    }


def _program_census(bundle, model: str, program: str, refused: dict[str, str]) -> dict[str, Any]:
    """The bundle's fixed-denominator census with the submission's refused kernels moved off the device."""
    data = json.loads((bundle.program_dir(model, program) / "census.json").read_text())
    rows = data["contractions"]
    on_device = [r for r in rows if r.get("placement") == "device" and r.get("symbol") not in refused]
    total = sum(int(r.get("macs") or 0) for r in rows)
    return {
        "denominator": int(data["denominator"]),
        "device": len(on_device),
        "refused": sum(1 for r in rows if r.get("placement") == "device" and r.get("symbol") in refused),
        "device_mac_fraction": round(sum(int(r.get("macs") or 0) for r in on_device) / total, 6) if total else None,
    }


def _replay_program(args, bundle, model, program, inputs, *, group, all_calls, work: Path) -> dict[str, Any]:
    catalog = bundle.catalog(model.name, program)
    cases = _cases(bundle, model.name, program, inputs, group=group, all_calls=all_calls)
    row = _replay_cases(args, catalog, cases, work)
    return {"program": program, **row, "census": _program_census(bundle, model.name, program, row["refused"])}


def _replay_cases(
    args, catalog, cases: Sequence[RP.ReplayCase], work: Path, *, cache_root: Path | None = None
) -> dict[str, Any]:
    """Build the package's kernels for ``cases``' symbols, replay every case and judge it exactly.

    ``cache_root`` holds the stored kernel objects and replay consoles (default: the build tree's);
    cases that must stay private (sealed variants) pass a private one.
    """
    from merlin.model_check import cache as KC
    from merlin.model_check import diagnostics as DG
    from merlin.model_check import export as EX

    symbols = sorted({c.symbol for c in cases})
    toolchain = RP.toolchain_for(args.host_package)
    use_cache = not getattr(args, "no_cache", False)
    started = time.monotonic()
    built = KC.build_kernels(
        catalog,
        symbols,
        package_dir=args.submission,
        workdir=work / "device",
        toolchain=toolchain,
        build=RP.build_kernels,
        enabled=use_cache,
        root=cache_root,
    )
    build_seconds = time.monotonic() - started
    refused, objects = built.refused, built.objects

    def run(image_dir: Path, main_c: Path, data_s: Path, linked) -> tuple[str, float, dict[str, Any]]:
        if args.engine != "spike":
            raise RP.ReplayError("replay on gsim is not wired yet in this version")
        return KC.replay_image(
            image_dir,
            main_c,
            data_s,
            linked,
            toolchain,
            target=catalog.target,
            dram_base=args.dram_base,
            dram_bytes=args.dram_bytes,
            timeout=args.timeout_s,
            enabled=use_cache,
            root=cache_root,
        )

    verdicts: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    runnable = [c for c in cases if c.symbol in built.kernels]
    for index, chunk in enumerate(_chunks(runnable)):
        image_dir = work / f"image_{index}"
        image_dir.mkdir(parents=True, exist_ok=True)
        main_c, data_s, plan = RP._write_image_sources(image_dir, chunk, built.kernels, catalog.interfaces)  # noqa: SLF001
        console, wall, record = run(image_dir, main_c, data_s, objects)
        verdicts += RP.judge_console(console, plan)
        images.append(
            {
                "cases": len(chunk),
                "engine": args.engine,
                "seconds": round(wall, 1),
                "elf": record.get("elf"),
                "cached": record["cached"],
            }
        )
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
        identity = {**_kernel_identity(catalog, symbol), "form": EX.describe(catalog, symbol)["form"]}
        if symbol in refused:
            kernels.append(
                {
                    "symbol": symbol,
                    **identity,
                    "status": "refused",
                    "detail": refused[symbol][:300],
                    "diagnosis": DG.refusal(catalog, symbol, refused[symbol]),
                }
            )
            continue
        row = per_symbol.get(symbol, {"symbol": symbol, "cases": 0, "exact": 0, "mismatches": []})
        status = "pass" if row["cases"] and row["exact"] == row["cases"] else "fail"
        kernels.append({**row, **identity, "status": status, "mismatches": row["mismatches"][:5]})
    diagnoses = DG.localize(
        runnable,
        verdicts,
        catalog=catalog,
        kernels=built.kernels,
        objects=objects,
        work=work / "diagnose",
        run=lambda *image: run(*image)[0],
        image_bytes=IMAGE_BYTES,
    )
    DG.attach(kernels, diagnoses)
    return {
        "build": {
            "seconds": round(build_seconds, 1),
            "built": len(built.kernels),
            "refused": len(refused),
            "cache": built.stats,
        },
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
    report["binding"] = manifest.get("binding")
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
        # The sealed split also carries the program's shape variants, judged like recorded calls.
        if args.split != "visible":
            varied = VAR.load(bundle.root, model.name, program, bundle.catalog(model.name, program))
            if varied is not None:
                private = VAR.scratch(bundle.root, model.name) / work.name / program
                row = _replay_cases(
                    args,
                    varied.catalog,
                    varied.cases(group=group),
                    private,
                    cache_root=VAR.scratch(bundle.root, model.name) / "cache",
                )
                varied.annotate(row["kernels"])
                report["programs"].append({"program": program, "split": "sealed", "source": VAR.SOURCE, **row})
    variant_calls = sum(k.get("cases", 0) for p in report["programs"] if p.get("source") for k in p["kernels"])
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
        f"bit-exact on {len(inputs)} input(s) and {variant_calls} shape variant(s); {len(failed)} kernel(s) differ",
    }
    for kernel in failed:
        failure = {
            "kind": "gate",
            "gate": "kernel_exact",
            "symbol": kernel["symbol"],
            "detail": ", ".join(kernel["mismatches"]),
            **({"split": "sealed"} if "variant_of" in kernel else {}),
        }
        if "diagnosis" in kernel:  # a visible input's: any other split's stays under the kernel's sealed key
            from merlin.model_check.diagnostics import summary

            failure["where"] = summary(kernel["diagnosis"])
        report["failures"].append(failure)
    return report


@register("replay-kernel")
def replay_kernel(args: argparse.Namespace, report: dict) -> dict:
    report = _replay(args, report, all_calls=bool(args.all_groups) or args.split != "visible", group=args.group)
    report["gates"]["kernel_exact"]["partial"] = False
    return report


def _reference_metrics(bundle: B.Bundle, model: str, split: str) -> dict[str, Any]:
    """The model metrics when every kernel is exact: the INT8 reference's own, per input and worst case."""
    per_input = []
    for input_id in bundle.inputs(model, split):
        path = bundle.root / model / input_id / "metrics_int8.json"
        if path.is_file():
            row = {k: v for k, v in json.loads(path.read_text()).items() if k != "per_channel"}
            per_input.append({"split": input_id.partition("/")[0], "id": input_id, **row})
    return {
        "source": "int8_reference",
        "inputs": len(per_input),
        "per_input": per_input,
        "worst": M.worst_of(per_input),
    }


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
        if exact:
            report["model_metrics"] = _reference_metrics(_bundle(args), args.model, args.split)
        report["gates"]["model_numerics"] = {
            "status": "pass" if exact else "fail",
            "detail": "every kernel exact, so the model computes its INT8 reference"
            if exact
            else "a kernel differs from its reference; the model gate cannot pass",
        }
    return report


@register("kernels-export")
def kernels_export(args: argparse.Namespace, report: dict) -> dict:
    from merlin.model_check import export as EX

    models = load_models(args.models)
    model = models.model(args.model)
    bundle = _bundle(args)
    report["bundle"] = {"path": str(bundle.root), "merlin_commit": bundle.manifest().get("merlin_commit")}
    record = EX.export_model(bundle, model, args.out_dir, programs=_programs(model, args))
    for program in record["programs"]:
        report["programs"].append(
            {
                "program": program["program"],
                "census": program["census"],
                "kernels": [
                    {k: kernel[k] for k in ("symbol", "form", "routed_groups", "interface_sha256", "path")}
                    for kernel in program["kernels"]
                ],
            }
        )
    count = sum(len(p["kernels"]) for p in record["programs"])
    report["gates"]["export"] = {
        "status": "pass",
        "detail": f"{count} kernel(s) of {len(record['programs'])} program(s) -> {Path(args.out_dir) / model.name}",
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
