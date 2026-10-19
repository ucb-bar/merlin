"""The subcommands behind ``python -m merlin.model_check`` (contract: ``docs/reference/model_check.md``)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from merlin.model_check import bundle as B
from merlin.model_check import engines as EN
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


@register("bundle-admission")
def bundle_admission(args: argparse.Namespace, report: dict) -> dict:
    """Re-judge a built bundle's reference admission under the models file's current rules (§4.5)."""
    from merlin.model_check import references as REF

    model = load_models(args.models).model(args.model)
    bundle = _bundle(args)
    admission = REF.rejudge(bundle, model)["admission"]
    report["bundle"] = {"path": str(bundle.root)}
    report["model_metrics"] = {"admission": admission}
    report["gates"]["admission"] = {
        "status": "pass" if admission and all(a["status"] == "pass" for a in admission.values()) else "fail",
        "detail": ", ".join(f"{split} {a['status']}" for split, a in admission.items()),
    }
    return report


@register("bundle-seal")
def bundle_seal(args: argparse.Namespace, report: dict) -> dict:
    from merlin.model_check import seal as SL

    bundle = _bundle(args)
    record = SL.seal(bundle, jobs=args.jobs)
    report["bundle"] = {"path": str(bundle.root), "seal": record}
    report["gates"]["seal"] = {"status": "pass", "detail": f"{record['files']} file(s), {record['bytes']} bytes"}
    return report


@register("bundle-verify")
def bundle_verify(args: argparse.Namespace, report: dict) -> dict:
    from merlin.model_check import seal as SL

    bundle = _bundle(args)
    result = SL.verify(bundle, jobs=args.jobs)
    report["bundle"] = {"path": str(bundle.root), "verify": result}
    counts = result.get("counts") or {}
    report["gates"]["bundle"] = {
        "status": result["status"],
        "detail": result.get("detail")
        or f"{result['files']} file(s); {counts.get('changed', 0)} changed, {counts.get('missing', 0)} missing, "
        f"{counts.get('extra', 0)} extra; {len(result['structure'])} structure problem(s)",
    }
    for kind in ("changed", "missing", "extra", "structure"):
        for item in result.get(kind) or ():
            report["failures"].append({"kind": "gate", "gate": "bundle", "detail": f"{kind}: {item}"})
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
    bundle: B.Bundle,
    model: str,
    program: str,
    inputs: Sequence[str],
    *,
    group: str | None,
    all_calls: bool,
    unique: bool = False,
) -> list[RP.ReplayCase]:
    """Replay cases for ``program``: every call, (fast) a few per symbol on a few inputs, or (``unique``)
    one call per symbol at its recorded shape: the call with the fewest operand bytes, the first on ties."""
    blobs = bundle.blobs(model)
    catalog = bundle.catalog(model, program)
    region_of = {}
    for row in catalog.sidecar.get("routed") or ():
        region_of.setdefault(str(row.get("symbol")), set()).add(str(row.get("source_region_id")))

    def case(input_id: str, call: dict) -> RP.ReplayCase:
        return RP.ReplayCase(
            symbol=call["symbol"],
            label=f"{input_id}/{program}/{call['symbol']}#{call['ordinal']}",
            roles=tuple(call["roles"]),
            operands=tuple(blobs.get(k) for k in call["operands"]),
            expected=tuple(blobs.get(k) for k in call["expected"]),
        )

    cases: list[RP.ReplayCase] = []
    picked: dict[str, tuple[int, str, dict]] = {}
    for input_id in inputs:
        path = bundle.root / model / input_id / program / "calls.json"
        if not path.is_file():  # an input whose references are still being built, or a host-only program
            continue
        doc = bundle.calls(model, input_id, program)
        for call in doc["calls"]:
            symbol = call["symbol"]
            if group is not None and group != symbol and group not in region_of.get(symbol, ()):
                continue
            if unique:
                size = sum((blobs.root / f"{k}.npy").stat().st_size for k in call["operands"])
                if symbol not in picked or size < picked[symbol][0]:
                    picked[symbol] = (size, input_id, call)
                continue
            if not all_calls and call["ordinal"] >= FAST_CALLS_PER_SYMBOL:
                continue
            cases.append(case(input_id, call))
    if unique:
        cases = [case(input_id, call) for _, (_, input_id, call) in sorted(picked.items())]
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


def _replay_program(
    args,
    bundle,
    model,
    program,
    inputs,
    *,
    group,
    all_calls,
    work: Path,
    unique: bool = False,
    engine=None,
    deadline: float | None = None,
) -> dict[str, Any]:
    catalog = bundle.catalog(model.name, program)
    cases = _cases(bundle, model.name, program, inputs, group=group, all_calls=all_calls, unique=unique)
    row = _replay_cases(args, catalog, cases, work, engine=engine, deadline=deadline)
    return {"program": program, **row, "census": _program_census(bundle, model.name, program, row["refused"])}


def _one_per_symbol(cases: Sequence[RP.ReplayCase]) -> list[RP.ReplayCase]:
    """Of ``cases``, each symbol's call with the fewest operand bytes (the first on ties)."""
    picked: dict[str, RP.ReplayCase] = {}
    for case in cases:
        size = sum(int(a.nbytes) for a in case.operands)
        if case.symbol not in picked or size < sum(int(a.nbytes) for a in picked[case.symbol].operands):
            picked[case.symbol] = case
    return [picked[s] for s in sorted(picked)]


def _replay_cases(
    args,
    catalog,
    cases: Sequence[RP.ReplayCase],
    work: Path,
    *,
    cache_root: Path | None = None,
    engine=None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Build the package's kernels for ``cases``' symbols, replay every case and judge it exactly.

    ``cache_root`` holds the stored kernel objects and replay consoles (default: the build tree's);
    cases that must stay private (sealed variants) pass a private one. ``engine``: board images on
    Spike, then on that RTL engine, within ``deadline``; those runs are never answered from the
    console store, and a mismatch is not localized (each localizing run would be another RTL run).
    """
    from merlin.model_check import cache as KC
    from merlin.model_check import diagnostics as DG
    from merlin.model_check import export as EX

    symbols = sorted({c.symbol for c in cases})
    toolchain = RP.toolchain_for(args.host_package)
    use_cache = not getattr(args, "no_cache", False)
    mandatory_build = getattr(args, "mode", None) == "full"
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
        stop_on_first_failure=mandatory_build,
    )
    build_seconds = time.monotonic() - started
    refused, objects = built.refused, built.objects
    build_failed = mandatory_build and bool(refused or built.unattempted)

    def run(image_dir: Path, main_c: Path, data_s: Path, linked) -> tuple[str, float, dict[str, Any]]:
        if engine is not None or args.engine != "spike":
            raise RP.ReplayError("only functional replays are linked and answered here (gsim: engines.py)")
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
    runnable = [] if build_failed else [c for c in cases if c.symbol in built.kernels]
    if engine is not None:  # board images, run on Spike then on the bound RTL engine
        verdicts, images = EN.replay_on_board(
            catalog.target,
            engine,
            runnable,
            built.kernels,
            catalog.interfaces,
            objects,
            toolchain,
            work,
            timeout_s=args.timeout_s,
            jobs=getattr(args, "jobs", 1),
            max_cycles=getattr(args, "max_cycles", None),
            call_max_cycles=getattr(args, "call_max_cycles", None),
            deadline=deadline,
            log=lambda line: print(line, file=sys.stderr, flush=True),
        )
    for index, chunk in enumerate(_chunks(runnable) if engine is None else ()):
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
        if symbol in built.unattempted or build_failed:
            detail = built.unattempted.get(symbol) or "not replayed after a required kernel build failure"
            kernels.append(
                {
                    "symbol": symbol,
                    **identity,
                    "status": "unfinished",
                    "cases": 0,
                    "exact": 0,
                    "mismatches": [detail],
                    "detail": detail,
                }
            )
            continue
        row = per_symbol.get(symbol, {"symbol": symbol, "cases": 0, "exact": 0, "mismatches": []})
        status = "pass" if row["cases"] and row["exact"] == row["cases"] else "fail"
        kernels.append({**row, **identity, "status": status, "mismatches": row["mismatches"][:5]})
    EN.annotate_kernels(kernels, verdicts, cycles_kind="functional" if engine is None else "rtl")
    if engine is None:
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
            **({"unattempted": len(built.unattempted)} if built.unattempted else {}),
            "cache": built.stats,
        },
        "kernels": kernels,
        "images": images,
        "refused": refused,
    }


def _replay(args, report, *, all_calls: bool, group: str | None, unique: bool = False) -> dict:
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
    if not all_calls and not unique:
        inputs = inputs[:FAST_INPUTS]
    from merlin.common.paths import build_dir

    engine = deadline = None
    if args.engine == "gsim":  # --timeout-s bounds the whole replay, so a caller's deadline holds
        deadline = time.monotonic() + args.timeout_s
        if args.rtl_facts is None:
            raise B.BundleError("gsim replay needs the RTL facts: pass --rtl-facts or use a bundle that records them")
        engine = EN.GsimEngine(models.target, args.rtl_facts, board_catalog=getattr(args, "board_catalog", None))
        report["engine"] = engine.identity()
    # A run gets its own directory: two checks of one model started in the same second shared one and
    # overwrote each other's replay images. Absolute: the image's compilers and the emulator run inside
    # it, and every gSIM file (image, console, emulator logs) lives under it.
    root = Path(args.work or (build_dir() / "model_check")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    work = Path(tempfile.mkdtemp(prefix=f"{model.name}-{stamp}-", dir=root))
    programs = _programs(model, args)
    for program_index, program in enumerate(programs):
        report["programs"].append(
            _replay_program(
                args,
                bundle,
                model,
                program,
                inputs,
                group=group,
                all_calls=all_calls,
                work=work / program,
                unique=unique,
                engine=engine,
                deadline=deadline,
            )
        )
        if getattr(args, "mode", None) == "full" and (
            report["programs"][-1]["refused"]
            or any(k["status"] == "unfinished" for k in report["programs"][-1]["kernels"])
        ):
            report["programs_unattempted"] = programs[program_index + 1 :]
            break
        # The sealed split also carries the program's shape variants, judged like recorded calls.
        if args.split != "visible":
            varied = VAR.load(bundle.root, model.name, program, bundle.catalog(model.name, program))
            if varied is not None:
                private = VAR.scratch(bundle.root, model.name) / work.name / program
                varied_cases = varied.cases(group=group)
                if unique:
                    varied_cases = _one_per_symbol(varied_cases)
                row = _replay_cases(
                    args,
                    varied.catalog,
                    varied_cases,
                    private,
                    cache_root=VAR.scratch(bundle.root, model.name) / "cache",
                    engine=engine,
                    deadline=deadline,
                )
                varied.annotate(row["kernels"])
                report["programs"].append({"program": program, "split": "sealed", "source": VAR.SOURCE, **row})
                if getattr(args, "mode", None) == "full" and (
                    row["refused"] or any(k["status"] == "unfinished" for k in row["kernels"])
                ):
                    report["programs_unattempted"] = programs[program_index + 1 :]
                    break
    variant_calls = sum(k.get("cases", 0) for p in report["programs"] if p.get("source") for k in p["kernels"])
    kernels = [k for p in report["programs"] for k in p["kernels"]]
    refused = [k for k in kernels if k["status"] == "refused"]
    failed = [k for k in kernels if k["status"] in ("fail", "unfinished")]
    if engine is not None:
        _engine_gates(report, kernels)
    report["gates"]["build"] = {
        "status": "pass"
        if not refused and not any(p["build"].get("unattempted") for p in report["programs"])
        else "fail",
        "detail": f"{sum(p['build']['built'] for p in report['programs'])}/{len(kernels)} kernels built",
    }
    if "programs_unattempted" in report:
        report["gates"]["build"].update(
            partial=True,
            detail=report["gates"]["build"]["detail"] + "; stopped after a required build failure; "
            f"{len(report['programs_unattempted'])} later program(s) not attempted",
        )
    report["gates"]["kernel_exact"] = {
        "status": "pass" if kernels and not failed and not refused else "fail",
        "partial": not all_calls,
        "detail": f"{sum(k.get('exact', 0) for k in kernels)}/{sum(k.get('cases', 0) for k in kernels)} calls "
        f"bit-exact on {'one recorded call per kernel' if unique else f'{len(inputs)} input(s)'} and "
        f"{variant_calls} shape variant(s); {len(failed)} kernel(s) differ or did not finish",
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


def _engine_gates(report: dict, kernels: Sequence[dict[str, Any]]) -> None:
    """RTL replay: the same image must print the same digests on Spike and on the RTL engine, call by
    call; a call an engine bound stopped is an infrastructure failure (exit 3), never a verdict."""
    judged = [k for k in kernels if k["status"] != "refused"]
    calls = sum(k.get("cases", 0) for k in judged)
    agree = sum(k.get("engines_agree", 0) for k in judged)
    unfinished = [k for k in judged if k.get("unfinished")]
    report["gates"]["engine_agreement"] = {
        "status": "pass" if judged and agree == calls and not unfinished else "fail",
        "detail": f"{agree}/{calls} calls print identical digests on spike and gsim (same board image)",
    }
    for kernel in unfinished:
        report["failures"].append(
            {
                "kind": "infrastructure",
                "gate": "kernel_exact",
                "symbol": kernel["symbol"],
                "detail": f"gsim did not finish {', '.join(kernel['unfinished'])}: {kernel.get('stopped')}",
            }
        )
    for kernel in judged:
        if kernel.get("over_call_budget"):
            report["failures"].append(
                {
                    "kind": "budget",
                    "gate": None,
                    "symbol": kernel["symbol"],
                    "detail": f"{kernel['over_call_budget']} call(s) over --call-max-cycles ({kernel.get('cycles')})",
                }
            )


@register("replay-kernel")
def replay_kernel(args: argparse.Namespace, report: dict) -> dict:
    report = _replay(
        args,
        report,
        all_calls=bool(args.all_groups) or args.split != "visible",
        group=args.group,
        unique=bool(getattr(args, "unique", False)) or args.engine == "gsim",  # RTL evidence is per kernel
    )
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
        "mean": M.mean_of(per_input),
    }


@register("check")
def check(args: argparse.Namespace, report: dict) -> dict:
    if args.mode == "full":  # fast's kernel replay, then every program's linked image and chain
        from merlin.model_check import full

        full.validate(args)
        report = _replay(args, report, all_calls=False, group=None)
        if report["gates"]["build"]["status"] != "pass":
            report["gates"]["kernel_exact"]["partial"] = True
            selected = getattr(args, "full_comparison", "host-twin")
            report["comparison"] = {
                "selected": selected,
                "status": "not_run",
                "detail": "a required accelerator kernel could not be built",
            }
            return report
        report["gates"]["kernel_exact"]["partial"] = False
        return full.check(args, report, models=load_models(args.models), bundle=_bundle(args))
    from merlin.model_check.approximations import declared

    approximations = declared(Path(args.submission))
    report = _replay(args, report, all_calls=False, group=None, unique=args.engine == "gsim")
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


@register("bundle-twins")
def bundle_twins(args: argparse.Namespace, report: dict) -> dict:
    from merlin.model_check import twins

    bundle = _bundle(args)
    _bind_engine_inputs(args, bundle.manifest())
    return twins.command(args, report, models=load_models(args.models), bundle=bundle)


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
