"""Build and check a provider-workflow object using an explicit native recipe.

The plan supplies the software harness and full-output checker. This owner never
discovers a board, flashes hardware or substitutes a different simulator. A
successful supplied gate is a scoped execution result, not certification.
"""

from __future__ import annotations

import csv
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

from merlin.runtime.backends import device_simulator as ds


def _bound(item, base, inputs):
    if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
        raise ValueError("each explicit input requires path and sha256")
    path = Path(item["path"])
    # A venv interpreter's lexical path selects its environment even when its
    # executable is a symlink. Hash its bytes without changing that invocation.
    path = (base / path).absolute() if not path.is_absolute() else path.absolute()
    digest = item["sha256"]
    if str(path) in inputs and inputs[str(path)] != digest:
        raise ValueError("conflicting input byte identities")
    ds.check_inputs({str(path): digest})
    inputs[str(path)] = digest
    return path


def _flags(value):
    if not isinstance(value, list) or any(not isinstance(s, str) or not s or "\0" in s for s in value):
        raise ValueError("flags must be a literal nonempty-string argv list")
    return value


def _invoke(command, output, name, timeout):
    """Keep failed raw streams and reap only this invocation's process group."""
    started = time.monotonic()
    with (output / (name + ".stdout")).open("wb") as stdout, (output / (name + ".stderr")).open("wb") as stderr:
        proc = subprocess.Popen(
            list(map(str, command)), stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True
        )
        try:
            code = proc.wait(timeout=timeout)
        except BaseException:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            raise
    return {"argv": list(map(str, command)), "returncode": code, "host_seconds": time.monotonic() - started}


def _entry_abi(llvm_path, expected):
    """Verify the closed, default-C flattened ranked-memref harness convention."""
    if not isinstance(expected, dict) or set(expected) != {"symbol", "argument_ranks"}:
        raise ValueError("entry_abi requires symbol and argument_ranks")
    symbol, ranks = expected["symbol"], expected["argument_ranks"]
    if not isinstance(symbol, str) or not symbol.isidentifier() or not symbol.isascii():
        raise ValueError("entry symbol must be a plain C identifier")
    if not isinstance(ranks, list) or not ranks or any(type(r) is not int or not 0 <= r <= 8 for r in ranks):
        raise ValueError("entry ABI requires explicit bounded ranked memrefs")
    anchor = "define void @" + symbol + "("
    definitions = [line for line in Path(llvm_path).read_text().splitlines() if line.startswith(anchor)]
    if len(definitions) != 1 or ")" not in definitions[0]:
        raise ValueError("no unique default-C void entry with the selected symbol")
    text = definitions[0][len(anchor) :].split(")", 1)[0]
    arguments = [part.strip().split() for part in text.split(",")]
    wanted = [kind for rank in ranks for kind in ("ptr", "ptr", "i64", *(["i64"] * (2 * rank)))]
    if len(arguments) != len(wanted) or any(
        len(a) != 2 or a[0] != kind or not a[1].startswith("%") for a, kind in zip(arguments, wanted)
    ):
        raise ValueError("emitted entry ABI differs from the bound harness ranks/types")
    return {**expected, "verified": True, "llvm_sha256": ds.sha(llvm_path), "flattened_types": wanted}


def execute(lowered, plan_path):
    """Execute an explicit native-only plan after compiling its source object."""
    plan_path = Path(plan_path).resolve(strict=True)
    plan = json.loads(plan_path.read_text())
    allowed = {
        "schema",
        "compiler",
        "sources",
        "objects",
        "linker_script",
        "compile_flags",
        "link_flags",
        "simulator",
        "isa",
        "harts",
        "active_hart_ids",
        "strict_postgate",
        "timeout_seconds",
        "postgate_timeout_seconds",
        "input_sha256",
        "entry_abi",
        "profile",
    }
    if not isinstance(plan, dict) or set(plan) - allowed or plan.get("schema") != "merlin.provider_execution.v1":
        raise ValueError("execution plan requires closed merlin.provider_execution.v1 schema")
    for key in ("timeout_seconds", "postgate_timeout_seconds"):
        value = plan.get(key, 120)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 86400:
            raise ValueError("execution and postgate timeouts must be finite and bounded")
    if lowered.riscv_obj is None:
        raise ValueError("native execution requires a compiled RISC-V object")
    work = Path(lowered.workdir)
    output = work / "execution"
    output.mkdir(exist_ok=False)
    receipt = {
        "schema": "merlin.provider_execution_receipt.v1",
        "status": "failed",
        "hardware_contacted": False,
        "numerical_certification": False,
        "timing_certification": False,
        "commands": [],
        "plan_sha256": ds.sha(plan_path),
    }
    inputs = {str(plan_path): ds.sha(plan_path), str(Path(__file__).resolve()): ds.sha(__file__)}
    try:
        base = plan_path.parent
        for name, digest in (plan.get("input_sha256") or {}).items():
            _bound({"path": name, "sha256": digest}, base, inputs)
        compiler = _bound(plan["compiler"], base, inputs)
        sources = [_bound(p, base, inputs) for p in plan["sources"]]
        objects = [_bound(p, base, inputs) for p in plan.get("objects", [])]
        if not sources:
            raise ValueError("execution plan requires a complete software harness source roster")
        script = _bound(plan["linker_script"], base, inputs)
        simulator_path = _bound(plan["simulator"], base, inputs)
        simulator = json.loads(simulator_path.read_text())
        inputs.update(ds.execution_inputs(simulator))
        gate = plan["strict_postgate"]
        if not isinstance(gate, dict) or set(gate) != {"python", "script"}:
            raise ValueError("strict_postgate requires bound python and script")
        python = _bound(gate["python"], base, inputs)
        checker = _bound(gate["script"], base, inputs)
        compilation = json.loads((work / "provider-workflow.json").read_text())
        if compilation.get("status") != "compiled" or compilation.get("object_sha256") != ds.sha(lowered.riscv_obj):
            raise ValueError("compiled object no longer matches its provider receipt")
        if compilation.get("llvm_sha256") != ds.sha(lowered.ll_path):
            raise ValueError("compiled LLVM entry identity no longer matches its provider receipt")
        receipt["entry_abi"] = _entry_abi(lowered.ll_path, plan["entry_abi"])
        for name, digest in compilation["input_sha256"].items():
            _bound({"path": name, "sha256": digest}, base, inputs)
        for name in compilation.get("additional_objects", []):
            objects.append(
                _bound({"path": name, "sha256": compilation["additional_object_sha256"][name]}, base, inputs)
            )
        objects.insert(0, Path(lowered.riscv_obj))
        for path in objects:
            inputs[str(path.resolve())] = ds.sha(path)
        inputs[str(work / "provider-workflow.json")] = ds.sha(work / "provider-workflow.json")
        flags = _flags(plan["compile_flags"])
        limit = plan.get("timeout_seconds", 120)
        for index, source in enumerate(sources):
            obj = output / f"harness-{index}.o"
            result = _invoke([compiler, *flags, "-c", source, "-o", obj], output, f"compile-{index}", limit)
            receipt["commands"].append(result)
            if result["returncode"]:
                raise RuntimeError("explicit software harness compilation failed")
            objects.append(obj)
        elf = output / "image.elf"
        result = _invoke(
            [compiler, *flags, *objects, "-T", script, *_flags(plan["link_flags"]), "-o", elf], output, "link", limit
        )
        receipt["commands"].append(result)
        if result["returncode"]:
            raise RuntimeError("explicit native image link failed")
        ds.check_inputs(inputs)
        run = ds.run(
            simulator,
            elf,
            isa=plan["isa"],
            harts=plan["harts"],
            active_hart_ids=tuple(plan["active_hart_ids"]),
            timeout=limit,
            output=output / "native",
        )
        receipt["native"] = {k: v for k, v in run.items() if k not in ("stdout", "stderr")}
        console = output / "console"
        console.write_text(run["stdout"] + "\n" + run["stderr"])
        if run["returncode"] != 0 or run["timed_out"] or run["status"] != "passed_guest_exit":
            raise RuntimeError("native guest did not complete successfully; raw streams retained")
        semantics = simulator.get("counter_semantics")
        if not isinstance(semantics, str) or not semantics:
            raise ValueError("simulator must explicitly declare counter_semantics before profiling")
        gate_output = output / "postgate"
        checked = _invoke(
            [python, checker, "--console", console, "--output", gate_output, "--counter-semantics", semantics],
            output,
            "postgate",
            plan.get("postgate_timeout_seconds", 120),
        )
        receipt["commands"].append(checked)
        report_path = gate_output / "report.json"
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        if checked["returncode"] != 0 or report.get("pass") is not True:
            raise RuntimeError("supplied complete-output postgate refused; no correctness claim")
        ds.check_inputs(inputs)
        if plan.get("profile") is not None:
            profile = plan["profile"]
            if not isinstance(profile, dict) or set(profile) - {"expected_regions", "record_path", "labels"}:
                raise ValueError("unknown observed profile selection")
            from merlin.runtime import profile_evidence
            from merlin.runtime.profile_evidence import (
                BoundFile,
                CounterDomain,
                Gate,
                RegionSource,
                collect,
                write_report,
            )

            inputs[str(Path(profile_evidence.__file__).resolve())] = ds.sha(profile_evidence.__file__)

            label_file = _bound(profile["labels"], base, inputs) if profile.get("labels") is not None else None
            domain = (
                "functional_instructions"
                if semantics == "instruction_count_not_modeled_cycles"
                else "modeled_cycles"
                if semantics.startswith("modeled_")
                else None
            )
            if domain is None:
                raise ValueError("profile counter domain is not an admitted functional/modeled simulator domain")
            binding = BoundFile(str(report_path.resolve()), ds.sha(report_path))
            observed = RegionSource(
                name="native",
                records=binding,
                elf=BoundFile(str(elf.resolve()), ds.sha(elf)),
                gate=Gate(binding, ("pass",), "explicit complete-output postgate", "caller-declared full-output gate"),
                domain=CounterDomain(domain, semantics),
                expected_regions=profile["expected_regions"],
                record_path=tuple(profile.get("record_path", ["regions"])),
                labels=BoundFile(str(label_file), ds.sha(label_file)) if label_file else None,
            )
            metrics, trace = collect([observed])
            write_report(output / "profile", metrics, trace)
            fields = (
                "source",
                "case_id",
                "region",
                "hart",
                "kind",
                "counter_semantics",
                "start",
                "end",
                "ticks",
                "cycles",
                "retired_instructions",
                "retired_per_cycle",
                "physical_busy",
            )
            with (output / "profile/regions.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(metrics["regions"])
            receipt["profile"] = str(output / "profile")
            ds.check_inputs(inputs)
        receipt.update(
            status="passed_supplied_gate",
            input_sha256=inputs,
            ELF_sha256=ds.sha(elf),
            strict_postgate_report=str(report_path),
            strict_postgate_report_sha256=ds.sha(report_path),
            counter_semantics=semantics,
            qualification="scoped supplied numerical/observability gate; no hardware or timing certification",
        )
        lowered.stats["execution"] = str(output / "receipt.json")
        lowered.stats["native_gate"] = "passed_supplied_gate"
        lowered.stats["counter_semantics"] = semantics
        return receipt
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")
