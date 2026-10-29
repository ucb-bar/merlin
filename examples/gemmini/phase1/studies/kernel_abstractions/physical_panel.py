"""Private frozen-panel build and physical observation import; never dispatch FPGA jobs.

Selections bind an evaluation plan, transfer panel, compiled objects and runtime.
The generated ELF runs three declared rounds, poisons outputs before each call,
closes the cycle window after kernel_complete, then checks every output byte.
Inputs and oracle arrays are materialized operator-only fixtures. No author
workspace is an output destination. The ``legacy-runtime`` action adapts pinned
HTIF/Gemmini support to this builder without changing retained source files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

from merlin.common.tree_hash import hash_tree

SCHEMA = "merlin.study.physical_panel.v1"
OBSERVATION = "merlin.study.physical_panel.observation.v1"
WIDTHS = {"i8": 1, "i32": 4}
DTYPES = {"i8": "int8", "i32": "int32"}


def pin(path):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path != path.resolve():
        raise ValueError(f"artifact must be an ordinary canonical file: {path}")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def reopen(record):
    if pin(record["path"]) != {k: record[k] for k in ("path", "sha256")}:
        raise ValueError(f"artifact changed: {record['path']}")
    return Path(record["path"])


def read(record):
    value = json.loads(reopen(record).read_bytes())
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def write(path, value):
    path = Path(path)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def _token(value):
    return (isinstance(value, str) and value and len(value) <= 120
            and all(c.isascii() and (c.isalnum() or c in "_.-") for c in value))


def _canonical_directory(path):
    path = Path(path)
    if not path.is_absolute() or path != path.resolve() or not path.is_dir():
        raise ValueError("select a canonical directory")
    return path


def _private_output(output, plan):
    output = Path(output).absolute()
    if output.exists() or output != output.resolve():
        raise ValueError("select an unused canonical private output directory")
    for subject in plan["subjects"]:
        source = Path(subject["source"]["path"]).resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("measurement output must be outside frozen compiler sources")
    return output


def _abi(cb):
    tensors = cb["tensors"]
    explicit = cb.get("kernel_abi") or (cb.get("params") or {}).get("compiler_argument_contract")
    if explicit:
        order = [row["tensor"] for row in explicit["args"]]
    else:
        order = [name for role in (True, False) for name, value in tensors.items()
                 if (value.get("role") in ("input", "weight", "bias", "scale")) == role]
    if len(order) != len(set(order)) or set(order) != set(tensors):
        raise ValueError("kernel pointer ABI must name every tensor exactly once")
    return order


def _array_bytes(archive, name, tensor):
    array = archive[name]
    if list(array.shape) != tensor["shape"] or str(array.dtype) != DTYPES.get(tensor["dtype"]):
        raise ValueError(f"fixture dtype/shape differs from ABI: {name}")
    if array.dtype.byteorder == ">":
        raise ValueError("fixture must use target little endian bytes")
    return array.tobytes(order="C")


def _artifact(row, subject, case):
    if row.get("compiler_sha256") != subject["source"]["sha256"]:
        raise ValueError("compiled artifact belongs to another frozen compiler")
    if row.get("state", "compiled") != "compiled":
        if not row.get("reason"):
            raise ValueError("missing/failed artifact requires an explicit reason")
        return None
    cb = read(row["command_buffer"])
    receipt = read(row["build_receipt"])
    expected = {"compiler_sha256": subject["source"]["sha256"], "case": case["case"],
                "source_sha256": case["source"]["sha256"],
                "object_sha256": row["object"]["sha256"],
                "command_buffer_sha256": row["command_buffer"]["sha256"]}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("build receipt does not bind source, object and ABI")
    reopen(row["object"])
    tensors = cb["tensors"]
    if not tensors or any(t.get("dtype") not in WIDTHS or not t.get("shape")
                          or any(type(d) is not int or d < 1 for d in t["shape"])
                          for t in tensors.values()):
        raise ValueError("physical panel currently supports positive-shape i8/i32 tensors")
    return cb, _abi(cb)


def prepare(selection_path, output):
    """Validate pins and produce a deterministic private plan; no compiler is invoked."""
    selection_pin = pin(selection_path)
    selection = read(selection_pin)
    if selection.get("schema") != SCHEMA or selection.get("rounds", 3) != 3:
        raise ValueError("physical panel requires its schema and three ordered rounds")
    plan, panel = read(selection["evaluation_plan"]), read(selection["panel"])
    output = _private_output(output, plan)
    if panel.get("schema") != "merlin.study.transfer_panel.v1":
        raise ValueError("physical panel requires the frozen transfer panel")
    subjects = {s["id"]: s for s in plan["subjects"]}
    selected = selection["subjects"]
    if not selected or len(selected) != len(set(selected)) or any(s not in subjects for s in selected):
        raise ValueError("select unique known frozen subjects")
    for name in selected:
        subject = subjects[name]
        if hash_tree(subject["source"]["path"])["sha256"] != subject["source"]["sha256"]:
            raise ValueError("frozen compiler changed")
        freeze = subject.get("freeze") or {}
        if (freeze.get("requirements_complete") is not True or type(freeze.get("frozen_unix")) not in (int, float)
                or not math.isfinite(freeze["frozen_unix"]) or freeze["frozen_unix"] <= 0):
            raise ValueError("source, rules, settings and freeze time must be pinned first")
        for key in ("rules", "settings"):
            reopen(freeze[key])
    panel_cases = {row["case"]: row for row in panel["cases"]}
    if len(panel_cases) != len(panel["cases"]):
        raise ValueError("transfer panel repeats a case")
    cases = selection.get("cases") or list(panel_cases)
    if len(cases) != len(set(cases)) or any(c not in panel_cases for c in cases):
        raise ValueError("select unique declared panel cases")
    plan_cases = {row["case"]: row for row in plan["cases"]}
    for name in cases:
        if name not in plan_cases or panel_cases[name]["source"] != plan_cases[name]["source"]:
            raise ValueError("evaluation and physical panels disagree on source contracts")
        reopen(panel_cases[name]["source"])
        for key in ("inputs", "outputs"):
            reopen(panel_cases[name]["fixtures"][key])
    domain = selection["domain"]
    if (not isinstance(domain.get("agfi"), str) or not domain["agfi"]
            or type(domain.get("target_clock_hz")) is not int or domain["target_clock_hz"] <= 0
            or domain.get("roi") != "complete_call_including_completion_v1"
            or not domain.get("cache_policy") or not domain.get("preparation_policy")):
        raise ValueError("select explicit deployment, preparation, cache and complete-call ROI")
    tools = selection["toolchain"]
    for key in ("compiler", "nm", "objcopy", "runtime_header", "linker_script"):
        reopen(tools[key])
    for item in tools["support_objects"]:
        reopen(item)
    for directory in tools.get("include_directories", []):
        _canonical_directory(directory)
    for item in tools.get("runtime_dependencies", []):
        reopen(item)
    for item in tools.get("include_trees", []):
        if hash_tree(_canonical_directory(item["path"]))["sha256"] != item["sha256"]:
            raise ValueError("toolchain include tree changed")
    if domain.get("runtime_header_sha256") != tools["runtime_header"]["sha256"]:
        raise ValueError("measurement domain must bind its exact runtime header")
    rows = {}
    for row in selection["compiled"]:
        key = row["subject"], row["case"]
        if key in rows or key[0] not in selected or key[1] not in cases:
            raise ValueError("compiled cells are duplicated or outside selected roster")
        rows[key] = row
    cells, available = [], {}
    for name in selected:
        for case in cases:
            row = rows.get((name, case))
            if row is None:
                row = {"subject": name, "case": case, "state": "missing", "reason": "no compiled artifact supplied",
                       "compiler_sha256": subjects[name]["source"]["sha256"]}
            parsed = _artifact(row, subjects[name], panel_cases[case])
            cells.append(row)
            if parsed is not None:
                available[name, case] = (row, parsed)
    if not available:
        raise ValueError("no compiled cells to measure")
    slots = []
    orders = [cases, list(reversed(cases)), cases[len(cases) // 2:] + cases[:len(cases) // 2]]
    for round_id, order in enumerate(orders):
        names = selected[round_id % len(selected):] + selected[:round_id % len(selected)]
        if round_id == 1:
            names = list(reversed(names))
        for case in order:
            for name in names:
                item = available.get((name, case))
                if item is None:
                    continue
                row, (cb, abi) = item
                slots.append({"slot": len(slots), "subject": name, "case": case, "round": round_id,
                              "compiler_sha256": subjects[name]["source"]["sha256"],
                              "object": row["object"], "command_buffer": row["command_buffer"],
                              "build_receipt": row["build_receipt"], "abi": abi,
                              "fixtures": panel_cases[case]["fixtures"]})
    # Repeat the first measured invocation after all other cases as an order-effect control.
    slots.append({**slots[0], "slot": len(slots), "round": "order_control", "repeats": 0})
    output.mkdir(parents=True, mode=0o700)
    result = {"schema": SCHEMA + ".prepared", "selection": selection_pin, "evaluation_plan": selection["evaluation_plan"],
              "panel": selection["panel"], "snapshot": selection["snapshot"], "subjects": selected,
              "cases": cases, "cells": cells, "slots": slots, "domain": domain,
              "domain_sha256": hashlib.sha256(json.dumps(domain, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
              "toolchain": tools, "paid_jobs_launched": 0, "physical_performance_verified": False}
    write(output / "prepared.json", result)
    return result


def _declare_blob(raw, output, blobs, declarations):
    digest = hashlib.sha256(raw).hexdigest()
    if digest not in blobs:
        name = "panel_blob_" + digest
        binary = output / (name + ".bin")
        binary.write_bytes(raw)
        blobs[digest] = (name, binary)
        declarations.append(f"extern const unsigned char _binary_{name}_bin_start[];")
    return "_binary_" + blobs[digest][0] + "_bin_start"


def render(prepared, output):
    """Generate a complete-call harness and independent exact-byte checks outside ROI."""
    import numpy as np

    output = Path(output)
    slots = prepared["slots"]
    tensors, maxima = {}, {}
    declarations, functions, calls, blobs = [], [], [], {}
    for slot in slots:
        cb = read(slot["command_buffer"])
        for name, tensor in cb["tensors"].items():
            if name not in slot["abi"]:
                raise ValueError("ABI changed after preparation")
            size = math.prod(tensor["shape"]) * WIDTHS[tensor["dtype"]]
            maxima[name] = max(maxima.get(name, 0), size)
        tensors[slot["slot"]] = cb["tensors"]
    for name, size in maxima.items():
        if not _token(name) or not name.replace("_", "a").isalnum() or name[0].isdigit():
            raise ValueError("tensor must have a C identifier name")
        declarations.append(f"static unsigned char panel_tensor_{name}[{size}] __attribute__((aligned(4096)));")
    for slot in slots:
        index = slot["slot"]
        declarations.append(f"extern void K{index}_entry({', '.join('void *' for _ in slot['abi'])});")
        declarations.append(f"extern void K{index}_complete(void);")
        setup, checks = [], []
        with np.load(reopen(slot["fixtures"]["inputs"]), allow_pickle=False) as inputs, \
                np.load(reopen(slot["fixtures"]["outputs"]), allow_pickle=False) as outputs:
            declared_outputs = {n for n, t in tensors[index].items() if t["role"] == "output"}
            declared_inputs = set(tensors[index]) - declared_outputs
            if set(inputs.files) != declared_inputs or set(outputs.files) != declared_outputs:
                raise ValueError("fixtures must name every input and every live output exactly once")
            for name, tensor in tensors[index].items():
                is_output = name in declared_outputs
                raw = _array_bytes(outputs if is_output else inputs, name, tensor)
                ref = _declare_blob(raw, output, blobs, declarations)
                dst = "panel_tensor_" + name
                setup.append(f"for (size_t j=0;j<{len(raw)};j++) ((volatile unsigned char*){dst})[j] = "
                             + (f"(unsigned char)~{ref}[j];" if is_output else f"{ref}[j];"))
                if is_output:
                    checks.append(f"for (size_t j=0;j<{len(raw)};j++) mismatches += ((volatile unsigned char*){dst})[j] != {ref}[j];")
        arguments = ", ".join("panel_tensor_" + name for name in slot["abi"])
        functions.append(f'''static void panel_slot_{index}(void) {{
  {' '.join(setup)}
  panel_prepare();
  const uint64_t start = panel_read_cycles();
  K{index}_entry({arguments}); K{index}_complete();
  const uint64_t end = panel_read_cycles();
  size_t mismatches = 0;
  {' '.join(checks)}
  panel_puts("PANEL {index} "); panel_number(end-start); panel_puts(" ");
  panel_number(mismatches); panel_puts("\\n");
}}''')
        calls.append(f"  panel_slot_{index}();")
    header = str(reopen(prepared["toolchain"]["runtime_header"]))
    text = f'''#include <stdint.h>
#include <stddef.h>
#include {json.dumps(header)}
static void panel_number(uint64_t value) {{
  char out[21]; unsigned i=20; out[i]=0;
  do {{ out[--i]=(char)('0'+value%10); value/=10; }} while(value);
  panel_puts(out+i);
}}
'''
    text += "\n".join(dict.fromkeys(declarations)) + "\n" + "\n".join(functions)
    text += '\nint main(void) {\n  panel_init();\n  panel_puts("PANEL_BEGIN\\n");\n'
    text += "\n".join(calls) + '\n  panel_puts("PANEL_DONE\\n"); panel_exit(0); return 0;\n}\n'
    harness = output / "harness.c"
    harness.write_text(text)
    return harness, blobs


def _run(argv, *, cwd, log):
    with log.open("ab") as stream:
        subprocess.run([str(a) for a in argv], cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=600)


def build(prepared_path):
    """Compile and link a prepared image; no emulator, hardware or cloud operation."""
    prepared_path = Path(prepared_path).absolute()
    prepared = read(pin(prepared_path))
    output = prepared_path.parent
    if (output / "build.json").exists():
        raise ValueError("build already exists; preserve immutable output")
    for item in prepared["slots"]:
        for key in ("object", "command_buffer", "build_receipt"):
            reopen(item[key])
        for key in ("inputs", "outputs"):
            reopen(item["fixtures"][key])
    tools = prepared["toolchain"]
    for item in tools.get("runtime_dependencies", []):
        reopen(item)
    for item in tools.get("include_trees", []):
        if hash_tree(_canonical_directory(item["path"]))["sha256"] != item["sha256"]:
            raise ValueError("toolchain include tree changed")
    compiler, objcopy, nm = [str(reopen(tools[key])) for key in ("compiler", "objcopy", "nm")]
    harness, blobs = render(prepared, output)
    objects, log = [], output / "build.log"
    for slot in prepared["slots"]:
        source = reopen(slot["object"])
        symbols = subprocess.check_output([nm, "--defined-only", "-g", str(source)], text=True, timeout=30)
        names = [line.split()[-1] for line in symbols.splitlines() if line.split()]
        if "kernel_entry" not in names or "kernel_complete" not in names:
            raise ValueError("object lacks entry/completion symbols")
        arguments = [f"--redefine-sym={name}=" + (f"K{slot['slot']}_entry" if name == "kernel_entry" else
                     f"K{slot['slot']}_complete" if name == "kernel_complete" else f"K{slot['slot']}_{name}")
                     for name in names]
        target = output / f"kernel_{slot['slot']}.o"
        _run([objcopy, *arguments, source, target], cwd=output, log=log)
        objects.append(target)
    for name, binary in blobs.values():
        target = output / (name + ".o")
        _run([objcopy, "-I", "binary", "-O", "elf64-littleriscv", "-B", "riscv", "--rename-section",
              ".data=.rodata.panel,alloc,load,readonly,data,contents", binary.name, target.name], cwd=output, log=log)
        objects.append(target)
    flags = tools["cflags"]
    dependency_file = output / "harness.d"
    includes = ["-I" + str(_canonical_directory(p)) for p in tools.get("include_directories", [])]
    obj = output / "harness.o"
    _run([compiler, *flags, *includes, "-MMD", "-MF", dependency_file, "-c", harness, "-o", obj], cwd=output, log=log)
    elf = output / "panel.elf"
    _run([compiler, *flags, "-T", reopen(tools["linker_script"]), obj, *objects,
          *[reopen(p) for p in tools["support_objects"]], *tools.get("ldflags", []), "-o", elf], cwd=output, log=log)
    if elf.read_bytes()[:4] != b"\x7fELF":
        raise ValueError("compiler did not produce an ELF")
    # The compiler reports the concrete transitive local headers it consumed.
    # Keep those pins beside the selected tool/runtime inputs; import reopens all.
    import shlex

    dependency_text = dependency_file.read_text().replace("\\\n", " ")
    _, separator, dependency_names = dependency_text.partition(":")
    if not separator:
        raise ValueError("compiler did not record its consumed header dependencies")
    dependencies = [pin((output / name).resolve()) for name in shlex.split(dependency_names)]
    result = {"schema": SCHEMA + ".build", "prepared": pin(prepared_path), "elf": pin(elf),
              "harness": pin(harness), "included_dependencies": dependencies, "blobs": [pin(binary) for _, binary in blobs.values()],
              "domain_sha256": prepared["domain_sha256"], "paid_jobs_launched": 0,
              "physical_performance_verified": False}
    write(output / "build.json", result)
    return result


def legacy_runtime(config_path, output):
    """Emit an executable adapter for the existing pinned HTIF primitive-sweep runtime.

    Config pins timing_header, accelerator_header and common_c; sweep_bytes is
    explicit and cache_policy must disclose that a sweep is not a cold-cache proof.
    Include directories and CRT/HTIF/libc/linker pins remain in the build selection.
    """
    config_pin = pin(config_path)
    config = read(config_pin)
    output = Path(output).absolute()
    if output.exists():
        raise ValueError("runtime output already exists")
    for key in ("timing_header", "accelerator_header", "common_c"):
        reopen(config[key])
    count = config["sweep_bytes"]
    if type(count) is not int or count <= 0 or count % 1024:
        raise ValueError("primitive sweep must be a positive multiple of 1024 bytes")
    output.parent.mkdir(parents=True, exist_ok=True)
    text = '\n'.join([
        '#include <stddef.h>',
        '#include ' + json.dumps(str(reopen(config["timing_header"]))),
        '#include ' + json.dumps(str(reopen(config["accelerator_header"]))),
        f'static unsigned char SWEEP[{count}] NOINIT;',
        '#include ' + json.dumps(str(reopen(config["common_c"]))),
        'static void panel_init(void) { console_init(); gemmini_flush(0); }',
        'static void panel_prepare(void) { l2_sweep(); }',
        'static uint64_t panel_read_cycles(void) { return rd_mcycle(); }',
        'static void panel_puts(const char *text) { htif_puts(text); }',
        'static void panel_exit(int status) { htif_exit(status); }', '',
    ])
    with output.open("x") as stream:
        stream.write(text)
    result = {"schema": SCHEMA + ".runtime", "config": config_pin, "header": pin(output),
              "sweep_bytes": count, "cache_claim": "declared sweep; cold cache is not established",
              "dependency_pins": {key: config[key] for key in ("timing_header", "accelerator_header", "common_c")}}
    write(output.with_suffix(output.suffix + ".json"), result)
    return result


def _uart(path, expected):
    text = path.read_text()
    rows, began, done = {}, 0, 0
    for line in text.splitlines():
        if line == "PANEL_BEGIN":
            began += 1
        elif line == "PANEL_DONE":
            done += 1
        elif line.startswith("PANEL "):
            fields = line.split()
            if len(fields) != 4:
                raise ValueError("malformed physical row")
            slot, cycles, mismatches = map(int, fields[1:])
            if began != 1 or done or slot in rows or slot not in expected or cycles <= 0 or mismatches < 0:
                raise ValueError("duplicate/foreign/nonpositive physical row")
            rows[slot] = {"cycles": cycles, "mismatched_bytes": mismatches}
    if began != 1 or done > 1:
        raise ValueError("ambiguous physical UART framing")
    return rows, done == 1


def collect(build_path, observation_path):
    """Import exact host observations; no callbacks, hardware launches or author feedback."""
    from merlin_experiments.post_freeze.booted_elf import _boot_path

    built = read(pin(build_path))
    prepared = read(built["prepared"])
    observation = read(pin(observation_path))
    if observation.get("schema") != OBSERVATION or observation.get("domain_sha256") != prepared["domain_sha256"]:
        raise ValueError("physical observation names another measurement domain")
    reopen(built["elf"])
    for item in built.get("included_dependencies", []):
        reopen(item)
    boot = read(observation["booted_elf"])
    invocation = read(observation["ssm_invocation"])
    _boot_path(boot, reopen(observation["sim_run"]))
    consumed = boot.get("consumed_elf", {})
    uploaded = observation["uploaded_elf"]
    if (pin(reopen(uploaded))["sha256"] != built["elf"]["sha256"]
            or consumed.get("sha256") != built["elf"]["sha256"]
            or boot.get("agfi") != prepared["domain"]["agfi"]
            or boot.get("ssm_command_id") != invocation.get("CommandId")
            or invocation.get("Status") != "Success" or invocation.get("ResponseCode") != 0):
        raise ValueError("physical boot, upload, deployment or successful SSM identity differs")
    for subject in read(prepared["evaluation_plan"])["subjects"]:
        if subject["id"] in prepared["subjects"] and hash_tree(subject["source"]["path"])["sha256"] != subject["source"]["sha256"]:
            raise ValueError("frozen compiler changed after physical build")
    actual = read(observation["runtime_domain"])
    if actual.get("domain") != prepared["domain"] or actual.get("ssm_command_id") != invocation["CommandId"]:
        raise ValueError("runtime-domain evidence does not bind the selected execution")
    uart_pin = observation["uart"]
    observed, terminated = _uart(reopen(uart_pin), {s["slot"] for s in prepared["slots"]})
    samples = []
    for slot in prepared["slots"]:
        row = observed.get(slot["slot"])
        state = "missing" if row is None else "fail" if row["mismatched_bytes"] else "pass"
        samples.append({**{k: slot[k] for k in ("slot", "subject", "case", "round", "compiler_sha256", "object")},
                        "state": state, "cycles": row["cycles"] if row else None,
                        "mismatched_bytes": row["mismatched_bytes"] if row else None})
    first, last = samples[0], samples[-1]
    order_ppm = None
    order_pass = False
    if first["state"] == last["state"] == "pass":
        order_ppm = abs(last["cycles"] - first["cycles"]) * 1_000_000 / first["cycles"]
        limit = read(prepared["selection"]).get("order_effect_bound_ppm", 10000)
        order_pass = order_ppm <= limit
    cells = []
    for supplied in prepared["cells"]:
        rows = [s for s in samples if s["subject"] == supplied["subject"] and s["case"] == supplied["case"]
                and type(s["round"]) is int]
        if any(r["state"] == "fail" for r in rows):
            state, reason = "fail", "physical all-output comparison failed"
        elif not terminated or len(rows) != 3 or any(r["state"] != "pass" for r in rows):
            state, reason = "missing", supplied.get("reason", "three completed physical samples are absent")
        elif not order_pass:
            state, reason = "unqualified", "batch order-effect control failed or is unavailable"
        else:
            state, reason = "pass", None
        cycles = [r["cycles"] for r in rows]
        cells.append({"subject": supplied["subject"], "snapshot": prepared["snapshot"], "case": supplied["case"],
                      "compiler_sha256": supplied["compiler_sha256"], "state": state, "reason": reason,
                      "ordered_samples": cycles, "sample_slots": [r["slot"] for r in rows],
                      "cycles_median": statistics.median(cycles) if state == "pass" else None,
                      "domain_sha256": prepared["domain_sha256"], "elf": built["elf"], "uart": uart_pin,
                      "all_output_bytes_checked": state == "pass"})
    return {"schema": OBSERVATION + ".collection", "build": pin(build_path), "observation": pin(observation_path),
            "evaluation_plan": prepared["evaluation_plan"], "panel": prepared["panel"], "domain": prepared["domain"],
            "domain_sha256": prepared["domain_sha256"], "cells": cells, "samples": samples,
            "complete": terminated and all(c["state"] in ("pass", "fail") for c in cells),
            "order_effect_ppm": order_ppm, "order_effect_pass": order_pass,
            "deployment_source_equivalence": "UNKNOWN; separate reviewed correspondence required",
            "scientific_qualification": False, "paid_jobs_launched": 0}


def apply_physical(collection, paths):
    """Attach observations only after replaying their original private custody join."""
    by_key = {}
    for path in paths:
        claimed = read(pin(path))
        replay = collect(reopen(claimed["build"]), reopen(claimed["observation"]))
        if replay != claimed:
            raise ValueError("physical collection differs from replayed evidence")
        if replay["evaluation_plan"] != collection["plan"]:
            raise ValueError("physical observations belong to another evaluation plan")
        for cell in replay["cells"]:
            key = cell["subject"], cell["case"]
            if key in by_key:
                raise ValueError("duplicate physical subject/case cell")
            by_key[key] = cell
    expected = {(c["subject"], c["case"]) for c in collection["cells"]}
    if not set(by_key) <= expected:
        raise ValueError("physical collection includes foreign cases or subjects")
    for cell in collection["cells"]:
        row = by_key.get((cell["subject"], cell["case"]))
        if row is None:
            continue
        if row["compiler_sha256"] != cell["source"]["sha256"]:
            raise ValueError("physical cell belongs to another frozen compiler")
        cell["physical"] = {**row, "status": row["state"], "cycles": row["cycles_median"],
                            "ordinary_numeric_pass": cell["state"] == "pass"}
    plan = read(collection["plan"])
    baseline_ids = {s["id"] for s in plan["subjects"] if s.get("role") == "initial_baseline" or s["id"] == "baseline"}
    lookup = {(c["subject"], c["case"]): c for c in collection["cells"]}
    for cell in collection["cells"]:
        if cell["subject"] in baseline_ids:
            continue
        other = lookup.get((cell.get("baseline_subject", "baseline"), cell["case"]))
        left, right = cell.get("physical", {}), (other or {}).get("physical", {})
        valid = (cell["state"] == "pass" and (other or {}).get("state") == "pass"
                 and left.get("state") == right.get("state") == "pass"
                 and left.get("domain_sha256") == right.get("domain_sha256"))
        cell["physical_comparison"] = {"status": "measured" if valid else "UNKNOWN",
                                       "candidate_to_baseline_ratio": left["cycles"] / right["cycles"] if valid else None}
    for summary in collection.get("summaries", []):
        cells = [c for c in collection["cells"] if c["subject"] == summary["subject"]
                 and c["split"] == summary["split"]]
        ratios = [(c.get("physical_comparison") or {}).get("candidate_to_baseline_ratio") for c in cells]
        complete = bool(cells) and all(type(r) in (int, float) and math.isfinite(r) and r > 0 for r in ratios)
        summary["physical_status"] = "measured" if complete else "UNKNOWN"
        summary["physical_candidate_to_baseline_geomean"] = (
            math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if complete else None)
        summary["physical_coverage"] = {
            "required": len(cells),
            "measured": sum(c.get("physical", {}).get("state") == "pass" for c in cells)}
    collection["physical_timing_qualified"] = False
    collection["physical_measurements_imported"] = len(by_key)
    collection["physical_scope"] = "Exact uploaded/booted image, selected runtime domain and all-output checks; source/build correspondence remains separate."
    return collection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--selection", type=Path, required=True)
    prepare_parser.add_argument("--out", type=Path, required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--prepared", type=Path, required=True)
    runtime_parser = sub.add_parser("legacy-runtime")
    runtime_parser.add_argument("--config", type=Path, required=True)
    runtime_parser.add_argument("--out", type=Path, required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--build", type=Path, required=True)
    collect_parser.add_argument("--observation", type=Path, required=True)
    collect_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        result = prepare(args.selection, args.out)
    elif args.action == "build":
        result = build(args.prepared)
    elif args.action == "legacy-runtime":
        result = legacy_runtime(args.config, args.out)
    else:
        result = collect(args.build, args.observation)
        write(args.out, result)
    print(json.dumps({"schema": result["schema"], "paid_jobs_launched": 0}))


if __name__ == "__main__":
    main()
