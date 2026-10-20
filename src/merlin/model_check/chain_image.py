"""Link completed program objects into one complete recurrent guest image.

This does not invoke a candidate compiler, evaluate a model or launch hardware.
The new guest/link/memory evidence never transfers old numerical or placement
verdicts. Original completed objects and each actual transformation are reopened.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from merlin.common.tree_hash import hash_tree
from merlin.llvmlower.compilation_recipe import CompilationRecipe
from merlin.runtime.execution_memory import MemoryMapBinding, MemoryReservation, admit_execution_memory

from . import chain_image_abi as abi
from . import chain_image_boot as boot
from . import chain_image_data as data
from . import chain_image_layout as layout
from . import chain_image_objects as objects
from . import chain_image_plan as planning
from . import chain_image_sections as sections

SCHEMA = "merlin.chain_image.selection.v1"


def _compiler_binding(program, compiler_sha256):
    receipt = program["production"]["receipt"]
    device = receipt["inputs"].get("device")
    if device is None:
        if (
            receipt.get("execution_route") != "host_baseline"
            or receipt["output"].get("device_artifacts") is not None
            or receipt["output"].get("matrix_routing") is not None
        ):
            raise ValueError("program lacks an explicit ordinary host-only build route")
        return {"execution_route": "host_baseline", "device_compiler_sha256": None}
    if device["package_tree"]["sha256"] != compiler_sha256:
        raise ValueError("program objects have another device compiler")
    return {"execution_route": receipt["execution_route"], "device_compiler_sha256": compiler_sha256}


def _run(argv, *, timeout, cwd=None):
    return subprocess.run([str(x) for x in argv], cwd=cwd, timeout=timeout, capture_output=True, text=True, check=True)


def _tool(compiler, name, timeout):
    selected = _run([compiler, "-print-prog-name=" + name], timeout=timeout).stdout.strip()
    path = Path(selected)
    if not path.is_absolute():
        # Do not select a second, PATH-discovered binutils installation.
        path = compiler.parent / path
    return objects.pin(path.resolve())


def _same_runtime(programs):
    seed = programs[0]["production"]
    selected = {name: row for name, row in seed["by_name"].items() if name in objects.COMMON_OBJECTS}
    for program in programs:
        own = program["production"]
        if own["link"]["executable"] != seed["link"]["executable"]:
            raise ValueError("chain programs selected different final-link compilers")
        common = {name: row for name, row in own["by_name"].items() if name in objects.COMMON_OBJECTS}
        if common.keys() != selected.keys() or any(
            common[name]["sha256"] != selected[name]["sha256"] for name in selected if name != "malloc.o"
        ):
            raise ValueError("chain programs disagree on the shared runtime/math/boot objects")

        def libraries(p):
            return sorted(row["sha256"] for row in p["link"]["inputs"] if row["path"].endswith(".a"))

        if libraries(own) != libraries(seed):
            raise ValueError("chain programs selected different archive suppliers")
    return selected


def _blob(recipe, record, name, directory, compile_command, timeout):
    source, product = directory / f"{name}.S", directory / f"{name}.o"
    source.write_text(f'.section .merlin_chain_{name},"a"\n.incbin {json.dumps(str(objects.reopen(record)))}\n')
    compiler = compile_command["argv"][0]
    abi = [
        arg for arg in compile_command["argv"][1:] if arg.startswith(("-march=", "-mabi=", "-mcmodel=", "-m32", "-m64"))
    ]
    recipe.run(
        [compiler, *abi, "-c", source, "-o", product],
        runner=lambda argv, **kw: _run(argv, timeout=timeout, **kw),
        inputs=(source, objects.reopen(record)),
        output=product,
    )
    return product


def build(selection_path: Path, destination: Path):
    selection_pin = objects.pin(selection_path)
    selection = json.loads(objects.payload(selection_pin))
    timeout = selection["command_timeout_s"]
    if (
        selection.get("schema") != SCHEMA
        or selection.get("transport") not in data.TRANSPORTS
        or type(timeout) not in (int, float)
        or not math.isfinite(timeout)
        or not 0 < timeout <= 600
    ):
        raise ValueError("select explicit chain transport and a finite bounded compilation timeout")
    if destination.exists() or destination.is_symlink() or destination.parent.resolve() != destination.parent:
        raise ValueError("chain image output must be a fresh canonical directory")
    compiler = selection["compiler"]
    if destination.is_relative_to(Path(compiler["path"])):
        raise ValueError("chain packaging cannot write into the frozen compiler")
    if hash_tree(Path(compiler["path"]))["sha256"] != compiler["sha256"]:
        raise ValueError("selected frozen compiler changed")
    destination.mkdir()
    plan = planning.prepare(selection, destination)
    compiler_bindings = {}
    for program in plan["programs"]:
        original = program["production"]["receipt"]["inputs"]
        compiler_bindings[program["name"]] = _compiler_binding(program, compiler["sha256"])
        if original["host_math_policy"] != plan["host_math_policy"]:
            raise ValueError("program objects have another host math policy")
    common = _same_runtime(plan["programs"])
    seed = plan["programs"][0]["production"]
    compile_command = layout.command(seed, "model_main.o")
    if any("MERLIN_CHAIN_HOST_FIXTURE" in flag for flag in compile_command["argv"]):
        raise ValueError("host-only fixture entry must not select production chain execution")
    gcc = Path(compile_command["executable"]["path"])
    tools = {name: _tool(gcc, name, timeout) for name in ("nm", "objcopy")}
    names = [
        objects.namespace(
            p["production"],
            i,
            destination / f"program{i}",
            nm=Path(tools["nm"]["path"]),
            objcopy=Path(tools["objcopy"]["path"]),
            timeout=timeout,
        )
        for i, p in enumerate(plan["programs"])
    ]
    reference = data.references(selection, plan, destination)
    storage = layout.layout(selection, plan, reference)
    counter = selection["counter"]
    counter_program = names[counter["program"]]
    if counter["symbol"] not in {
        line.split()[0] for line in Path(counter_program["mapping"]["path"]).read_text().splitlines()
    }:
        raise ValueError("selected counter is not defined by its retained program objects")
    objects.reopen(counter["contract"])
    header = destination / "chain_data.h"
    header.write_text(
        data.render(
            plan,
            names,
            storage,
            reference,
            transport=selection["transport"],
            counter_symbol=counter_program["prefix"] + counter["symbol"],
        )
    )
    from merlin.common.paths import runtime_dir

    runtime = runtime_dir() / "c"
    source = runtime / "chain_guest.c"
    selected_header = destination / "merlin_model.h"
    selected_header.write_bytes((runtime / "merlin_model.h").read_bytes())
    recipe = CompilationRecipe(destination, producer=Path(__file__))
    recipe.bind_preparation("selected_chain", selection_path)
    recipe.bind_preparation("generated_plan", header)
    driver = destination / "chain_guest.o"

    def run(argv, **kw):
        return _run(argv, timeout=timeout, **kw)

    abi_records = [
        abi.verify(p, selected_header, recipe=recipe, runner=run, directory=destination) for p in plan["programs"]
    ]
    runtime_command = layout.command(seed, "merlin_model.o")
    runtime_sources = [objects.reopen(row) for row in runtime_command["inputs"] if row["path"].endswith(".c")]
    if len(runtime_sources) != 1:
        raise ValueError("retained descriptor runtime lacks its original C source")
    descriptor_source = destination / "descriptor_runtime.c"
    descriptor_source.write_bytes(runtime_sources[0].read_bytes())
    descriptor_object = destination / "descriptor_runtime.o"
    recipe.run(
        layout.compile_argv(runtime_command, descriptor_source, descriptor_object),
        runner=run,
        inputs=(descriptor_source, runtime_sources[0], selected_header),
        output=descriptor_object,
    )
    recipe.run(
        layout.compile_argv(compile_command, source, driver, includes=(destination, runtime)),
        runner=run,
        inputs=(source, header, runtime / "merlin_model.h"),
        output=driver,
    )
    malloc_command = layout.command(seed, "malloc.o")
    malloc_sources = [Path(row["path"]) for row in malloc_command["inputs"] if row["path"].endswith(".c")]
    if len(malloc_sources) != 1:
        raise ValueError("retained arena compilation has no unique original runtime source")
    arena = destination / "arena.o"
    recipe.run(
        layout.compile_argv(
            malloc_command,
            malloc_sources[0],
            arena,
            defines=(
                ("MERLIN_ARENA_BASE_ADDR", str(storage["arena"]["begin"]) + "ULL"),
                ("MERLIN_ARENA_SIZE_BYTES", str(storage["arena"]["bytes"]) + "ULL"),
            ),
        ),
        runner=run,
        inputs=tuple(objects.reopen(row) for row in malloc_command["inputs"]),
        output=arena,
    )
    blobs = [
        _blob(recipe, plan["input_blob"], "inputs", destination, compile_command, timeout),
        _blob(recipe, reference["blob"], "references", destination, compile_command, timeout),
    ]
    scripts = [objects.reopen(row) for row in seed["link"]["inputs"] if row["path"].endswith(".ld")]
    if len(scripts) != 1:
        raise ValueError("retained program lacks one original selected linker layout")
    script = layout.linker_script(scripts[0], destination / "chain.ld", storage)
    linked = [objects.reopen(row) for name, row in common.items() if name not in {"malloc.o", "merlin_model.o"}]
    linked += [
        descriptor_object,
        arena,
        driver,
        *blobs,
        *(objects.reopen(obj) for name in names for obj in name["objects"]),
    ]
    elf = destination / "chain.elf"
    argv, libraries = layout.final_link(
        seed["link"], objects=linked, script=script, output=elf, wraps=[name for row in names for name in row["wraps"]]
    )
    suppliers = {}
    for program, renamed in zip(plan["programs"], names, strict=True):
        correspondence = dict(
            zip((r["path"] for r in renamed["original_objects"]), (r["path"] for r in renamed["objects"]), strict=True)
        )
        mapping = dict(line.split() for line in Path(renamed["mapping"]["path"]).read_text().splitlines())
        for symbol, record in (program["production"]["recipe"].get("link_suppliers") or {}).get("symbols", {}).items():
            original = record["selected_input"]
            replacement = correspondence.get(original["path"])
            if replacement is None:
                replacement = next(
                    (str(p) for p in [*linked, *libraries] if objects.pin(p)["sha256"] == original["sha256"]), None
                )
            if replacement is None:
                raise ValueError("original defining supplier was omitted from the chain link")
            selected_symbol = mapping.get(symbol, symbol)
            if selected_symbol in suppliers and suppliers[selected_symbol] != Path(replacement):
                raise ValueError("chain programs disagree on a defining supplier")
            suppliers[selected_symbol] = Path(replacement)
    argv[-2:-2] = ["-Wl,--trace-symbol=" + symbol for symbol in sorted(suppliers)]
    result = recipe.run(argv, runner=run, inputs=(*linked, *libraries, script), output=elf)
    if suppliers:
        recipe.record_link_suppliers(suppliers, result)
    memory_selection = selection["memory_map"]
    binding = MemoryMapBinding(
        objects.reopen(memory_selection), memory_selection["sha256"], selection["execution_identity"]
    )
    memory = admit_execution_memory(
        elf, binding, (MemoryReservation("activation_arena", **storage["arena"], alignment=64),)
    )
    boot_record = boot.verify(
        elf, selection["boot"], nm=Path(tools["nm"]["path"]), directory=destination, timeout=timeout
    )
    for record in [
        selection_pin,
        *plan["source_pins"],
        *reference["source_pins"],
        *storage["source_pins"],
        counter["contract"],
    ]:
        objects.reopen(record)
    for program in plan["programs"]:
        objects.completed_program(program["production"]["receipt_pin"])
    if hash_tree(Path(compiler["path"]))["sha256"] != compiler["sha256"]:
        raise ValueError("compiler changed during chain packaging")
    recipe.completed(elf)
    contract_path = objects.reopen(selection["workload_contract"])
    manifest = {
        "schema": "merlin.post_freeze.firesim.chain_image.v1",
        "phase": selection["phase"],
        "model": selection["model"],
        "execution_scope": "complete_chain",
        "compiler_sha256": compiler["sha256"],
        "program_compiler_bindings": compiler_bindings,
        "contract_sha256": selection["workload_contract"]["sha256"],
        "workload_contract": objects.pin(contract_path),
        "boot_elf": {"role": "boot_elf", **objects.pin(elf)},
        "chain_steps": len(plan["schedule"]),
        "output_roster": [
            {"stage": p["name"], "step": step, "index": index, "dtype": dtype, "shape": shape}
            for p in plan["programs"]
            for step in range(p["steps"])
            for index, (shape, dtype) in enumerate(p["outputs"])
        ],
        "transport": selection["transport"],
        "plan": planning.public_plan(plan),
        "namespace": names,
        "reference": reference,
        "memory_admission": memory,
        "counter": counter,
        "boot": boot_record,
        "argument_abi": abi_records,
        "recipe": objects.pin(recipe.path),
        "selection": selection_pin,
        "producer": objects.pin(Path(__file__)),
        "scientific_qualification": False,
        "producer_sources": [
            objects.pin(Path(p.__file__)) for p in (data, boot, abi, layout, objects, planning, sections)
        ],
        "scope": "New retained-object chain packaging; numerical/placement/hardware qualification pending.",
    }
    (destination / "chain-image.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    build(args.selection.absolute(), args.out.absolute())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
