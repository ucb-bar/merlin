"""The package's answer to an open model's device part, run WITHOUT the model's host code.

A whole open model on a board is hours of host code; what a package changes is its kernels at the
model's regimes. :func:`build_kernel_bench` states the device part exactly as
:func:`merlin.analysis.perf.whole_model_open.build` does and links one program that runs each dispatch the
package answers once, on seeded operands, projection-checked and digested; :func:`grade_kernel_bench`
reads its console. Re-exported by :mod:`merlin.analysis.perf.whole_model_open`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

BENCH_SCHEMA = "whole_model_kernel_bench_v1"


def build_kernel_bench(
    package_dir: str | Path,
    model_capsule: str | Path,
    *,
    target: str,
    machine: str,
    header: str | Path,
    header_sha256: str | None = None,
    out: str | Path,
    timeout: int = 600,
    jobs: int | None = None,
    decline: Sequence[Any] = (),
    prohibited_roles: Sequence[str] = (),
    dram_base: int = 0x80000000,  # derived-ok: the RISC-V platform DRAM base the bare-metal harness links at
    evict_bytes: int = 0,
    check: bool = True,
    only: Sequence[int] | None = None,
    dram_bytes: int | None = None,
    compile_timeout: int = 7200,
    prune: bool = True,
) -> dict[str, Any]:
    """Build the package's answer to an open model's device part as a program of its own: every dispatch
    the package answers, run once at its model shape on operands of a stated seed, projection-checked and
    digested, with no host code (the target driver's ``render_kernel_bench``).

    The whole model on a board takes hours because of its host code; this program takes minutes and
    measures what a package changes -- its kernels, at the model's regimes -- on the same machine, and
    shows each kernel correct there (exact projection) and equal to a functional-simulator run of the
    same image (``GM_WORDS``). It is not the model's cycle count: the host code and the cache state it
    leaves are absent, and the record says so. ``evict_bytes`` is written between each kernel's operand
    fill and its call, so its operands arrive cold as a model's weights do (stated, recorded). Written
    to ``<out>/kernel_bench_build.json``."""
    import os

    from merlin.analysis.perf import whole_model_build as WMB
    from merlin.common import provenance as PROV
    from merlin.runtime.backends import base as backends
    from merlin.runtime.backends import spike_model as SM

    capsule = WMB.load_model_capsule(model_capsule)
    abi_header = WMB.machine_header(machine, header, header_sha256)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = jobs or min(16, os.cpu_count() or 1)
    driver = backends.whole_model_driver(target)
    if not hasattr(driver.dispatch, "render_kernel_bench"):
        raise WO.OpenModelError(f"the {target} driver writes no kernel bench")
    # The machine first: a header it cannot use, or one with no full-width readout, refuses the bench
    # before the package is asked anything.
    recipe = WMB._with_header(backends.harness_build_recipe(target), Path(header), out / "harness")
    includes = [f"-I{root}" for root in recipe.include_roots]
    parameter_header = driver.program._parameter_header(None, includes)
    full_width = driver.program.full_width_readout(parameter_header)
    if full_width is not True:
        raise WO.OpenModelError(
            f"{WO.MACHINE_CANNOT_READ_OUT}: {machine!r}'s parameter header states no full-width accumulator "
            "readout, so no kernel committing the accumulator can run on its unit"
        )
    part = WO._device_part(
        capsule, target=target, package_dir=package_dir, out=out, timeout=timeout, jobs=jobs, decline=decline
    )
    routes: dict[int, dict[str, Any]] = {}
    for dispatch in part["dispatches"]:
        row = part["by_group"][dispatch.group]
        statement = part["stated"][dispatch.group]
        if row["on"] != WMB.ON_PACKAGE:
            routes[dispatch.group] = {"on": WMB.ON_HOST, "cause": row.get("cause") or "not_answered_by_the_package"}
            continue
        operands = {str(v): str(k) for k, v in (statement.get("operands") or {}).items()}
        roles = [operands.get(str(a.get("program"))) for a in row["args"]]
        if None in roles or any(a.get("gather") for a in row["args"]):
            routes[dispatch.group] = {"on": WMB.ON_HOST, "cause": "kernel_argument_not_a_dispatch_operand"}
            continue
        routes[dispatch.group] = {"on": WMB.ON_PACKAGE, "symbol": row["symbol"], "object": row["object"], "args": roles}
    rendered = driver.dispatch.render_kernel_bench(
        [{**d.to_dict(), "op": part["stated"][d.group].get("op")} for d in part["dispatches"]],
        routes,
        uart=driver.program.UART,
        words_helper=getattr(driver.program, "_WORDS_HELPER", ""),
        evict_bytes=int(evict_bytes),
        check=check,
        only=only,
        row_padding=WMB.pointee_row_padding(target)["multiple"],
    )
    program = out / "program"
    program.mkdir(parents=True, exist_ok=True)
    (program / "bench.c").write_text(rendered["source"], encoding="utf-8")

    kernels = [Path(o) for o in rendered["objects"]]
    code_reserve = SM._CODE_RESERVE_FIXED + sum(k.stat().st_size for k in kernels)
    megabyte = 1 << 20
    arena_base = (int(dram_base) + code_reserve + megabyte - 1) & ~(megabyte - 1)
    benched = [r for r in rendered["census"] if r.get("in_bench")]
    largest = sum(max((r["m"] * r["k"], r["k"] * r["n"], 4 * r["m"] * r["n"])[i] for r in benched) for i in range(3))
    arena_bytes = (largest + int(evict_bytes) + 64 * megabyte + megabyte - 1) & ~(megabyte - 1)
    end = arena_base + arena_bytes
    if dram_bytes is not None and end - int(dram_base) > int(dram_bytes):
        raise WO.OpenModelError(
            f"the bench needs {end - int(dram_base):#x} bytes of DRAM and {machine!r} states {int(dram_bytes):#x}"
        )
    harness = SM._harness_dir()
    gcc_flags = [*WO._cross_flags(recipe.cflags), "-O2", "-ffreestanding", "-fno-builtin"]
    address = [f"-DMERLIN_ARENA_BASE_ADDR={hex(arena_base)}ULL", f"-DMERLIN_ARENA_SIZE_BYTES={hex(arena_bytes)}ULL"]
    units = {
        "bench.o": (program / "bench.c", [*includes, "-DBAREMETAL=1", "-Wno-incompatible-pointer-types"]),
        "crt.o": (harness / "crt.S", []),
        "console.o": (harness / "htif.c", []),
        "libc_min.o": (harness / "libc_min.c", []),
        "printf_min.o": (harness / "printf_min.c", ["-I", harness]),
        "malloc.o": (harness / "merlin_malloc.c", [*address, "-I", harness]),
    }
    objects = []
    for name, (source, extra_flags) in units.items():
        WO._run(
            [recipe.compiler, *gcc_flags, *extra_flags, "-c", source, "-o", program / name], timeout=compile_timeout
        )
        objects.append(program / name)
    elf = program / "bench.elf"
    WO._run(
        [
            recipe.compiler,
            *gcc_flags,
            "-nostdlib",
            "-nostartfiles",
            # No weights: the (empty) weights section is placed past the arena.
            f"-Wl,--defsym,MERLIN_WEIGHTS_BASE={hex(end)}",
            f"-Wl,--defsym,MERLIN_STACK_BYTES={hex(1 << 22)}",
            f"-Wl,--defsym,{SM.DRAM_BASE_SYMBOL}={hex(int(dram_base))}",
            f"-Wl,--defsym,{SM.DRAM_SPAN_SYMBOL}={hex(end - int(dram_base))}",
            "-T",
            harness / "model_link.ld",
            *objects,
            *kernels,
            "-lm",
            "-lgcc",
            "-o",
            elf,
        ],
        timeout=compile_timeout,
    )
    isa = None
    if prohibited_roles:
        from merlin.analysis.perf.isa_prohibition import check_program

        isa = check_program(
            elf,
            target=target,
            roles=prohibited_roles,
            compiler=recipe.compiler,
            group_objects={str(r["group"]): Path(r["object"]) for r in benched if r.get("object")},
            library_groups=[],
        )
        if not isa["clean"]:
            raise WO.OpenModelError(f"the linked bench issues a prohibited instruction: {isa['summary']}")
    counts: dict[str, int] = {}
    for row in rendered["census"]:
        counts[row["on"]] = counts.get(row["on"], 0) + 1
    record = {
        "schema": BENCH_SCHEMA,
        "target": target,
        "machine": machine,
        "what_it_measures": "each device dispatch the package answers, once, at its model shape, on seeded "
        "operands evicted from the caches first; no host code, so NOT the model's cycle count",
        "capsule": {
            "name": capsule.name,
            "directory": str(capsule.directory),
            "interface_sha256": WO._sha256(capsule.interface),
        },
        "package": {
            "directory": str(Path(package_dir).resolve()),
            "replies": part["buffer"]["whole_program"].get("package_replies"),
        },
        "elf": str(elf),
        "elf_sha256": WO._sha256(elf),
        "evict_bytes": int(evict_bytes),
        "projection_checked": bool(check),
        "only": None if only is None else sorted(int(g) for g in only),
        "macs": rendered["macs"],
        "layout": {"dram_base": hex(int(dram_base)), "arena_base": hex(arena_base), "arena_bytes": hex(arena_bytes)},
        "program": {
            "abi_header": abi_header,
            "compiler": str(recipe.compiler),
            "flags": gcc_flags,
            "includes": includes,
            "full_width_readout": full_width,
            "linked_kernels": [{"path": str(k), "sha256": WO._sha256(k)} for k in kernels],
        },
        "linked_objects": [{"path": str(k), "sha256": WO._sha256(k)} for k in kernels],
        "census": {"counts": counts, "per_group": rendered["census"]},
        "isa_prohibition": isa,
        "provenance": PROV.record(pins=WMB._pins_for(target), sources=[capsule.interface], artifacts={"elf": elf}),
    }
    (out / "kernel_bench_build.json").write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
    if prune:
        record["pruned_bytes"] = WO.prune_intermediates(out)
    return record


def grade_kernel_bench(uart: str, record: Mapping[str, Any], *, reference_uart: str | None = None) -> dict[str, Any]:
    """A kernel bench's console against its build record: every benched dispatch exact by projection
    (``GM_LOCAL mismatches=0``), its bracket, and -- given another run of the SAME image
    (``reference_uart``, e.g. the functional simulator's) -- equal result digests dispatch by dispatch.
    A dispatch with no line is absent, never agreed."""
    benched = [r for r in (record.get("census") or {}).get("per_group") or () if r.get("in_bench")]
    groups = [str(r["group"]) for r in benched]
    local: dict[str, dict[str, str]] = {}
    cycles: dict[str, int] = {}
    done = False
    for line in uart.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "GM_LOCAL" and len(parts) > 1:
            local[parts[1]] = dict(p.split("=", 1) for p in parts if "=" in p)
        elif parts[0] == "GM_GROUP" and len(parts) > 3 and parts[3].isdigit():
            cycles[parts[1]] = int(parts[3])
        elif parts[0] == "DONE":
            done = True
    bad = [g for g in groups if (local.get(g) or {}).get("mismatches") != "0"]
    missing = [g for g in groups if g not in cycles]
    by_route: dict[str, dict[str, int]] = {}
    for row in benched:
        slot = by_route.setdefault(str(row["on"]), {"groups": 0, "cycles": 0, "macs": 0})
        slot["groups"] += 1
        slot["cycles"] += cycles.get(str(row["group"]), 0)
        slot["macs"] += int(row["m"]) * int(row["k"]) * int(row["n"])
    regimes: dict[str, dict[str, int]] = {}
    for row in benched:
        key = f"{row['m']}x{row['k']}x{row['n']}"
        slot = regimes.setdefault(key, {"groups": 0, "cycles": 0, "macs": 0})
        slot["groups"] += 1
        slot["cycles"] += cycles.get(str(row["group"]), 0)
        slot["macs"] += int(row["m"]) * int(row["k"]) * int(row["n"])
    verdict: dict[str, Any] = {
        "completed": done,
        "local": {"groups": len(groups), "exact": len(groups) - len(bad), "disagree_or_absent": bad},
        "bracketed": {"absent": missing, "cycles": sum(cycles.get(g, 0) for g in groups), "by_route": by_route},
        "by_regime": regimes,
    }
    if reference_uart is not None:
        verdict["words"] = WO.words_bridge(reference_uart, uart)
    if record.get("projection_checked") is False:
        # NO PROJECTION IN THIS IMAGE: correct only by its digests agreeing with a checked run of the
        # same seeds, dispatch by dispatch.
        verdict["local"] = {"groups": len(groups), "checked": False}
        verdict["correct"] = done and not missing and reference_uart is not None and verdict["words"]["bridged"]
        return verdict
    verdict["correct"] = done and not bad and not missing and (reference_uart is None or verdict["words"]["bridged"])
    return verdict


# LAST, so either module can be imported first: each finds the other's names already defined.
from merlin.analysis.perf import whole_model_open as WO  # noqa: E402
