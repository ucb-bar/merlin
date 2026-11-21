"""The package's answer to an open model's device part, run WITHOUT the model's host code.

A whole open model on a board is hours of host code; what a package changes is its kernels at the
model's regimes. :func:`build_kernel_bench` states the device part exactly as
:func:`merlin.analysis.whole_model.build.open_driver.build` does and links one program that runs each dispatch the
package answers once, on seeded operands, projection-checked and digested; :func:`grade_kernel_bench`
reads its console. Re-exported by :mod:`merlin.analysis.whole_model.build.open_program`.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# ---- whole_model_open_bench --------------------------------------------------------------------

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

    from merlin.analysis.whole_model.build import open_driver as _merlin_open_driver_1
    from merlin.analysis.whole_model.build import open_program as WO
    from merlin.analysis.whole_model.build import stages as WMB
    from merlin.common import provenance as PROV
    from merlin.runtime.backends import registry as backends
    from merlin.runtime.backends.baremetal import image_link as SM
    from merlin.runtime.backends.baremetal import model_build as _model_elf_metadata
    from merlin.runtime.backends.baremetal import model_build as _model_memory_layout

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
    code_reserve = _model_memory_layout._CODE_RESERVE_FIXED + sum(k.stat().st_size for k in kernels)
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
            f"-Wl,--defsym,{_model_elf_metadata.DRAM_BASE_SYMBOL}={hex(int(dram_base))}",
            f"-Wl,--defsym,{_model_elf_metadata.DRAM_SPAN_SYMBOL}={hex(end - int(dram_base))}",
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
        from merlin.analysis.whole_model.grading.isa_prohibition import check_program

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
        record["pruned_bytes"] = _merlin_open_driver_1.prune_intermediates(out)
    return record


def grade_kernel_bench(uart: str, record: Mapping[str, Any], *, reference_uart: str | None = None) -> dict[str, Any]:
    """A kernel bench's console against its build record: every benched dispatch exact by projection
    (``GM_LOCAL mismatches=0``), its bracket, and -- given another run of the SAME image
    (``reference_uart``, e.g. the functional simulator's) -- equal result digests dispatch by dispatch.
    A dispatch with no line is absent, never agreed."""
    from merlin.analysis.whole_model.build import open_program as WO

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


# ---- whole_model_open_oracle -------------------------------------------------------------------
# An open model's oracle, recomputed in a process of its own or read back from its content-keyed cache.
#
# The oracle (:func:`merlin.analysis.whole_model.build.open_program.oracle`: the model evaluated in numpy, every device
# dispatch's operands and result digested) depends on the model and the target, never on the package
# under test, so it starts with the build and is joined only where the build first needs it. For a
# transformer it is minutes of interpreter-bound work that used to run after the device part, on the
# build's critical path; recorded by the digest of everything it is a function of, a later build of the
# same model (another candidate, a rebuild) reads it back.
#
# The child recomputes which groups are device dispatches exactly as the build's cut does (the normalized
# module's non-host groups, joined to the capture's groups by region), and the build refuses a result
# whose dispatch set is not its own. Nothing here names a target.

#: Recorded open-model oracles, by :func:`cache_key`.
NAMESPACE = "whole-model-open-oracle"


def device_indices(capsule, original_groups: Sequence[Any], *, target: str) -> list[int]:
    """The capture-group index of every device dispatch the build's cut makes, in the cut's order."""
    from merlin.analysis.whole_model.build import open_program as WO
    from merlin.compiler.partition.groups import (
        formation_compute_groups as _merlin_formation_compute_groups_2,
    )

    # The capture's own device groups, as the statement states them (the build checks that the two sets
    # agree before it uses the result).
    by_region = {WO._region_of(g): int(g.index) for g in original_groups if g.placement != CG.HOST}
    from merlin.compiler.ir import mlir_query as mq

    module = mq.parse(Path(capsule.interface)).clone()  # a private copy of the shared parse, to normalize
    WO.normalize(module)
    indices = []
    for group in _merlin_formation_compute_groups_2.form_groups(module, target):
        if group.placement == CG.HOST:
            continue
        index = by_region.get(WO._region_of(group))
        if index is None:
            raise WO.OpenModelError(f"device group {group.index} ({WO._region_of(group)}) is no group of the capture")
        indices.append(index)
    return indices


def cache_key(capsule, *, target: str, extra: str | Path | None) -> str:
    """The closed model's oracle key (model, code, driver, support, contract, facts) plus the leaves file
    an open build may be handed from outside the capsule, under its own label."""
    import hashlib

    from merlin.analysis.whole_model.build.oracle import oracle_cache_key

    digest = hashlib.sha256(b"open-model-oracle\0")
    digest.update(oracle_cache_key(capsule, target=target).encode())
    digest.update(b"\0")
    digest.update(Path(extra).read_bytes() if extra is not None else b"<capsule-declared>")
    return digest.hexdigest()


def compute(capsule, *, target: str, extra: str | Path | None = None) -> tuple[dict[str, Any], list[int]]:
    """``(oracle, device indices)`` for ``capsule``, in this process."""
    from types import SimpleNamespace

    from merlin.analysis.whole_model.build import open_program as WO
    from merlin.compiler.ir import mlir_query as mq
    from merlin.compiler.partition.groups import (
        formation_compute_groups as _merlin_formation_compute_groups_3,
    )
    from merlin.runtime.backends import registry as backends

    arguments, _sources = WO.forward_arguments(capsule, extra=extra)
    # By path: one parse of the capture serves its groups here and, cloned, the normalized cut.
    original = mq.parse(Path(capsule.interface))
    groups = _merlin_formation_compute_groups_3.form_groups(original, target)
    indices = device_indices(capsule, groups, target=target)
    result = WO.oracle(
        original,
        arguments,
        [SimpleNamespace(group=i) for i in indices],
        groups,
        group_digest=backends.whole_model_driver(target).program.group_digest,
    )
    return result, indices


def _write(directory: Path, result: dict[str, Any], indices: list[int]) -> None:
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "outputs.npz", *[np.asarray(o) for o in result["outputs"]])
    rows = [[int(k), v] for k, v in result["dispatch"].items()]
    (directory / "dispatch.json").write_text(json.dumps({"indices": indices, "rows": rows}), encoding="utf-8")
    # Written last: a directory holding done.json is a complete record.
    (directory / "done.json").write_text(json.dumps({"outputs": len(result["outputs"])}), encoding="utf-8")


def _read(directory: Path) -> tuple[dict[str, Any], list[int]] | None:
    import numpy as np

    if not (directory / "done.json").is_file():
        return None
    count = int(json.loads((directory / "done.json").read_text(encoding="utf-8"))["outputs"])
    with np.load(directory / "outputs.npz") as stored:
        outputs = [np.asarray(stored[f"arr_{i}"]) for i in range(count)]
    recorded = json.loads((directory / "dispatch.json").read_text(encoding="utf-8"))
    return {"outputs": outputs, "dispatch": {int(k): v for k, v in recorded["rows"]}}, list(recorded["indices"])


def _main(argv: list[str]) -> int:
    (
'``python -m merlin.analysis.whole_model.build.open_execution <capsule '
'dir> <interface> <target> <extra|-> <out>``'
)
    from merlin.analysis.whole_model.build import stages as WMB

    capsule_dir, interface, target, extra, out = argv
    capsule = dataclasses.replace(WMB.load_model_capsule(capsule_dir), interface=Path(interface))
    result, indices = compute(capsule, target=target, extra=None if extra == "-" else extra)
    _write(Path(out), result, indices)
    return 0


class OpenOracleJob:
    """The open oracle, started with the build in a child process, or read from its cache."""

    def __init__(self, capsule, *, target: str, extra: str | Path | None = None, cache: bool = True):
        from merlin.common.artifacts import cache_dir

        self.key = cache_key(capsule, target=target, extra=extra)
        self.entry = Path(cache_dir(NAMESPACE)) / self.key[:2] / self.key if cache else None
        self.state = "hit" if self.entry is not None and _read(self.entry) is not None else "miss"
        self._process = None
        if self.state == "miss":
            parent = self.entry.parent if self.entry is not None else Path(tempfile.gettempdir())
            parent.mkdir(parents=True, exist_ok=True)
            self._staged = Path(tempfile.mkdtemp(prefix=".oracle.", dir=parent))
            self._log = self._staged.with_suffix(".log")
            argv = [str(capsule.directory), str(capsule.interface), target, str(extra) if extra else "-"]
            with self._log.open("w", encoding="utf-8") as log:
                self._process = subprocess.Popen(
                    [sys.executable, "-m", __name__, *argv, str(self._staged)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=dict(os.environ),
                )

    def result(self, dispatch_groups: Sequence[int]) -> dict[str, Any]:
        """The oracle, refused unless it was computed for exactly ``dispatch_groups`` (the cut's own)."""
        from merlin.analysis.whole_model.build import open_program as WO


        if self._process is None:
            recorded = _read(self.entry)
        else:
            if self._process.wait() != 0:
                tail = self._log.read_text(encoding="utf-8", errors="replace")[-1500:]
                raise WO.OpenModelError(f"the oracle could not be recomputed: {tail}")
            recorded = _read(self._staged)
            if self.entry is not None:
                try:
                    self._staged.rename(self.entry)
                except OSError:  # another build recorded the same oracle first; it is the same value
                    shutil.rmtree(self._staged, ignore_errors=True)
            else:
                shutil.rmtree(self._staged, ignore_errors=True)
            self._log.unlink(missing_ok=True)
        if recorded is None:
            raise WO.OpenModelError("the oracle job wrote no complete record")
        result, indices = recorded
        if sorted(indices) != sorted(int(g) for g in dispatch_groups):
            raise WO.OpenModelError("the oracle was computed for another set of device dispatches than the cut's")
        return result

    def cancel(self) -> None:
        """Stop a child that is still computing (the build failed before it needed the oracle)."""
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        if self._process is not None:
            shutil.rmtree(self._staged, ignore_errors=True)
            self._log.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))


# ---- whole_model_open_service ------------------------------------------------------------------
# Running and serving an open model: the functional run, the service builder, and the CLI.
#
# The half of :mod:`merlin.analysis.whole_model.build.open_program` driven by the whole-model service's
# builder (``whole_model_builder.build`` routes an open model here) and ``merlin-whole-model-open`` --
# kept beside the build it calls. Re-exported by :mod:`merlin.analysis.whole_model.build.open_program`.

# ------------------------------------------------------------------------------------------ run


def run_functional(
    record: Mapping[str, Any],
    *,
    command: Sequence[str],
    environment: Mapping[str, str] | None = None,
    out: str | Path,
    timeout: int = 6 * 3600,
) -> dict[str, Any]:
    """Run a build's ELF on a functional simulator (``command`` is the simulator and its switches, as a
    machine spec states them) over the build's own memory map; keep the console; grade it.

    The simulator is told the DRAM span the image was laid out for, from the record -- never a default,
    because an image whose arena lies past the simulated memory faults on its first allocation."""
    import os
    import subprocess
    import time

    from merlin.analysis.whole_model.build import open_program as WO

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    from merlin.runtime.backends.baremetal.image_link import declared_isa

    layout = record["layout"]
    span = f"-m{layout.get('dram_base', hex(0x80000000))}:{layout['mem_bytes']}"
    # A two-hart program states its harts, and its host code's ISA, in the image.
    harts = (record.get("program") or {}).get("harts") or {}
    isa = declared_isa(record["elf"]) if harts else None
    machine = [*([f"-p{harts['count']}"] if harts else []), *([f"--isa={isa}"] if isa else [])]
    argv = [*command, *machine, span, record["elf"]]
    env = dict(os.environ)
    env.update(environment or {})
    started = time.time()
    # Streamed to the file as it runs: an open model runs for hours on a functional simulator, and a
    # console held in memory until exit makes a slow run and a hung one look the same.
    with (out / "console.txt").open("wb") as sink:
        process = subprocess.Popen([str(a) for a in argv], stdout=sink, stderr=subprocess.STDOUT, env=env)
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            returncode = process.wait()
    console = (out / "console.txt").read_text(encoding="utf-8", errors="replace")
    expectations = json.loads(Path(record["expectations"]).read_text(encoding="utf-8"))
    verdict = WO.grade(console, expectations)
    result = {
        "argv": [str(a) for a in argv],
        "returncode": returncode,
        "wall_s": round(time.time() - started, 1),
        "elf_sha256": record["elf_sha256"],
        "verdict": verdict,
        "console": str(out / "console.txt"),
    }
    (out / "functional_run.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


# ------------------------------------------------------------------------ the service's builder


def is_open_model(model_capsule: str | Path, target: str) -> bool:
    """Whether the capsule's model computes on the host BETWEEN its groups (see ``model_closure``)."""
    from merlin.analysis.whole_model.build import stages as WMB
    from merlin.compiler.ir import mlir_query as mq
    from merlin.compiler.ir.xdsl_io import IR_LOCK
    from merlin.compiler.partition.groups import (
        formation_compute_groups as _merlin_formation_compute_groups_4,
    )
    from merlin.compiler.partition.groups import model_closure as MC

    # Without datapath facts every epilogue lands on the host and a closed model reads as open.
    WMB.require_datapath_facts(target)
    capsule = WMB.load_model_capsule(model_capsule)
    with IR_LOCK:
        groups = _merlin_formation_compute_groups_4.form_groups(
            mq.parse(capsule.interface.read_text(encoding="utf-8")), target
        )
    return bool(MC.open_host_regions(groups))


#: Build intermediates the linked ELF already contains (the open build's own layout).
PRUNABLE = ("lower/prefetch", "lower/*.generated", "program/lower_main", "program/lower_host", "bundle")


def service_build(
    package_dir: str | Path | None,
    *,
    target: str,
    out_dir: str | Path,
    model_capsule: str,
    machine: str,
    header: str,
    header_sha256: str | None = None,
    verify: str = "on_target",
    jobs: int | None = None,
    timeout: int = 600,
    prohibited_roles: Sequence[str] = (),
    protocol_keys: Sequence[str] = (),
    host_hart: int | None | str = "vector",
    phase0_recipe: str | Path | None = None,
    descriptor: str | Path | None = None,
    chunk_ops: int | str | None = None,
) -> dict[str, Any]:
    """:func:`build`, returned in the shape the measurement service and the whole-model gate read.

    ``host_hart="vector"`` (the default) runs the host code on the machine's vector hart when its
    registry entry declares one (:func:`merlin.analysis.whole_model.build.open_program.vector_host_hart`), a two-hart
    program; a machine that declares no harts keeps the one-hart program. ``None`` forces one hart.

    ``verify='local'`` is the locally graded build; any other mode is the timing build (no checks inside
    the program, the same dispatch digests after the window). Every device group is an EXACT group,
    graded locally by the program's own projection check; its oracle digests travel with it. The end
    result is the output tensor under the capsule's numeric policy (``expectations.output``), not a
    class, so ``expectations.argmax`` is ``None``.

    ``chunk_ops`` is passed straight through to :func:`whole_model_open.build` (``None`` keeps the
    unchunked program; ``"auto"`` derives the size from the forward). The build's per-stage seconds,
    kernel-object dedup and forward chunks ride along under the keys the whole-model gate copies into
    its build check.
    """
    from merlin.analysis.whole_model.build import open_driver as _merlin_open_driver_1
    from merlin.analysis.whole_model.build import open_program as WO
    from merlin.runtime.backends import registry as backends

    record = _merlin_open_driver_1.build(
        package_dir,
        model_capsule,
        target=target,
        machine=machine,
        header=header,
        header_sha256=header_sha256,
        out=out_dir,
        verify="local" if verify == "local" else "none",
        timeout=timeout,
        jobs=jobs,
        prohibited_roles=prohibited_roles,
        host_hart=WO.vector_host_hart(machine) if host_hart == "vector" else host_hart,
        phase0_recipe=phase0_recipe,
        descriptor=descriptor,
        chunk_ops=chunk_ops,
    )
    driver = backends.whole_model_driver(target)
    expectations = json.loads(Path(record["expectations"]).read_text(encoding="utf-8"))
    groups = {
        str(g): {
            "compare": "exact",
            "sum": expectations["dispatch"][str(g)]["sum"],
            "fnv1a": expectations["dispatch"][str(g)]["fnv1a"],
            # A dispatch reads the HOST's output, never another dispatch's directly.
            "inputs_from": [],
            "output_element_bytes": 4,
        }
        for g in expectations["groups"]
    }
    return {
        "elf": record["elf"],
        "elf_sha256": record["elf_sha256"],
        "parameter_header_sha256": record["program"]["abi_header"]["sha256"],
        "stage_times": record.get("stage_seconds"),
        "object_dedup": record.get("object_dedup"),
        "chunks": record.get("forward_chunks"),
        "expectations": {
            "groups": groups,
            "group_count": len(groups),
            "argmax": None,
            "output": {"elements": len(expectations["oracle_output"]), "policy": expectations["numeric_policy"]},
            "source": record["expectations"],
        },
        "groups": [
            {k: row.get(k) for k in ("group", "op", "on", "cause", "why") if k in row}
            | {"lowering": None, "call": "library" if row.get("library_call") else None}
            for row in record["attribution"]["per_group"]
        ],
        "protocol": {key: driver.program.UART[key] for key in (*protocol_keys, "output") if key in driver.program.UART},
        "prunable": list(PRUNABLE),
        "notes": {
            "builder": "merlin.analysis.whole_model.build.open_program",
            "verify": record.get("verify"),
            "attribution_counts": record["attribution"]["counts"],
            "abi_header": record["program"]["abi_header"],
            "harness_overrides": {},
            "build_record": str(Path(out_dir) / "whole_model_open_build.json"),
        },
        "provenance": record.get("provenance"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import sys

    from merlin.analysis.whole_model.build import open_driver as _merlin_open_driver_1
    from merlin.analysis.whole_model.build import open_program as WO

    parser = argparse.ArgumentParser(
        prog="merlin-whole-model-open",
        description="Build an open model (host regions between its groups) as one program, or grade a run of one.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("build")
    make.add_argument("--package", type=Path)
    make.add_argument("--capsule", required=True, type=Path)
    make.add_argument("--target", required=True)
    make.add_argument("--machine", required=True)
    make.add_argument("--header", required=True, type=Path)
    make.add_argument("--header-sha256")
    make.add_argument("--extra", type=Path, help="an .npz holding the leaf arguments the capsule does not store")
    make.add_argument("--out", required=True, type=Path)
    make.add_argument("--verify", choices=("local", "none"), default="local")
    make.add_argument("--prohibit-role", action="append", default=[])
    make.add_argument("--decline", action="append", default=[])
    make.add_argument("--dram-bytes", type=lambda v: int(v, 0))
    make.add_argument("--jobs", type=int)
    make.add_argument("--host-hart", type=int, help="a two-hart program: the host code on this (vector) hart")
    make.add_argument(
        "--chunk-ops",
        help="cut the host forward into functions of this many ops, or 'auto' to derive the size from the "
        "forward (a large forward otherwise compiles as one function for hours); omitted, it stays one function",
    )
    check = sub.add_parser("grade")
    check.add_argument("--uart", required=True, type=Path)
    check.add_argument("--expectations", required=True, type=Path)
    check.add_argument("--reference", type=Path, help="the reference arm's cache entry (whole_model_reference)")
    bench = sub.add_parser("bench", help="build the package's kernels at the model's regimes, without host code")
    for flag in ("--package", "--capsule", "--header", "--out"):
        bench.add_argument(flag, required=True, type=Path)
    bench.add_argument("--target", required=True)
    bench.add_argument("--machine", required=True)
    bench.add_argument("--header-sha256")
    bench.add_argument("--prohibit-role", action="append", default=[])
    bench.add_argument("--evict-bytes", type=lambda v: int(v, 0), default=0)
    bench.add_argument("--jobs", type=int)
    graded = sub.add_parser("grade-bench")
    graded.add_argument("--uart", required=True, type=Path)
    graded.add_argument("--record", required=True, type=Path)
    graded.add_argument("--reference-uart", type=Path)
    args = parser.parse_args(argv)
    if args.command == "grade-bench":
        verdict = WO.grade_kernel_bench(
            args.uart.read_text(encoding="utf-8", errors="replace"),
            json.loads(args.record.read_text()),
            reference_uart=args.reference_uart.read_text(encoding="utf-8", errors="replace")
            if args.reference_uart
            else None,
        )
        print(json.dumps(verdict, indent=1))
        return 0 if verdict["correct"] else 1
    if args.command == "bench":
        try:
            record = WO.build_kernel_bench(
                args.package,
                args.capsule,
                target=args.target,
                machine=args.machine,
                header=args.header,
                header_sha256=args.header_sha256,
                out=args.out,
                prohibited_roles=args.prohibit_role,
                evict_bytes=args.evict_bytes,
                jobs=args.jobs,
            )
        except WO.OpenModelError as refusal:
            print(f"not built: {refusal}", file=sys.stderr)
            return 2
        print(f"built {record['elf']} ({record['elf_sha256'][:12]}) census {record['census']['counts']}")
        return 0
    if args.command == "grade":
        verdict = WO.grade(
            args.uart.read_text(encoding="utf-8", errors="replace"),
            json.loads(args.expectations.read_text()),
            reference=json.loads(args.reference.read_text()) if args.reference else None,
        )
        print(json.dumps(verdict, indent=1))
        return 0 if verdict["quotable"] else 1
    try:
        record = _merlin_open_driver_1.build(
            args.package,
            args.capsule,
            target=args.target,
            machine=args.machine,
            header=args.header,
            header_sha256=args.header_sha256,
            extra=args.extra,
            out=args.out,
            verify=args.verify,
            prohibited_roles=args.prohibit_role,
            decline=args.decline,
            dram_bytes=args.dram_bytes,
            jobs=args.jobs,
            host_hart=args.host_hart,
            chunk_ops=args.chunk_ops,
        )
    except WO.OpenModelError as refusal:
        print(f"not built: {refusal}", file=sys.stderr)
        return 2
    print(f"built {record['elf']} ({record['elf_sha256'][:12]}) groups {record['attribution']['counts']}")
    return 0


# LAST, so either module can be imported first: each finds the other's names already defined.
