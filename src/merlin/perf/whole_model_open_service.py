"""Running and serving an open model: the functional run, the service builder, and the CLI.

The half of :mod:`merlin.perf.whole_model_open` a caller drives it through -- the whole-model service's
builder (``whole_model_builder.build`` routes an open model here) and ``merlin-whole-model-open`` --
kept beside the build it calls. Re-exported by :mod:`merlin.perf.whole_model_open`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    from merlin.runtime.backends.spike_model import declared_isa

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
    from merlin.common import mlir_query as mq
    from merlin.common.ir_lock import IR_LOCK
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import model_closure as MC

    from . import whole_model_build as WMB

    # Without datapath facts every epilogue lands on the host and a closed model reads as open.
    WMB.require_datapath_facts(target)
    capsule = WMB.load_model_capsule(model_capsule)
    with IR_LOCK:
        groups = CG.form_groups(mq.parse(capsule.interface.read_text(encoding="utf-8")), target)
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
    registry entry declares one (:func:`merlin.perf.whole_model_open.vector_host_hart`), a two-hart
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
    from merlin.runtime.backends import base as backends

    record = WO.build(
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
            "builder": "merlin.perf.whole_model_open",
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
        record = WO.build(
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
from . import whole_model_open as WO  # noqa: E402
