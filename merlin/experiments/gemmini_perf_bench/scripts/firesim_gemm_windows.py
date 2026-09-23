#!/usr/bin/env python3
"""Measure GEMM shapes on the FPGA as N SEALABLE WINDOWS IN ONE QUEUE JOB, and admit each on its digest.

WHY THIS EXISTS.  ``layer_shape_table.py`` measures the same shapes on GSIM, both arms, every row
digest-checked.  GSIM is a simulator; a claim about the hardware needs the hardware.  But the
programs that harness builds print ``LB_RECORD``/``LB_DONE`` and nothing else, and
``merlin.perf.firesim_receipt`` seals a run only when its UART carries ``MERLIN_INVOCATIONS``, the
four ``MERLIN_PROFILE`` lines and one ``METRIC cycles N``.  Submitting one of those ELFs would burn
a queue slot and produce a log no receipt could read.  So this script renders the same measurement
inside the frame the sealer accepts (``gemmini_layer_bench.render_window_program``).

WHY ONE ELF WITH N WINDOWS AND NOT N JOBS.  ``merlin.perf.execution_policy`` derives the queue
lifecycle from the daemon's own trace: exactly ``kill -> infrasetup -> runworkload -> kill``, one
``runworkload`` per job.  Holding one session and replaying N ELFs is inadmissible by construction --
no receipt can be sealed from it.  One bootbinary, N windows, one ``runworkload-full``.

ADMISSION.  A cycle count is not a result without its correctness receipt beside it.  Every window
prints ``LB_RECORD <label> cycles=<n> digest=<d>``; ``d`` is compared, off-device and in exact
integer arithmetic (``merlin.perf.layer_bench.reference.expected_digest``), against what the
contract says the output must be.  A window that clears its cycle count and misses its digest is a
FAILURE, not an incomplete -- FireSim job 730 published a number under both declared thresholds,
with an argmax marker byte-identical to a correct run's, and was wrong.

ORDER-EFFECT CONTROL.  A batch's last window repeats window 0.  Without it nothing distinguishes
"this shape is faster" from "this shape ran third, with the array and DRAM in a state the first one
did not see", and a batched number is not comparable to a solo one.  A divergence beyond the
declared ppm bound invalidates every window in the batch, including the ones that looked fine.

    # one window, end to end
    .venv/bin/python merlin/experiments/gemmini_perf_bench/scripts/firesim_gemm_windows.py run \\
        --window 512x512x512:schedule --batch-id solo1 --project gemm-window-proof

    # the full head-to-head, both arms, with the order-effect control
    ... run --window 512x512x512:schedule --window 512x512x512:library ... --batch-id gemm12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

#: Fixed queue/workload policy for this host.  Not derived: the queue is a host fact, and a wrong
#: guess here submits to the wrong daemon or stages under the wrong workload.
QUEUE_EXECUTABLE = Path("/opt/firesim-queue/bin/firesim-queue")
QUEUE_STATE_ROOT = Path("/scratch/firesim_queue")
#: The environment key naming the chipyard checkout whose FireSim deploy holds THIS design. It is a
#: per-machine location, so it is read from the environment / the gitignored ``.env`` rather than
#: written here: a checkout path baked into the script is one person's machine, and the next one's
#: run would submit against a tree that is not theirs.
CHIPYARD_ENV = "MERLIN_CHIPYARD_GEMMINI_ROCKET"
WORKLOAD = "merlin-checkpoint"
BOOTBINARY = "merlin-checkpoint.elf"
HW_CONFIG = "alveo_u250_firesim_gemmini_rocket_30mhz"
#: The launcher that keeps the daemon's own cwd when it resolves the bare ``firesim`` command.
LAUNCHER_DIR = Path(__file__).resolve().with_name("firesim_queue_cwd_launcher")
#: Identity variables a cross-user daemon must NOT inherit.  Proven from the queue's own database:
#: jobs 722-724 DONE with HOME absent, 725 FAILED with it set, nothing else different.
IDENTITY_VARIABLES = ("HOME", "USER", "LOGNAME")

#: What a window may measure. ``schedule`` and ``library`` are this repo's own two renderers. The
#: three published arms are a third party's kernels, compiled as they are written and linked beside the
#: program (see ``render_autocomp_kernel_unit``); ``package`` is an agent-generated compiler invoked
#: through its manifest argv. Every arm in one batch computes ONE function and is admitted against ONE
#: digest -- an arm that cannot be made numerically identical to the others belongs in a different run,
#: not in a column beside them.
ARMS = (
    "schedule",
    "library",
    "exo_opt",
    "exo_baseline",
    "gemmini_baseline",
    "autocomp_generated",
    "package",
)
#: Arms whose kernel is a third party's published C file, compiled as it is written and linked
#: beside the program.  The first three are files the AutoComp artifact ships under ``sols/exo/``.
#: ``autocomp_generated`` is AutoComp's OWN generated kernel, which the paper prints in full
#: (arXiv:2505.18574, Figs. 30-32) but the artifact does not ship in machine-readable form; it is
#: read from ``--generated-kernel`` and rendered through exactly the same unit renderer, operand
#: blob and oracle digest as the three that ARE shipped, so the transcription is adjudicated by
#: the shared digest rather than by reading it.
PUBLISHED_ARMS = ("exo_opt", "exo_baseline", "gemmini_baseline", "autocomp_generated")
#: The one published arm whose kernel text does not come from the artifact tree.
GENERATED_ARM = "autocomp_generated"
#: One label tag per arm, so a batch's window labels stay short, unique and readable in a UART log.
ARM_TAG = {
    "schedule": "s",
    "library": "l",
    "exo_opt": "eo",
    "exo_baseline": "eb",
    "gemmini_baseline": "gb",
    "autocomp_generated": "ag",
    "package": "pk",
}


def chipyard_root() -> Path:
    """The chipyard checkout this design's FireSim deploy lives in, from the environment or ``.env``.

    Resolved when a run needs it, never at import, so building a program and inspecting a plan work on
    a machine that has no FPGA host configured at all.
    """
    from merlin.common.paths import env

    value = env(CHIPYARD_ENV)
    if not value:
        raise SystemExit(f"{CHIPYARD_ENV} is unset; set it in the environment or <repo>/.env")
    root = Path(value)
    if not root.is_dir():
        raise SystemExit(f"{CHIPYARD_ENV}={root} is not a directory")
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_window(text: str) -> tuple[tuple[int, int, int], str]:
    """``"MxNxK:arm"`` -> ``((M, N, K), arm)``.  Structural split; a malformed member raises."""
    shape, separator, arm = text.partition(":")
    if not separator:
        raise SystemExit(f"--window {text!r} is not MxNxK:arm")
    if arm not in ARMS:
        raise SystemExit(f"--window {text!r} names arm {arm!r}, not one of {ARMS}")
    parts = shape.lower().split("x")
    if len(parts) != 3:
        raise SystemExit(f"--window {text!r} does not start with MxNxK")
    try:
        m, n, k = (int(part) for part in parts)
    except ValueError as exc:
        raise SystemExit(f"--window {text!r} has a non-integer extent") from exc
    if min(m, n, k) <= 0:
        raise SystemExit(f"--window {text!r} has a non-positive extent")
    return (m, n, k), arm


def window_label(shape: tuple[int, int, int], arm: str) -> str:
    return f"{ARM_TAG[arm]}{shape[0]}x{shape[1]}x{shape[2]}"


def harness_extents(harness_c: Path) -> dict[str, int]:
    """``MAT_DIM_I/K/J`` from a published harness file, read structurally (no pattern matching).

    A ``#define`` line is split on whitespace and the third token parsed as an int; a file missing any
    of the three, or spelling one non-numerically, raises rather than defaulting -- the shape a kernel
    is measured at is the one thing this script must not guess.
    """
    want = {"MAT_DIM_I": "m", "MAT_DIM_K": "k", "MAT_DIM_J": "n"}
    found: dict[str, int] = {}
    for line in harness_c.read_text(encoding="utf-8", errors="replace").splitlines():
        tokens = line.split()
        if len(tokens) < 3 or tokens[0] != "#define" or tokens[1] not in want or tokens[1] in found:
            continue
        try:
            found[want[tokens[1]]] = int(tokens[2])
        except ValueError as exc:
            raise SystemExit(f"{harness_c}: {tokens[1]} is not an integer extent ({tokens[2]!r})") from exc
    missing = sorted(set(want.values()) - set(found))
    if missing:
        raise SystemExit(f"{harness_c} declares no {missing} extent")
    return found


def published_tests(autocomp_root: Path) -> dict[tuple[int, int, int], int]:
    """``(M, N, K) -> published test index``, read from the artifact's own harnesses.

    The shapes are the artifact's, not this script's: a ``--window`` naming extents no published
    harness declares has no published kernel to measure and is refused rather than approximated.
    """
    harnesses = autocomp_root / "harnesses" / "exo"
    index: dict[tuple[int, int, int], int] = {}
    for harness in sorted(harnesses.glob("test*.c")):
        stem = harness.stem[len("test") :]
        if not stem.isdigit():
            continue
        extents = harness_extents(harness)
        index[(extents["m"], extents["n"], extents["k"])] = int(stem)
    if not index:
        raise SystemExit(f"no published harnesses under {harnesses}")
    return index


def solution_signature_shape(text: str) -> dict[str, int]:
    """``(m, n, k)`` as a published kernel's OWN signature declares them, read structurally.

    A kernel whose tile counts and scratchpad bases are literals is that kernel at exactly one shape,
    and the only statement the file itself makes about which shape that is, is its parameter list:
    ``void solution(elem_t A[M][K], elem_t B[K][N], elem_t C[M][N])``.  Read it rather than assume it,
    so measuring the kernel at other extents is refused by the file instead of by a constant written
    beside it.  Split on the punctuation the declaration is made of; a parameter list that is not
    three 2-D operands agreeing on one shape raises rather than being approximated.
    """
    _, opened, rest = text.partition("void solution(")
    if not opened:
        raise SystemExit("the published kernel text does not define solution()")
    params, closed, _ = rest.partition(")")
    if not closed:
        raise SystemExit("the published kernel's solution() parameter list is unterminated")
    extents: list[list[int]] = []
    for param in params.split(","):
        dims: list[int] = []
        remainder = param
        while True:
            _, bracket, remainder = remainder.partition("[")
            if not bracket:
                break
            inner, shut, remainder = remainder.partition("]")
            if not shut:
                raise SystemExit(f"unterminated array extent in solution() parameter {param.strip()!r}")
            token = inner.strip()
            if not token.isdigit():
                raise SystemExit(f"solution() parameter {param.strip()!r} declares a non-literal extent {token!r}")
            dims.append(int(token))
        extents.append(dims)
    if len(extents) != 3 or any(len(dims) != 2 for dims in extents):
        raise SystemExit(f"solution() does not declare three 2-D operands; read {extents}")
    (m_a, k_a), (k_b, n_b), (m_c, n_c) = extents
    if k_a != k_b or m_a != m_c or n_b != n_c:
        raise SystemExit(f"solution()'s three operands disagree on one shape; read {extents}")
    return {"m": m_a, "n": n_b, "k": k_a}


def published_kernel(
    autocomp_root: Path,
    *,
    test: int,
    arm: str,
    shape: tuple[int, int, int],
    generated_kernel: Path | None = None,
) -> tuple[str, str]:
    """``(the kernel's text, how it was obtained)`` for one published arm at one shape.

    The text is the published file's, unedited. The one case that is not a verbatim file is the
    library arm at the shapes the artifact ships no file for: it publishes
    ``sol{0,1,2}_gemmini_baseline.c`` and no others, those three are identical below their signature
    line, and their body is written ENTIRELY in the harness's macros (``MAT_DIM_I/J/K``,
    ``A_MATRIX_NAME``, ...). So for the remaining shapes this returns that same body under the
    signature the shape requires and says so; nothing inside the function is touched.
    """
    if arm == GENERATED_ARM:
        if generated_kernel is None:
            raise SystemExit(f"arm {arm!r} needs --generated-kernel")
        if not generated_kernel.is_file():
            raise SystemExit(f"no transcribed AutoComp kernel at {generated_kernel}")
        text = generated_kernel.read_text(encoding="utf-8")
        if "void solution(" not in text:
            raise SystemExit(f"{generated_kernel} does not define solution()")
        return text, f"transcribed from the paper's printed listing (arXiv:2505.18574 Figs. 30-32): {generated_kernel}"
    sols = autocomp_root / "sols" / "exo"
    own = sols / f"sol{test}_{arm}.c"
    if own.is_file():
        text = own.read_text(encoding="utf-8")
        if "void solution(" not in text:
            raise SystemExit(f"{own} does not define solution()")
        return text, f"published verbatim: {own.name}"
    if arm != "gemmini_baseline":
        raise SystemExit(f"no published kernel {own}")
    donors = sorted(sols.glob("sol*_gemmini_baseline.c"))
    if not donors:
        raise SystemExit(f"no sol*_gemmini_baseline.c under {sols}")
    donor = donors[0]
    text = donor.read_text(encoding="utf-8")
    head, _, rest = text.partition("void solution(")
    _, _, tail = rest.partition(")")
    m, n, k = shape
    signature = f"void solution(int8_t A[{m}][{k}], int8_t B[{k}][{n}], int8_t C[{m}][{n}])"
    return (
        head + signature + tail,
        f"body verbatim from {donor.name} under a re-signed prototype ({signature}); the artifact "
        f"publishes no sol{test}_gemmini_baseline.c, and that body is written only in harness macros",
    )


def objcopy_for(target: str) -> Path:
    """The ``objcopy`` of the SAME toolchain this target's harness is compiled and linked with.

    Derived from the recipe's own compiler path rather than named here, so a host that moves its
    toolchain moves this with it, and the tool that rewrites an object is provably the one whose
    linker will read it back. A toolchain whose compiler is not a ``<prefix>gcc`` (or that ships no
    matching ``objcopy``) fails here rather than falling back to whatever is on PATH.
    """
    from merlin.runtime.backends import base

    compiler = Path(base.harness_build_recipe(target).compiler)
    name = compiler.name
    suffix = "gcc"
    if not name.endswith(suffix):
        raise SystemExit(f"cannot derive an objcopy from the harness compiler {compiler}")
    candidate = compiler.with_name(name[: -len(suffix)] + "objcopy")
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise SystemExit(f"the harness toolchain ships no objcopy beside {compiler} (looked for {candidate})")
    return candidate


def package_kernel(
    spec: dict[str, Any],
    *,
    package: Path,
    readout: str,
    target: str,
    symbol: str,
    workdir: Path,
    obj_cache: Path,
    timeout_s: float,
) -> dict[str, Any]:
    """Compile one GEMM with the PACKAGE and return what a window needs to call it.

    The package is invoked only through its manifest argv -- it is never imported -- exactly as
    ``targetgen.oot_runner.load_package`` describes the command protocol. This is
    ``layer_shape_table._package_kernel``'s body with one addition a BATCH needs and a single program
    does not: the lowered object defines the target's one harness entry symbol, so six shapes in one
    ELF would be six definitions of it. The object is therefore copied with that symbol RENAMED to a
    name of this window's own, which changes no instruction in it.
    """
    import yaml

    from merlin.llvmlower import toolchain
    from merlin.targetgen.contract.compile import llvm_mlir_to_object

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from layer_package_table import iface_module, pack_for_abi  # noqa: PLC0415

    from merlin.perf.layer_bench.reference import operand_arrays  # noqa: PLC0415
    from merlin.runtime.backends import base  # noqa: PLC0415
    from merlin.targetgen.contract.harness_abi import for_target  # noqa: PLC0415

    backend = base.get_backend(target)
    if int(spec.get("bias_span", 0)) != 0:
        raise SystemExit(
            "the interface GEMM handed to a package declares no bias operand, so with a non-zero bias "
            "span the package would compute a different function from the arms beside it and the "
            "digests would be incomparable by construction. Run the batch with --bias-span 0."
        )
    module, binding = iface_module(spec, float(spec["scale"]), target, readout=readout)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "layer.iface.mlir").write_text(module, encoding="utf-8")
    manifest = yaml.safe_load((package / "manifest.yaml").read_text(encoding="utf-8"))
    tool = package / manifest["entrypoints"]["tool"]
    argv = [
        token.replace("{tool}", str(tool))
        .replace("{input_mlir}", str(workdir / "layer.iface.mlir"))
        .replace("{output_json}", str(workdir / "cb.json"))
        for token in manifest["commands"]["emit_analysis_bundle"]["argv"]
    ]
    started = time.monotonic()
    compiled = subprocess.run(
        [sys.executable, *argv] if tool.suffix == ".py" or not os.access(tool, os.X_OK) else argv,
        capture_output=True,
        text=True,
        env={**os.environ, "MERLIN_PYTHON": sys.executable},
        timeout=timeout_s,
    )
    compile_seconds = time.monotonic() - started
    if compiled.returncode != 0 or not (workdir / "cb.json").is_file():
        raise SystemExit(f"package compile rc={compiled.returncode}: {compiled.stderr[-600:]}")
    cb = json.loads((workdir / "cb.json").read_text(encoding="utf-8"))
    if cb.get("declined") or not cb.get("commands"):
        raise SystemExit(f"package declined: {json.dumps(cb.get('declined'))[:400]}")
    lowered = compiled.stdout
    derive = getattr(backend, "kernel_abi_from_commands", None)
    if derive is None:
        raise SystemExit(f"backend for {target!r} cannot resolve a command buffer's implied kernel ABI")
    cb["kernel_abi"] = derive(cb)
    blob, offsets = pack_for_abi(cb, dict(operand_arrays(spec)), binding)
    # The object is a function of the lowered artifact, the target and the clang that built it, and is
    # by far the slowest step, so it is cached on exactly those -- independent of the harness around it.
    key = hashlib.sha256(
        lowered.encode() + b"\0" + target.encode() + b"\0" + str(toolchain.clang()).encode()
    ).hexdigest()
    cached = obj_cache / f"{key}.o"
    started = time.monotonic()
    if not cached.is_file():
        built = llvm_mlir_to_object(lowered, workdir / "obj", target=target)
        obj_cache.mkdir(parents=True, exist_ok=True)
        staging = obj_cache / f".{key}.{os.getpid()}.tmp"
        staging.write_bytes(built.read_bytes())
        os.replace(staging, cached)
    lower_seconds = time.monotonic() - started
    shutil.rmtree(workdir / "obj", ignore_errors=True)
    entry = for_target(target).entry_symbol
    renamed = workdir / f"{symbol}.o"
    rename = subprocess.run(
        [str(objcopy_for(target)), f"--redefine-sym={entry}={symbol}", str(cached), str(renamed)],
        capture_output=True,
        text=True,
    )
    if rename.returncode != 0 or not renamed.is_file():
        raise SystemExit(f"could not rename {entry!r} to {symbol!r} in the package object: {rename.stderr[-400:]}")
    return {
        "cb": cb,
        "blob": blob,
        "offsets": offsets,
        "object": renamed,
        "entry_symbol": entry,
        "symbol": symbol,
        "output_tensor": cb["kernel_abi"]["outputs"][0],
        "lowered_sha256": hashlib.sha256(lowered.encode()).hexdigest(),
        "commands": [command.get("opcode") for command in cb["commands"]],
        "abi_args": [argument["tensor"] for argument in cb["kernel_abi"]["args"]],
        "compile_seconds": round(compile_seconds, 2),
        "lower_seconds": round(lower_seconds, 2),
    }


def build_windows(
    members: list[tuple[tuple[int, int, int], str]],
    *,
    target: str,
    scale: float,
    relu: bool,
    workdir: Path,
    order_control: bool,
    elem_mode: str = "i8_full",
    bias_span: int | None = None,
    autocomp_root: Path | None = None,
    generated_kernel: Path | None = None,
    package: Path | None = None,
    package_readout: str = "requant_i8",
    package_timeout_s: float = 10_800,
) -> dict[str, Any]:
    """Pack ONE shared operand blob, render the framed program, and state each window's oracle digest.

    One blob for every window is what makes the members comparable AND what
    :func:`merlin.perf.firesim_batch.link_batch` requires: members that disagree on their operand
    bytes are two questions, not two candidates for one.  A shape's operands depend only on its
    extents and the seed, so the two arms of a shape share bytes rather than duplicating them.
    """
    from merlin.common.paths import artifacts_dir
    from merlin.perf.layer_bench.reference import OPERAND_ALIGN, expected_digest, pack_operands
    from merlin.runtime.backends import base
    from merlin.sched.check.static import check_kernel
    from merlin.sched.codegen import emit_c_function
    from merlin.sched.contract import contract
    from merlin.sched.ir import TensorArg

    backend = base.get_backend(target)
    contract_obj = contract("per_tensor_readout_v1", backend.readout_facts())
    obj_cache = artifacts_dir() / "perf-bench" / target / "layer_obj_cache"
    iset = backend.sched_instruction_set()
    elem_dtype, acc_dtype = iset.facts["elem_dtype"], iset.facts["acc_dtype"]

    blob = bytearray()
    shape_offsets: dict[tuple[int, int, int], dict[str, int]] = {}
    specs: dict[tuple[int, int, int], dict[str, Any]] = {}
    for shape, _arm in members:
        if shape in shape_offsets:
            continue
        m, n, k = shape
        spec = {
            "op": "matmul",
            "m": m,
            "n": n,
            "k": k,
            "relu": bool(relu),
            "scale": float(scale),
            "seed": 1,
            "elem_mode": elem_mode,
        }
        if bias_span is not None:
            spec["bias_span"] = int(bias_span)
        payload, offsets = pack_operands(spec, accumulator_dtype=contract_obj.accumulator_dtype)
        blob += b"\0" * ((-len(blob)) % OPERAND_ALIGN)
        base_offset = len(blob)
        blob += payload
        shape_offsets[shape] = {name: base_offset + offset for name, offset in offsets.items()}
        specs[shape] = spec

    ordered = list(members) + ([members[0]] if order_control and members else [])
    windows: list[dict[str, Any]] = []
    declared: list[dict[str, Any]] = []
    #: Extra translation units linked beside the window program -- one per published kernel, because
    #: every published file names its function ``solution`` and the library arm's is written in the
    #: harness's per-shape macros.
    units: list[tuple[Path, str]] = []
    #: Objects linked beside the program -- one per package-compiled shape, each with the target's
    #: harness entry symbol renamed to that window's own.
    objects: list[Path] = []
    sources: dict[str, dict[str, Any]] = {}
    tests = published_tests(autocomp_root) if autocomp_root is not None else {}
    unknown = sorted({shape for shape, arm in ordered if arm in PUBLISHED_ARMS and shape not in tests})
    if unknown:
        raise SystemExit(
            f"no published harness declares the extents {unknown}; there is no published kernel to measure"
        )
    for index, (shape, arm) in enumerate(ordered):
        label = window_label(shape, arm)
        if order_control and index == len(ordered) - 1:
            # The label link_batch mints for the repeat, so the plan and the program agree.
            label = f"{label}__order_control"
        spec = dict(specs[shape], label=label)
        offsets = shape_offsets[shape]
        entry: dict[str, Any] = {"label": label, "spec": spec, "offsets": offsets}
        if arm in PUBLISHED_ARMS:
            if autocomp_root is None:
                raise SystemExit(f"arm {arm!r} needs --autocomp-root")
            test = tests[shape]
            text, note = published_kernel(
                autocomp_root, test=test, arm=arm, shape=shape, generated_kernel=generated_kernel
            )
            symbol = f"lb_pub_{index}"
            unit = workdir / f"pub_{index}.c"
            # The transcribed kernel is the one file whose extents are its own literals rather than the
            # harness macros the artifact's files are written in, so its shape is read out of it and the
            # renderer's declared-shape check becomes a real one for this arm.
            declared_shape = solution_signature_shape(text) if arm == GENERATED_ARM else dict(zip("mnk", shape))
            unit_source = backend.render_autocomp_kernel_unit(
                text, symbol=symbol, m=shape[0], n=shape[1], k=shape[2], declared_shape=declared_shape
            )
            units.append((unit, unit_source))
            declaration, body, n_out, call = backend.autocomp_window_call(
                symbol=symbol, offsets=offsets, m=shape[0], n=shape[1], k=shape[2]
            )
            entry.update(
                kernel_c=declaration,
                external={"body": body, "n_out": n_out, "call": call},
            )
            sources[label] = {
                "arm": arm,
                "published_test": test,
                "note": note,
                "symbol": symbol,
                "kernel_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "kernel_bytes": len(text.encode()),
                "unit_sha256": hashlib.sha256(unit_source.encode()).hexdigest(),
            }
        elif arm == "package":
            if package is None:
                raise SystemExit("arm 'package' needs --package <dir holding manifest.yaml>")
            symbol = f"lb_pkg_{index}"
            built = package_kernel(
                spec,
                package=package,
                readout=package_readout,
                target=target,
                symbol=symbol,
                workdir=workdir / f"pkg_{index}",
                obj_cache=obj_cache,
                timeout_s=package_timeout_s,
            )
            # The package's operands are laid out for the ABI ITS command buffer declares, which is not
            # the library layout, so its bytes are their own region of the one shared blob. The values
            # are the same draws -- the digest below is the same oracle every other arm is admitted on.
            blob += b"\0" * ((-len(blob)) % OPERAND_ALIGN)
            package_base = len(blob)
            blob += built["blob"]
            declaration, body, n_out, call = backend.package_window_call(
                built["cb"],
                offsets={name: package_base + offset for name, offset in built["offsets"].items()},
                symbol=symbol,
                output=built["output_tensor"],
                element_dtype=contract_obj.output_dtype,
            )
            objects.append(built["object"])
            entry.update(kernel_c=declaration, external={"body": body, "n_out": n_out, "call": call})
            sources[label] = {
                "arm": arm,
                "note": "compiled by the package through its own manifest argv",
                "symbol": symbol,
                "package": str(package),
                "package_readout": package_readout,
                **{
                    key: built[key]
                    for key in (
                        "entry_symbol",
                        "lowered_sha256",
                        "commands",
                        "abi_args",
                        "output_tensor",
                        "compile_seconds",
                        "lower_seconds",
                    )
                },
            }
        elif arm == "schedule":
            m, n, k = shape
            shapes = {"a": (m, k), "b": (k, n), "d": (n,), "c": (m, n)}
            names = {"a": "a", "b": "b", "d": "d", "c": "output"}
            operands = {
                role: TensorArg(
                    names[role],
                    shapes[role],
                    acc_dtype if role == "d" else elem_dtype,
                    "write" if role == "c" else "read",
                )
                for role in ("a", "b", "d", "c")
            }
            symbol = f"lb_sched_{index}"
            kernel = backend.sched_matmul_reference(
                name=f"mm_{label}", m=m, n=n, k=k, operands=operands, relu=bool(relu), scale=float(scale)
            )
            errors = check_kernel(kernel, iset)
            if errors:
                raise SystemExit(f"window {label}: schedule does not check: {'; '.join(errors[:5])}")
            entry.update(
                kernel_c=emit_c_function(kernel, iset, symbol=symbol),
                symbol=symbol,
                arg_names=[argument.name for argument in kernel.args],
                kernel_digest=kernel.digest(),
            )
        windows.append(entry)
        declared.append(
            {
                "label": label,
                "arm": arm,
                "shape": f"{shape[0]}x{shape[1]}x{shape[2]}",
                "macs": shape[0] * shape[1] * shape[2],
                "digest_expected": expected_digest(spec, contract_obj),
                "is_order_control": bool(order_control and index == len(ordered) - 1),
                "repeats": windows[0]["label"] if (order_control and index == len(ordered) - 1) else None,
                "kernel_digest": entry.get("kernel_digest"),
                "source": sources.get(label),
            }
        )

    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / backend.LIBRARY_LAYER_OPERAND_BLOB).write_bytes(bytes(blob))
    for path, source in units:
        path.write_text(source, encoding="utf-8")
    return {
        "windows": windows,
        "declared": declared,
        "units": [str(path) for path, _ in units],
        "objects": [str(path) for path in objects],
        "blob_bytes": len(blob),
        "blob_sha256": hashlib.sha256(bytes(blob)).hexdigest(),
        "contract_digest": contract_obj.digest(),
        "target": target,
    }


def render_and_build(plan: dict[str, Any], *, target: str, workdir: Path, batch_id: str | None) -> dict[str, Any]:
    from merlin.perf.layer_bench import build_program
    from merlin.runtime.backends import base

    backend = base.get_backend(target)
    source = backend.render_window_program(plan["windows"], batch_id=batch_id)
    (workdir / "windows.c").write_text(source, encoding="utf-8")
    built = build_program(
        [
            workdir / "windows.c",
            *(Path(unit) for unit in plan.get("units", [])),
            *(Path(obj) for obj in plan.get("objects", [])),
        ],
        workdir,
        target=target,
        max_loaded_bytes=None,
        elf_name="gemm_windows.elf",
        support_first=True,
    )
    return {
        "elf": built.elf,
        "elf_sha256": built.elf_sha256,
        "loaded_bytes": built.loaded_bytes,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }


def validation_policy(plan: dict[str, Any], *, policy_id: str) -> dict[str, Any]:
    """The v2 policy: for every window, its exact marker AND the digest its output must have.

    A v1 policy names markers only, which is the shape job 730 walked through: its argmax marker
    printed identically to a correct run's.  ``merlin.perf.firesim_batch`` refuses v1 for a cycle
    claim BY NAME for exactly that reason, so this is the only policy shape that can support one.
    """
    from merlin.runtime.backends import base

    backend = base.get_backend(plan["target"])
    return {
        "schema": "merlin_firesim_uart_validation_policy_v2",
        "policy_id": policy_id,
        "workload": WORKLOAD,
        "checksum_line": dict(backend.WINDOW_CHECKSUM_LINE),
        "per_window": [
            {
                "label": row["label"],
                "markers": [backend.window_marker(row["label"])],
                "checksums": {row["label"]: int(row["digest_expected"])},
                # One CONTIGUOUS span: the window frames `t0 = read_cycles()`, the call, its fence,
                # then `t1`. Declared because the whole-model program reports a different quantity --
                # the SUM of its per-group spans, which excludes every inter-group host step -- and a
                # ratio between the two would divide unlike windows. An undeclared kind seals as
                # UNKNOWN and any ratio against it is refused, so silence here is not free.
                "window_kind": WINDOW_KIND_CONTIGUOUS,
                **({"repeats": row["repeats"]} if row["repeats"] else {}),
            }
            for row in plan["declared"]
        ],
    }


def client_environment() -> dict[str, str]:
    """The submitter's environment minus its identity, plus the cwd-preserving launcher.

    ``merlin.perf.firesim_checkpoint.client_environment`` owns both rules; this calls it rather
    than restating them, because the reason for the identity drop is a measured incident recorded
    in that module's docstring and must not come to live in two places.
    """
    from merlin.perf.firesim_checkpoint import client_environment as build_environment

    for launcher in ("firesim", "make"):
        path = LAUNCHER_DIR / launcher
        if path.is_symlink() or not path.is_file() or not os.access(path, os.X_OK):
            raise SystemExit(f"FireSim queue launcher is missing, symlinked or not executable: {path}")
    return build_environment(path_prefix=(LAUNCHER_DIR,), drop=IDENTITY_VARIABLES)


def device_provenance(hwdb_entry: Path) -> dict[str, Any]:
    """WHICH DEVICE.  The hwdb entry's bytes and the bitstream artifact it names, by NAME and digest.

    Two bitstreams on this host share the config string ``FireSimGemminiRocketConfig``, so that
    string cannot tell them apart; the build ARTIFACT's directory name can, and its tarball's digest
    confirms the name.  Anything not establishable is recorded as ``UNKNOWN(<reason>)`` rather than
    guessed: a result attributed to the wrong device is worse than no result.
    """
    import yaml

    document = yaml.safe_load(hwdb_entry.read_text(encoding="utf-8")) or {}
    entries = sorted(document)
    record: dict[str, Any] = {
        "hw_config": HW_CONFIG,
        "hwdb_entry_path": str(hwdb_entry),
        "hwdb_entry_sha256": _sha256(hwdb_entry),
        "hwdb_entry_keys": entries,
        "config_string_is_not_an_identity": (
            "two bitstreams on this host share the config string 'FireSimGemminiRocketConfig'; the "
            "identity below is the build ARTIFACT name plus the tarball digest, never that string"
        ),
    }
    entry = document.get(HW_CONFIG) if isinstance(document, dict) else None
    url = (entry or {}).get("bitstream_tar") if isinstance(entry, dict) else None
    if not isinstance(url, str) or not url.startswith("file://"):
        record["bitstream"] = f"UNKNOWN(hwdb entry {HW_CONFIG!r} declares no file:// bitstream_tar)"
        return record
    tarball = Path(url[len("file://") :])
    record["bitstream_tar"] = str(tarball)
    record["bitstream_artifact_name"] = tarball.parent.parent.name
    record["bitstream_build_name"] = tarball.parent.name
    if not tarball.is_file():
        record["bitstream_tar_sha256"] = f"UNKNOWN(bitstream tarball not readable at {tarball})"
    else:
        record["bitstream_tar_sha256"] = _sha256(tarball)
        record["bitstream_tar_bytes"] = tarball.stat().st_size
    return record


def run_on_gsim_first(elf: Path, *, target: str, max_cycles: int, timeout_s: float) -> dict[str, Any]:
    """Prove the PROGRAM before spending a queue slot on it: same ELF, cycle-accurate simulator.

    This is not the hardware result and is never reported as one.  It establishes that the window
    frame is emitted, that every digest matches the oracle, and that the program terminates -- the
    three ways a submission can waste a slot without telling you why.
    """
    from merlin.perf.layer_bench import run_on_gsim

    run = run_on_gsim(elf, target=target, max_cycles=max_cycles, timeout_s=timeout_s, backdoor=True)
    return {
        "completed": run.completed,
        "wall_seconds": round(run.wall_seconds, 2),
        "engine_cycles": run.finish.cycles if run.finish else None,
        "load_path": run.load_path,
        "records": [{"label": record.label, "cycles": record.cycles, **record.fields} for record in run.records],
        "stdout_tail": run.stdout_tail,
        "stderr_tail": run.stderr_tail,
    }


def submit_and_seal(
    *,
    elf: Path,
    policy_path: Path,
    plan: dict[str, Any],
    evidence_dir: Path,
    project: str,
    timeout_s: int,
    priority: int,
    batch_id: str | None,
    order_effect_bound_ppm: int,
    observed_window_seconds: float | None,
) -> dict[str, Any]:
    from merlin.perf.firesim_batch import WINDOW_KIND_CONTIGUOUS, admit_batch, link_batch, load_batch_validation_policy
    from merlin.perf.firesim_checkpoint import QueueHost, QueueSubmission, submit
    from merlin.perf.firesim_receipt import (
        FireSimReceiptError,
        parse_batched_firesim_receipt,
        parse_queued_firesim_receipt,
        write_queued_firesim_receipt,
    )

    host = QueueHost(queue_executable=QUEUE_EXECUTABLE, queue_state_root=QUEUE_STATE_ROOT, chipyard=chipyard_root())
    hwdb_entry = _hwdb_entry()
    submission = QueueSubmission(
        host=host,
        workload=WORKLOAD,
        bootbinary=BOOTBINARY,
        elf=elf,
        hw_config=HW_CONFIG,
        hwdb_config_artifact=hwdb_entry,
        timeout_s=timeout_s,
        priority=priority,
        project=project,
    )
    evidence = submit(submission, evidence_dir, env=client_environment())
    result: dict[str, Any] = {
        "job_id": evidence.job_id,
        "wall_seconds": evidence.wall_s,
        "uartlog_sha256": _sha256(evidence.uart_log),
        "evidence_dir": str(evidence_dir),
    }
    uart_text = evidence.uart_log.read_text(encoding="utf-8", errors="replace")
    # The FPGA console emits CRLF; the admission rules compare exact lines.
    normalized = evidence_dir / "uartlog.normalized"
    normalized.write_text("\n".join(line.rstrip("\r") for line in uart_text.splitlines()) + "\n", encoding="utf-8")

    policy = load_batch_validation_policy(policy_path)
    if batch_id is None:
        receipt = parse_queued_firesim_receipt(
            queue_client_log=evidence.client_log,
            queue_daemon_log=evidence.daemon_log,
            uart_log=normalized,
            expected_queue_executable=QUEUE_EXECUTABLE,
            expected_submission_json=evidence.submission_json,
            expected_job_id=evidence.job_id,
            expected_workload=WORKLOAD,
            validation_policy_json=policy_path,
            cycle_claim_window=plan["declared"][0]["label"],
        )
        written = write_queued_firesim_receipt(receipt, evidence_dir / "firesim-receipt.json")
        result["sealed"] = True
        result["receipt"] = str(written)
        cycles = receipt.queue_receipt.warm_profile.total_compute_cycles
        result["windows"] = [{"label": plan["declared"][0]["label"], "status": "pass", "cycles": cycles}]
        return result

    from merlin.perf.firesim_batch import BatchMember

    members = [
        BatchMember(
            label=row["label"],
            program_sha256=plan["elf_sha256"],
            weights_sha256=plan["blob_sha256"],
            observed_window_seconds=float(observed_window_seconds or 1.0),
            markers=(),
        )
        for row in plan["declared"]
        if not row["is_order_control"]
    ]
    batch = link_batch(
        members,
        batch_id=batch_id,
        queue_wall_limit_seconds=float(timeout_s),
        order_effect_bound_ppm=order_effect_bound_ppm,
    )
    result["batch"] = batch.to_dict()
    # The admission is recorded WHETHER OR NOT it passes. A refused batch is evidence about the
    # run -- which window ran and was wrong, which never ran -- and dropping it would leave only
    # the exception text, which says less than the per-window verdicts do.
    admission = admit_batch(normalized.read_text(encoding="utf-8"), batch, policy)
    result["admission"] = admission.to_dict()
    try:
        receipt = parse_batched_firesim_receipt(
            queue_client_log=evidence.client_log,
            queue_daemon_log=evidence.daemon_log,
            uart_log=normalized,
            expected_queue_executable=QUEUE_EXECUTABLE,
            expected_submission_json=evidence.submission_json,
            expected_job_id=evidence.job_id,
            expected_workload=WORKLOAD,
            validation_policy_json=policy_path,
            batch=batch,
        )
    except FireSimReceiptError as refusal:
        result["sealed"] = False
        result["refusal"] = str(refusal)
        return result
    written = write_queued_firesim_receipt(receipt, evidence_dir / "firesim-batch-receipt.json")
    result["sealed"] = True
    result["receipt"] = str(written)
    result["order_effect_ppm"] = receipt.order_effect_ppm
    result["windows"] = [
        {
            "label": window.label,
            "status": "pass",
            "cycles": window.cycles,
            "is_order_control": window.is_order_control,
        }
        for window in receipt.windows
    ]
    return result


def _hwdb_entry() -> Path:
    from merlin.common.paths import artifacts_dir

    path = artifacts_dir() / "measurements" / f"firesim_{HW_CONFIG}" / "_design" / "hwdb_entry.yaml"
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"no hwdb entry for {HW_CONFIG} at {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build", "gsim", "run"))
    parser.add_argument("--window", action="append", required=True, help="MxNxK:arm; repeatable, in run order")
    parser.add_argument("--target", default="gemmini")
    parser.add_argument("--scale", type=float, default=0.00390625)
    parser.add_argument("--relu", action="store_true")
    parser.add_argument(
        "--elem-mode",
        default="i8_full",
        help="how the int8 operands are drawn (merlin.perf.layer_bench.reference.ELEM_MODES). A batch "
        "holding an arm that reads the accumulator at scale 1 needs sparse_binary, or its digest "
        "checks a saturation pattern instead of the values.",
    )
    parser.add_argument("--bias-span", type=int, default=None, help="bias magnitude; 0 for a bias-free GEMM")
    parser.add_argument("--autocomp-root", type=Path, default=None, help="read-only checkout of the AutoComp artifact")
    parser.add_argument(
        "--generated-kernel",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "published" / "autocomp_fig30_32" / "sol2_autocomp_generated.c",
        help="the AutoComp-generated kernel the paper prints, transcribed into compilable C (the "
        "artifact ships no machine-readable copy). Used by --window MxNxK:autocomp_generated.",
    )
    parser.add_argument("--package", type=Path, default=None, help="package dir holding manifest.yaml")
    parser.add_argument("--package-readout", default="requant_i8", help="what the package is asked to leave in memory")
    parser.add_argument("--batch-id", default=None, help="frame the program as a batch; omit for one solo window")
    parser.add_argument("--order-control", action="store_true", help="append a repeat of window 0")
    parser.add_argument("--order-effect-bound-ppm", type=int, default=10_000)
    parser.add_argument("--project", default="gemm-windows")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--priority", type=int, default=5)
    parser.add_argument("--observed-window-seconds", type=float, default=None)
    parser.add_argument("--no-gsim", action="store_true", help="skip the simulator preflight")
    parser.add_argument("--gsim-max-cycles", type=int, default=400_000_000)
    parser.add_argument("--gsim-timeout-s", type=float, default=7200)
    parser.add_argument("--out-dir", type=Path, default=None)
    arguments = parser.parse_args(argv)

    from merlin.common.paths import artifacts_dir

    members = [parse_window(text) for text in arguments.window]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = arguments.out_dir or (artifacts_dir() / "perf-bench" / arguments.target / f"firesim_gemm_windows_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = out_dir / "work"

    plan = build_windows(
        members,
        target=arguments.target,
        scale=arguments.scale,
        relu=arguments.relu,
        workdir=workdir,
        order_control=arguments.order_control,
        elem_mode=arguments.elem_mode,
        bias_span=arguments.bias_span,
        autocomp_root=arguments.autocomp_root.resolve() if arguments.autocomp_root else None,
        generated_kernel=arguments.generated_kernel.resolve() if arguments.generated_kernel else None,
        package=arguments.package.resolve() if arguments.package else None,
        package_readout=arguments.package_readout,
    )
    built = render_and_build(plan, target=arguments.target, workdir=workdir, batch_id=arguments.batch_id)
    plan.update(built)
    plan.pop("backend", None)
    elf = out_dir / "gemm_windows.elf"
    shutil.copyfile(built["elf"], elf)

    policy_document = validation_policy(plan, policy_id=f"gemmini/gemm-windows/{arguments.batch_id or 'solo'}")
    policy_path = out_dir / "uart_validation_policy_v2.json"
    policy_path.write_text(json.dumps(policy_document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    document: dict[str, Any] = {
        "schema": "firesim_gemm_windows_v1",
        "target": arguments.target,
        "batch_id": arguments.batch_id,
        "scale": arguments.scale,
        "relu": bool(arguments.relu),
        "elem_mode": arguments.elem_mode,
        "bias_span": arguments.bias_span,
        "autocomp_root": str(arguments.autocomp_root) if arguments.autocomp_root else None,
        "generated_kernel": str(arguments.generated_kernel) if arguments.generated_kernel else None,
        "package": str(arguments.package) if arguments.package else None,
        "package_readout": arguments.package_readout if arguments.package else None,
        "contract_digest": plan["contract_digest"],
        "blob_sha256": plan["blob_sha256"],
        "blob_bytes": plan["blob_bytes"],
        "elf": str(elf),
        "elf_sha256": built["elf_sha256"],
        "loaded_bytes": built["loaded_bytes"],
        "source_sha256": built["source_sha256"],
        "validation_policy": str(policy_path),
        "windows": plan["declared"],
        "device": device_provenance(_hwdb_entry()),
    }
    print(json.dumps({k: document[k] for k in ("elf_sha256", "loaded_bytes", "blob_bytes", "batch_id")}, indent=1))
    for row in plan["declared"]:
        print(f"  window {row['label']:<40} {row['shape']:>16} {row['arm']:<9} digest={row['digest_expected']}")

    if arguments.command == "gsim" or (arguments.command == "run" and not arguments.no_gsim):
        document["gsim_preflight"] = run_on_gsim_first(
            elf, target=arguments.target, max_cycles=arguments.gsim_max_cycles, timeout_s=arguments.gsim_timeout_s
        )
        print(json.dumps({k: document["gsim_preflight"][k] for k in ("completed", "wall_seconds", "engine_cycles")}))

    if arguments.command == "run":
        # This checkout-only benchmark uses the example-owned FireSim policy.
        sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
        from examples.gemmini.phase2.firesim_loop import LOOP_PRIORITY, default_governor, guarded_submission

        def _submit() -> dict[str, Any]:
            return submit_and_seal(
                elf=elf,
                policy_path=policy_path,
                plan=plan | {"elf_sha256": built["elf_sha256"]},
                evidence_dir=out_dir / "evidence",
                project=arguments.project,
                timeout_s=arguments.timeout,
                priority=arguments.priority,
                batch_id=arguments.batch_id,
                order_effect_bound_ppm=arguments.order_effect_bound_ppm,
                observed_window_seconds=arguments.observed_window_seconds,
            )

        # A submission at the LOOP priority is an automated one, and an automated loop goes through
        # the guards the queue does not provide: a self-imposed FPGA-seconds budget, a yield while
        # another user is queued, and a halt after consecutive failures. A human submitting at the
        # queue's own default tier keeps the direct path; the guards are for the caller that can
        # resubmit faster than anyone can notice.
        if arguments.priority == LOOP_PRIORITY:
            governor = default_governor(run_id=arguments.batch_id or arguments.project)
            with guarded_submission(
                governor,
                queue_executable=QUEUE_EXECUTABLE,
                window_count=len(plan["declared"]),
                note=arguments.batch_id or arguments.project,
            ) as settlement:
                document["firesim"] = _submit()
                if document["firesim"].get("wall_seconds"):
                    settlement.observed(
                        float(document["firesim"]["wall_seconds"]), job_id=document["firesim"].get("job_id")
                    )
                if not document["firesim"].get("sealed"):
                    settlement.failed(str(document["firesim"].get("refusal") or "the batch was not sealed"))
            document["firesim"]["loop_guard"] = settlement.verdict.to_dict()
        else:
            document["firesim"] = _submit()

    (out_dir / "firesim_gemm_windows.json").write_text(
        json.dumps(document, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
