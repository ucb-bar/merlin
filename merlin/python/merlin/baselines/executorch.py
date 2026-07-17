"""ExecuTorch + XNNPACK baseline arm — run OUR models whole-model on the K1 with RVV, honestly.

ExecuTorch is the *forced-whole-model* arm and, by design, the one that surfaces the MOST scalar
fallback: ExecuTorch delegates whatever the XNNPACK partitioner can claim to XNNPACK's RVV
microkernels, and runs everything else on the **portable (scalar) reference kernels**. This arm does
not hide that — it labels every portable-kernel region as a :class:`ScalarFallback` and reports the
binary-level RVV coverage from :mod:`.rvv_audit`.

Pipeline (per model, fp32 first)::

    model2MLIR loader.get_model_and_inputs()   (HF torch module)
      -> torch.export.export (with OUR captured input, so it matches golden.npy)
      -> to_edge_transform_and_lower(partitioner=[XnnpackPartitioner()])   [XNNPACK RVV delegate]
      -> BundledProgram(.bpte): one test case = (captured input) -> golden.npy      [AOT, ET venv]
      -> cmake --preset riscv64-linux, SpacemiT-clang toolchain, -march=rv64gcv,
         EXECUTORCH_BUILD_XNNPACK=ON  ->  executor_runner (rv64gcv glibc ELF)        [the RVV binary]
      -> rvv_audit.audit_binary(executor_runner + libXNNPACK.a objects)   [mechanical RVV honesty]
      -> push + run on the K1 (board_lock): bundled-IO Test_result: PASS/FAIL + error stats + timing

Why the AOT export runs in a separate venv: ExecuTorch + its pinned torch are heavy and are NOT in
merlin's ``.venv``. This runner shells out to ``build/baselines/executorch/et-venv`` (built once by
``third_party/baselines/executorch/install_executorch.sh``) to produce the ``.bpte``, then does all
the cross-compile / audit / board work itself. If that venv is absent, the model is a clean
``not_built`` gap with a specific reason — never a fabricated result.

Honesty (``not_run_is_not_pass``): torch.export failure, an unsupported op, a cross-compile break, a
board-down condition — each yields a ``not_built``/``not_run`` result with a SPECIFIC ``gap_reason``.
Correctness is gated on the board via the bundled-IO comparison against OUR golden.npy (max relative
error < tolerance) plus a cosine computed from the dumped output; we never invent a cos/rel.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from merlin.baselines import bundle as _bundle
from merlin.baselines import k1_exec, profile, rvv_audit
from merlin.baselines.contract import BaselineResult, RegionProfile, ScalarFallback
from merlin.common import artifacts
from merlin.common.paths import build_dir, repo_root
from merlin.rvvgen import k1

FRAMEWORK = "executorch"

# --- layout (build/ is gitignored; the ET source tree is the pinned submodule) ------------------
_BUILD_ROOT = build_dir() / "baselines" / "executorch"
_ET_SRC = repo_root() / "third_party" / "baselines" / "executorch"
_TOOLCHAIN_CMAKE = Path(__file__).with_name("executorch_spacemit_toolchain.cmake")
_ET_EXPORT_HELPER = Path(__file__).with_name("_et_export.py")


def et_venv_python() -> Path:
    """Python interpreter of the ExecuTorch export venv. Override with ``MERLIN_ET_VENV``."""
    v = os.environ.get("MERLIN_ET_VENV")
    if v:
        return Path(v) / "bin" / "python"
    return _BUILD_ROOT / "et-venv" / "bin" / "python"


def et_venv_available() -> bool:
    """True iff the export venv exists and can import executorch + torch."""
    py = et_venv_python()
    if not py.is_file():
        return False
    r = subprocess.run([str(py), "-c", "import executorch, torch"],
                       capture_output=True, text=True, timeout=120)
    return r.returncode == 0


def et_commit() -> str:
    try:
        r = subprocess.run(["git", "-C", str(_ET_SRC), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


class ExecuTorchError(RuntimeError):
    pass


def _run(cmd: list, **kw) -> subprocess.CompletedProcess:
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        raise ExecuTorchError(
            f"command failed: {' '.join(map(str, cmd))[:400]}\n"
            f"STDOUT:{proc.stdout[-2000:]}\nSTDERR:{proc.stderr[-2000:]}")
    return proc


# --- capture-bundle resolution (legacy fp32 LLM dir names, like buddy) --------------------------

_LEGACY_FP32_DIRS: dict[str, str] = {
    "tiny_llama": "tiny_consistent",
    "small_llama": "small_consistent",
}


def resolve_bundle(model: str, variant: str = "fp32") -> _bundle.CaptureBundle:
    b = _bundle.resolve(model, variant)
    if b.golden.is_file() and b.inputs.is_file():
        return b
    if variant == "fp32" and model in _LEGACY_FP32_DIRS:
        legacy = artifacts.recaptures_dir() / _LEGACY_FP32_DIRS[model]
        if (legacy / "golden.npy").is_file():
            return _bundle.CaptureBundle(model=model, variant=variant, root=legacy)
    return b


# --- AOT export (in the ET venv) ----------------------------------------------------------------

@dataclass
class ExportResult:
    pte: Path
    ptd_files: list[Path]
    input_files: list[Path]
    golden: Path
    delegated_nodes: int | None = None
    total_call_nodes: int | None = None
    summary: dict | None = None


def export_pte(model: str, b: _bundle.CaptureBundle, work: Path, *,
               xnnpack: bool = True, quantize: bool = False, compute_golden: bool = False,
               int8_subgraph: bool = False, int8_whole_model: bool = False,
               extra_env: dict[str, str] | None = None, timeout: int = 3600) -> ExportResult:
    """Run the AOT export helper under the ET venv to produce ``model.pte`` (+ ``.ptd`` weights).

    Uses OUR captured ``inputs.npz`` (so the export trace matches the golden) and writes the raw
    input bytes + external-constant ``.ptd`` alongside the ``.pte``. A whole fp32 LLM's weights blow
    past flatbuffer's 2 GB program limit if embedded, hence external constants. Raises
    :class:`ExecuTorchError` with a concrete message on torch.export / lowering failure.

    ``compute_golden``: recompute the reference from the eager torch model on the captured input
    (used for a layer-reduced fit-on-board config whose captured golden was made with the full
    model); the correctness gate then compares ExecuTorch vs eager-torch for THIS exact config.
    ``extra_env``: passed to the export subprocess (e.g. ``M2M_LLAMA_LAYERS`` for the reduced build).
    """
    py = et_venv_python()
    loader = b.torch_loader
    if not loader.is_file():
        raise ExecuTorchError(f"model2MLIR torch loader missing at {loader} "
                              "(ExecuTorch ingests torch; no loader -> cannot export)")
    # int8-subgraph / whole-model int8 force compute-golden (the fp32 reference is recomputed from
    # THIS exact loaded model, so the int8-vs-fp32 cosine is measured against the right baseline).
    if int8_subgraph or int8_whole_model:
        compute_golden = True
    work = work.resolve()   # the subprocess runs with cwd=work, so all paths must be absolute
    work.mkdir(parents=True, exist_ok=True)
    out = work / "model.pte"
    # If compute_golden, write the golden into the work dir (do not clobber the captured golden).
    golden = (work / "golden.npy") if compute_golden else b.golden.resolve()
    cmd = [py, str(_ET_EXPORT_HELPER),
           "--loader", str(loader.resolve()), "--inputs-npz", str(b.inputs.resolve()),
           "--golden-npy", str(golden), "--out", str(out), "--model-name", model,
           "--m2m-root", str(_bundle.model2mlir_root())]
    if not xnnpack:
        cmd.append("--no-xnnpack")
    if quantize:
        cmd.append("--quantize")
    if compute_golden:
        cmd.append("--compute-golden")
    if int8_subgraph:
        cmd.append("--int8-subgraph")
    if int8_whole_model:
        cmd.append("--int8-whole-model")
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    # Run from the work dir (NOT this package dir): the sibling ``executorch.py`` would otherwise
    # sit on sys.path[0] and shadow the installed ``executorch`` package in the ET venv.
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                          timeout=timeout, env=env, cwd=str(work))
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-1200:]
        raise ExecuTorchError(f"AOT export failed: {tail}")
    summary = None
    for line in proc.stdout.splitlines():
        if line.startswith("ET_EXPORT_JSON "):
            summary = json.loads(line[len("ET_EXPORT_JSON "):])
    if not out.is_file():
        raise ExecuTorchError(f"export produced no .pte at {out}: {proc.stdout[-400:]}")
    s = summary or {}
    return ExportResult(pte=out,
                        ptd_files=[Path(p) for p in s.get("ptd_files", [])],
                        input_files=[Path(f["path"]) for f in s.get("input_files", [])],
                        golden=golden,
                        delegated_nodes=s.get("delegated_nodes"),
                        total_call_nodes=s.get("total_call_nodes"),
                        summary=summary)


# --- cross-compile the executor_runner for rv64gcv (SpacemiT clang) -----------------------------

def _toolchain_root() -> Path | None:
    return k1._toolchain_root()


def cross_compile_runner(work: Path, *, xnnpack: bool = True, etdump: bool = False,
                         timeout: int = 5400) -> Path:
    """Cross-compile ExecuTorch's ``executor_runner`` for rv64gcv with the SpacemiT clang.

    Uses the pinned source tree's ``riscv64-linux`` cmake preset but overrides its toolchain file
    with :data:`_TOOLCHAIN_CMAKE` (SpacemiT clang, ``-march=rv64gcv -mabi=lp64d`` on the whole
    build) and enables ``EXECUTORCH_BUILD_XNNPACK`` so XNNPACK's RVV microkernels are linked in.
    Enforces the vector march (``rvv_audit.enforce_rvv_march``) before configuring — no scalar-only
    binary slips through. The build is cached under ``build/baselines/executorch/cmake-out``; a
    stale/failed configure is wiped and retried once.

    ``etdump=True`` flips ``EXECUTORCH_BUILD_RISCV_ETDUMP`` ON (→ devtools + event tracer in the
    preset), so the runner accepts ``--etdump_path`` and emits per-op timing events the devtools
    Inspector correlates back to layer fqns (the per-region ExecuTorch timing). It is a DISTINCT
    build (cached under a separate dir) from the plain runner so the two do not clobber each other.
    """
    rvv_audit.enforce_rvv_march(k1.K1_MARCH)
    root = _toolchain_root()
    if root is None:
        raise ExecuTorchError("SpacemiT toolchain not found (set MERLIN_K1_TOOLCHAIN)")

    build_dir = work / ("cmake-out-etdump" if etdump else "cmake-out")
    env = dict(os.environ)
    env["MERLIN_K1_TOOLCHAIN_ROOT"] = str(root)

    # ExecuTorch's cmake runs host-side codegen (gen_oplist, flatc bindings) via PYTHON_EXECUTABLE;
    # it must be the ET venv python (has executorch + the codegen module), NOT merlin's .venv.
    py = et_venv_python()
    cfg = [
        "cmake", "-S", str(_ET_SRC), "-B", str(build_dir),
        "--preset", "riscv64-linux",
        f"-DCMAKE_TOOLCHAIN_FILE={_TOOLCHAIN_CMAKE}",
        f"-DPYTHON_EXECUTABLE={py}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if xnnpack:
        cfg.append("-DEXECUTORCH_BUILD_XNNPACK=ON")
    if etdump:
        cfg.append("-DEXECUTORCH_BUILD_RISCV_ETDUMP=ON")

    def _configure_and_build() -> None:
        subprocess.run([str(c) for c in cfg], capture_output=True, text=True,
                       timeout=timeout, env=env, check=True)
        subprocess.run(["cmake", "--build", str(build_dir), "-j", str(os.cpu_count() or 8),
                        "--target", "executor_runner"],
                       capture_output=True, text=True, timeout=timeout, env=env, check=True)

    try:
        _configure_and_build()
    except subprocess.CalledProcessError as e:
        # Wipe a possibly-stale configure and retry once with full logs on failure.
        shutil.rmtree(build_dir, ignore_errors=True)
        try:
            _configure_and_build()
        except subprocess.CalledProcessError as e2:
            out = (e2.stdout or "") + (e2.stderr or "")
            raise ExecuTorchError(f"executor_runner cross-compile failed: {out[-2000:]}") from e2

    runner = build_dir / "executor_runner"
    if not runner.is_file():
        raise ExecuTorchError(f"cross-compile produced no executor_runner at {runner}")
    # Sanity: it must be a RISC-V ELF, not a host binary.
    fr = subprocess.run(["file", str(runner)], capture_output=True, text=True, timeout=30)
    if "RISC-V" not in fr.stdout:
        raise ExecuTorchError(f"executor_runner is not a RISC-V ELF: {fr.stdout.strip()}")
    return runner


# --- RVV audit (executor_runner + XNNPACK objects) ----------------------------------------------

# Symbols that are harness/CRT/libc plumbing, not model-compute kernels — excluded from the
# scalar-fallback list (we only label compute kernels that stayed scalar).
_IGNORE_SYMS = ("_start", "__libc", "abort", "printf", "puts", "fwrite", "memcpy", "memset",
                "memmove", "malloc", "free", "pthread", "clock_", "frame_dummy",
                "register_tm", "gflags", "std::", "__cxx", "operator", "_GLOBAL__",
                "et_pal", "executorch::runtime", "flatbuffers")


def _region_of_symbol(sym: str) -> str:
    s = sym.lower()
    if any(t in s for t in ("gemm", "matmul", "igemm", "spmm", "conv", "dwconv", "fully_connected")):
        return "gemm"
    if any(t in s for t in ("softmax", "attention", "attn", "sdpa")):
        return "attention"
    if any(t in s for t in ("norm", "rsqrt", "layernorm", "rmsnorm", "mean")):
        return "norm"
    if any(t in s for t in ("add", "mul", "gelu", "silu", "sigmoid", "vunary", "vbinary",
                            "elementwise", "clamp", "relu", "exp")):
        return "elementwise"
    return "other"


def _preferred_objdump() -> str | None:
    """Prefer the toolchain's ``llvm-objdump`` for RVV decoding.

    The SpacemiT GNU ``riscv64-unknown-linux-gnu-objdump`` silently mis-decodes rv64gcv vector
    instructions in bulk ``-d`` mode (it decodes a forced address range correctly, but a whole-file
    ``-d`` emits ~3 vector insns for a binary that actually has ~85k) — which would fabricate a
    false 0% RVV coverage. ``llvm-objdump`` decodes them correctly, so we pin it for this arm. We do
    NOT patch the shared ``rvv_audit._objdump`` (other arms depend on it); we pass ``objdump=`` here.
    """
    root = _toolchain_root()
    if root is not None:
        cand = root / "bin" / "llvm-objdump"
        if cand.is_file():
            return str(cand)
    import shutil as _sh
    return _sh.which("llvm-objdump")


def audit_binary(runner: Path) -> tuple[float | None, list[ScalarFallback], dict]:
    """RVV-audit the executor_runner ELF (statically links XNNPACK + portable kernels).

    Returns (coverage_overall, fallbacks, per-symbol dict). Every compute-bearing symbol with zero
    vector instructions is labeled a scalar fallback (reason 'no XNNPACK RVV ukernel (portable
    kernel)') — this is where ExecuTorch's substantial scalar surface is recorded honestly.
    """
    report = rvv_audit.audit_binary(runner, objdump=_preferred_objdump())
    fallbacks = [
        ScalarFallback(symbol=sym, reason="no XNNPACK RVV ukernel (portable/scalar kernel)",
                       region=_region_of_symbol(sym))
        for sym in report.scalar_fallback_symbols(ignore=_IGNORE_SYMS)
    ]
    by_symbol = {n: {"vector": sc.vector, "scalar_compute": sc.scalar_compute,
                     "coverage": sc.coverage} for n, sc in report.by_symbol.items()}
    return report.coverage_overall, fallbacks, by_symbol


# --- on-board run + parse -----------------------------------------------------------------------

# executor_runner log lines we parse.
_TIME_RE = re.compile(r"Model executed successfully .* in ([\d.]+) ms")
_LOAD_RE = re.compile(r"Model loaded in ([\d.]+) ms")
_ITER_RE = re.compile(r"Iteration \d+ of \d+: ([\d.]+) ms")


@dataclass
class BoardRun:
    ran: bool = False
    wall_ns: int | None = None          # execution time (ms->ns) reported by executor_runner
    load_ns: int | None = None          # model-load time (ms->ns)
    cos: float | None = None
    rel: float | None = None
    console: str = ""


def _board_free_bytes() -> int | None:
    """Free bytes on the board's rootfs (K1_REMOTE_DIR lives there), or None if unreachable."""
    try:
        r = k1_exec.run(["df", "-k", k1_exec.K1_REMOTE_DIR])
        for line in r.stdout.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 4:
                return int(parts[3]) * 1024
    except Exception:  # noqa: BLE001
        pass
    return None


def _run_on_board(res: BaselineResult, runner: Path, exp: "ExportResult",
                  *, num_executions: int = 1, timeout: int = 1200,
                  mmap_model: bool = False) -> BoardRun:
    """Push runner + .pte + .ptd + input(s) to the K1, run under the board lock, dump the output.

    Uses the stock executor_runner's ``--model_path`` / ``--data_path`` (external weights) /
    ``--inputs`` (raw captured input bytes) / ``--output_file`` (dump to tmpfs) path. The runner
    prints ``Model loaded in X ms`` and ``Model executed successfully N time(s) in Y ms`` — Y is the
    honest E2E wall time (this foreign runner does NOT expose the K1 rdtime CSR, so wall_ns is the
    truth and we do NOT fabricate a tick/cycle count). The output is scp'd back and compared to the
    golden OFF-DEVICE (cos/rel) — never a fabricated correctness number. Fail-closed if board down.

    Board is disk-constrained and SHARED with other agents: we check free space before pushing and
    fail-closed (``not_run`` with a specific reason) if the model won't fit, rather than filling a
    shared board's disk.
    """
    out = BoardRun()
    total = runner.stat().st_size + exp.pte.stat().st_size
    total += sum(p.stat().st_size for p in exp.ptd_files if p.is_file())
    total += sum(p.stat().st_size for p in exp.input_files if p.is_file())

    with k1_exec.board_lock():
        free = _board_free_bytes()
        if free is not None and free < total + 64 * 1024 * 1024:  # 64MB headroom
            raise k1_exec.BoardUnavailable(
                f"board rootfs has {free/1e9:.2f} GB free but the model needs {total/1e9:.2f} GB "
                f"(shared board is disk-constrained; not filling it)")
        # A whole-model .pte can be multi-GB; the default 300 s scp timeout truncates it (which then
        # fails on the board). Scale the timeout to the payload (~5 MB/s worst case over this link).
        def _push(p: Path, remote: str) -> str:
            secs = max(300, int(p.stat().st_size / (2 * 1024 * 1024)) + 120)
            return k1_exec.push(p, remote, timeout=secs)

        remote_runner = _push(runner, f"{k1_exec.K1_REMOTE_DIR}/executor_runner")
        remote_pte = _push(exp.pte, f"{k1_exec.K1_REMOTE_DIR}/model.pte")
        remote_ptds = [_push(p, f"{k1_exec.K1_REMOTE_DIR}/{p.name}")
                       for p in exp.ptd_files if p.is_file()]
        remote_inputs = [_push(p, f"{k1_exec.K1_REMOTE_DIR}/{p.name}")
                         for p in exp.input_files if p.is_file()]
        remote_out = "/tmp/et_out"   # tmpfs (RAM, 1.9G) — output is small (few MB), keeps flash free
        local_out = exp.pte.parent / "et_out-0.bin"
        argv = [remote_runner, f"--model_path={remote_pte}",
                f"--num_executions={num_executions}",
                f"--output_file={remote_out}", "--print_output=none"]
        if mmap_model:
            # mmap the (multi-GB) program so its const weight pages demand-load and stay evictable
            # under the board RAM ceiling instead of being read fully-resident by FileDataLoader.
            argv.append("--mmap_model=true")
        if remote_ptds:
            argv.append(f"--data_path={remote_ptds[0]}")
        if remote_inputs:
            argv.append("--inputs=" + ",".join(remote_inputs))
        try:
            k1_exec.run(["chmod", "+x", remote_runner])
            proc = k1_exec.run(argv, timeout=timeout)
            # pull the dumped output (output 0) for off-device cos/rel.
            try:
                _scp_from_board(f"{remote_out}-0.bin", local_out)
            except Exception:  # noqa: BLE001
                local_out = None  # type: ignore[assignment]
        finally:
            try:
                k1_exec.run(["rm", "-f", remote_runner, remote_pte, *remote_ptds,
                             *remote_inputs, f"{remote_out}-0.bin"])
            except Exception:  # noqa: BLE001
                pass

    console = proc.stdout + proc.stderr
    out.console = console
    m = _TIME_RE.search(console)
    if m:
        out.ran = True
        out.wall_ns = int(round(float(m.group(1)) * 1e6))
    lm = _LOAD_RE.search(console)
    if lm:
        out.load_ns = int(round(float(lm.group(1)) * 1e6))
    # correctness: compare the dumped output to the golden OFF-DEVICE.
    if out.ran and local_out is not None and Path(local_out).is_file():
        out.cos, out.rel = _cos_rel(Path(local_out), exp.golden)
    return out


def _scp_from_board(remote: str, local: Path) -> None:
    """scp a file FROM the board (k1_exec only pushes)."""
    opts = ["-i", k1_exec.K1_SSH_KEY, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]
    r = subprocess.run(["scp", *opts, f"{k1_exec.K1_HOST}:{remote}", str(local)],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise ExecuTorchError(f"scp from board failed: {r.stderr[:200]}")


def _cos_rel(out_bin: Path, golden: Path) -> tuple[float | None, float | None]:
    """Cosine similarity + relative L2 error of a dumped fp32 output vs the golden .npy."""
    try:
        import numpy as np

        gold = np.load(golden).astype(np.float64).ravel()
        got = np.fromfile(out_bin, dtype=np.float32).astype(np.float64).ravel()
        n = min(gold.size, got.size)
        if n == 0:
            return None, None
        g, a = gold[:n], got[:n]
        denom = (np.linalg.norm(a) * np.linalg.norm(g)) or 1.0
        cos = float(np.dot(a, g) / denom)
        rel = float(np.linalg.norm(a - g) / (np.linalg.norm(g) or 1.0))
        return cos, rel
    except Exception:  # noqa: BLE001
        return None, None


# --- the runner ---------------------------------------------------------------------------------

# LLM subset first (cleanest torch.export + XNNPACK partition), then attempt the rest. Whole-model
# is FORCED for every model — ExecuTorch has no per-op opt-out here.
DEFAULT_MODELS = ("tiny_llama", "small_llama", "bitvla", "rdt2", "rdt", "openvla",
                  "molmoact", "groot_n1d7", "xr0", "pi05", "smolvla")


def run_model(model: str, variant: str = "fp32", *, work_root: Path | None = None,
              write: bool = True, run_board: bool | None = None,
              xnnpack: bool = True, quantize: bool | None = None, compute_golden: bool = False,
              int8_subgraph: bool = False, int8_whole_model: bool | None = None,
              full_fidelity: bool = True,
              export_env: dict[str, str] | None = None,
              runner_override: Path | None = None) -> BaselineResult:
    """Run one (model, variant) through the ExecuTorch+XNNPACK arm end-to-end -> BaselineResult.

    Re-runnable: with the board down it still produces a ``not_run`` result carrying the built
    rv64gcv executor_runner, its RVV coverage, and the labeled scalar-fallback table; a later
    invocation with the board up fills in timing + correctness with no code change.

    ``compute_golden`` / ``export_env`` support a layer-reduced fit-on-board build (e.g.
    ``export_env={'M2M_LLAMA_LAYERS': '1'}`` + ``compute_golden=True``): the whole fp32 LLM (~4 GB)
    does not fit the shared 3.8 GB board, so a reduced config that DOES fit is run to prove the
    ExecuTorch+XNNPACK-RVV path end-to-end on real silicon; the full-config attempt is recorded as a
    board-fit gap. ``runner_override`` reuses an already-built executor_runner (it is model-agnostic).
    """
    # int8 variant defaults to ExecuTorch's OFFICIAL whole-model llama recipe (source-transform
    # weight-only int8 per-channel) — the path that unblocks full-model int8 on HF Llama. Falls back
    # to the decoder-linear subgraph only if explicitly requested.
    if int8_whole_model is None:
        int8_whole_model = (variant == "int8") and not int8_subgraph
    if quantize is None:
        quantize = (variant == "int8")
    if int8_subgraph or int8_whole_model:
        quantize = True
    # Full-fidelity: load the model with the exact loader env the golden was captured on (so we
    # ingest the IDENTICAL architecture). Merged UNDER any explicit export_env override.
    if full_fidelity:
        ff = _bundle.full_env(model)
        if ff:
            export_env = {**ff, **(export_env or {})}
    cos_thr, rel_thr = _bundle.tolerance(model)
    # W8A8 quantization loses precision vs the fp32 golden — the fp32 gate (cos>=0.9999) is not the
    # right bar for an int8 result. Use an int8-appropriate gate so a genuinely-correct int8 run is
    # not spuriously marked fail (still honest: cos is the MEASURED int8-vs-fp32 cosine, not faked).
    if quantize:
        cos_thr, rel_thr = 0.99, 5e-2
    # Random-init models ship no reproducible weights, so their CAPTURED golden is unreachable by a
    # re-instantiated export: gating against it measures weight provenance, not the framework. Recompute
    # the reference from THIS instance (as the int8 path already does) and LABEL the cell — the cos then
    # means lowering-exactness, never a semantic match. Non-random-init models are untouched.
    lowering_exact_only = _bundle.golden_unreproducible(model) and not compute_golden
    if lowering_exact_only:
        compute_golden = True
    res = BaselineResult(framework=FRAMEWORK, model=model, variant=variant,
                         substrate="k1_spacemit", cos_threshold=cos_thr, rel_threshold=rel_thr,
                         march=k1.K1_MARCH,
                         toolchain="spacemit-clang-19+executorch+xnnpack(rvv)",
                         framework_commit=et_commit(), timestamp=artifacts.utc_stamp())
    if lowering_exact_only:
        res.notes += _bundle.lowering_exactness_note(model)

    b = resolve_bundle(model, variant)
    if not b.golden.is_file():
        res.gap_reason = f"golden missing: {b.root}/golden.npy absent (cannot gate correctness)"
        return _finish(res, model, variant, write)
    if not b.inputs.is_file():
        res.gap_reason = f"inputs missing: {b.root}/inputs.npz absent (needed to match golden)"
        return _finish(res, model, variant, write)
    if not et_venv_available():
        res.gap_reason = ("ExecuTorch export venv unavailable at "
                          f"{et_venv_python()} (build via third_party/baselines/executorch/"
                          "install_executorch.sh); cannot torch.export -> .pte")
        return _finish(res, model, variant, write)

    work = (work_root or (_BUILD_ROOT / "runs")) / f"{model}_{variant}"
    work.mkdir(parents=True, exist_ok=True)

    # 1. AOT export -> .pte (+ .ptd) via the XNNPACK partitioner. torch.export / unsupported-op
    #    failures -> not_built with a specific reason.
    try:
        exp = export_pte(model, b, work, xnnpack=xnnpack, quantize=quantize,
                         compute_golden=compute_golden, int8_subgraph=int8_subgraph,
                         int8_whole_model=int8_whole_model, extra_env=export_env)
        if exp.summary and exp.summary.get("subgraph_note"):
            res.notes += " " + exp.summary["subgraph_note"]
        if exp.delegated_nodes is not None:
            res.notes += (f" xnnpack_delegated_nodes={exp.delegated_nodes}"
                          f"/{exp.total_call_nodes}")
        if quantize:
            res.notes += " pt2e_w8a8=True"
        if compute_golden:
            res.notes += " golden=eager-torch(this-config)"
        if export_env:
            res.notes += " export_env=" + ",".join(f"{k}={v}" for k, v in export_env.items())
    except ExecuTorchError as e:
        res.gap_reason = f"torch.export/.pte lowering failed: {str(e)[:400]}"
        return _finish(res, model, variant, write)

    # 2. cross-compile executor_runner (rv64gcv, XNNPACK RVV). The runner is model-agnostic, so an
    #    already-built one may be reused across models via runner_override.
    if runner_override is not None and Path(runner_override).is_file():
        runner = Path(runner_override)
        res.built = True
        res.notes += f" runner={runner} (reused)"
    else:
        try:
            runner = cross_compile_runner(work, xnnpack=xnnpack)
            res.built = True
            res.notes += f" runner={runner}"
        except ExecuTorchError as e:
            res.gap_reason = f"executor_runner cross-compile failed: {str(e)[:400]}"
            return _finish(res, model, variant, write)

    # 3. RVV audit of the emitted binary — the mechanical honesty (do this before the board).
    try:
        cov, fallbacks, _by = audit_binary(runner)
        res.rvv_coverage_overall = cov
        res.scalar_fallbacks = fallbacks
    except Exception as e:  # noqa: BLE001
        res.notes += f" rvv-audit failed: {str(e)[:150]}"

    # 4. K1 on-board run — the ONLY board-gated step. Fail-closed when the board is down / too full.
    #    RAM-infeasible models (7B-class VLAs: openvla/molmoact/pi05) are BUILT (export succeeds) but
    #    NOT run on-board — their fp32 embeddings alone exceed the 3.8 GB board. Honest not_run gap,
    #    never a false fit. (The _run_on_board free-space guard also fail-closes if the .pte is too
    #    big, but we short-circuit the known-infeasible set to avoid a pointless multi-GB transfer.)
    if model in _bundle.K1_RAM_INFEASIBLE:
        res.gap_reason = (f"{model} is RAM-infeasible whole-model on the K1 (3.8 GB board): a "
                          "7B-class VLA whose fp32 embeddings/weights exceed board RAM even at int8. "
                          "Exported + RVV-audited off-board; not run on-board (no false fit).")
        res.board_vlenb = k1_exec.board_vlenb()
        return _finish(res, model, variant, write)
    # Whole-model int8 is const-folded (dequant weights -> fp32 program constants), giving a
    # multi-GB .pte whose weight pages must demand-load; mmap it so the board's RAM ceiling is not
    # blown by a fully-resident read. The layer-reduced/subgraph paths have small .ptes -> no mmap.
    mmap_model = bool(int8_whole_model)
    do_board = k1_exec.board_available() if run_board is None else run_board
    if do_board:
        try:
            _do_board(res, runner, exp, mmap_model=mmap_model)
        except k1_exec.BoardUnavailable as e:
            res.gap_reason = res.gap_reason or f"K1 board run failed: {str(e)[:250]}"
        except Exception as e:  # noqa: BLE001
            res.gap_reason = res.gap_reason or f"K1 board run error: {str(e)[:250]}"
    else:
        res.gap_reason = "K1 board unavailable (MERLIN_K1_HOST unset / unreachable)"

    res.board_vlenb = k1_exec.board_vlenb()
    return _finish(res, model, variant, write)


def _do_board(res: BaselineResult, runner: Path, exp: "ExportResult",
              *, mmap_model: bool = False) -> None:
    """Run on the board and fill correctness + E2E/region profile from the executor_runner run."""
    br = _run_on_board(res, runner, exp, mmap_model=mmap_model)
    res.ran = br.ran
    if br.wall_ns is not None:
        res.e2e_wall_ns = br.wall_ns
        # This foreign runner does not expose the K1 rdtime CSR; the reported wall time is the
        # honest E2E truth. We record ONE whole-model region on wall time (no fabricated
        # tick/cycle count). A per-region split would need etdump per-op events.
        res.regions = [RegionProfile(name="other", wall_ns=br.wall_ns,
                                     rvv_coverage=res.rvv_coverage_overall,
                                     note="whole-model forward (executor_runner wall time; region "
                                          "split needs etdump per-op events)")]
    # Correctness: compared OFF-DEVICE (cos/rel of the dumped output vs golden). Never fabricated.
    if br.cos is not None:
        res.cos = br.cos
    if br.rel is not None:
        res.rel = br.rel
    if not res.ran and not res.gap_reason:
        res.gap_reason = ("K1 run produced no 'Model executed successfully' line "
                          f"(console tail): {br.console[-300:]}")


def _finish(res: BaselineResult, model: str, variant: str, write: bool) -> BaselineResult:
    res.validate()
    if write:
        m = artifacts.new_measurement("k1_spacemit", model, "cross_framework")
        res.write(m.path)
    return res


def run_all(models=DEFAULT_MODELS, variant: str = "fp32", *, write: bool = True,
            xnnpack: bool = True, **kw) -> list[BaselineResult]:
    out = []
    for m in models:
        try:
            out.append(run_model(m, variant, write=write, xnnpack=xnnpack, **kw))
        except Exception as e:  # noqa: BLE001 - one model must never sink the batch
            r = BaselineResult(framework=FRAMEWORK, model=m, variant=variant,
                               gap_reason=f"runner exception: {str(e)[:200]}",
                               timestamp=artifacts.utc_stamp())
            if write:
                try:
                    md = artifacts.new_measurement("k1_spacemit", m, "cross_framework")
                    r.write(md.path)
                except Exception:  # noqa: BLE001
                    pass
            out.append(r)
    return out


# All 11 m2m models: the 8 K1-runnable + the 3 RAM-infeasible VLAs (attempted, RAM-gapped).
ALL_MODELS = tuple(sorted(_bundle.K1_RUNNABLE | _bundle.K1_RAM_INFEASIBLE))


def run_all_int8(models=ALL_MODELS, *, write: bool = True, runner_override: Path | None = None,
                 run_board: bool | None = None) -> list[BaselineResult]:
    """Whole-model int8 (ExecuTorch official llama recipe) across the full corpus.

    Llama-family models use the source-transform whole-model int8 path; the RAM-infeasible VLAs
    (openvla/molmoact/pi05) are attempted-and-RAM-gapped; non-llama archs that won't torch.export
    (dynamic control flow) surface an honest ``not_built`` with the specific op. Full-fidelity
    loader env is applied automatically (``bundle.full_env``).
    """
    return run_all(models, "int8", write=write, int8_whole_model=True,
                   runner_override=runner_override, run_board=run_board)


def _main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="ExecuTorch + XNNPACK K1-RVV baseline arm")
    ap.add_argument("models", nargs="*", default=list(DEFAULT_MODELS),
                    help="models to run (default: full corpus, LLM subset first)")
    ap.add_argument("--variant", default="fp32")
    ap.add_argument("--no-xnnpack", action="store_true",
                    help="portable-kernel-only baseline (no XNNPACK delegate)")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    results = run_all(tuple(args.models) if args.models else DEFAULT_MODELS, args.variant,
                      write=not args.no_write, xnnpack=not args.no_xnnpack)
    for r in results:
        cov = f"{100*r.rvv_coverage_overall:.0f}%RVV" if r.rvv_coverage_overall is not None else "?RVV"
        print(f"{r.model}/{r.variant}: {r.status():10s} {cov} "
              f"fallbacks={len(r.scalar_fallbacks)} {r.gap_reason}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
