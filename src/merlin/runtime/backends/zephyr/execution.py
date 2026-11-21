"""FireSim workload handoff; infrastructure execution belongs to the operator.

Legacy runner selection remains an explicit migration error. This module never
loads launch plugins, submits queues, provisions machines or runs a simulator.
"""

from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from merlin.compiler.build.zephyr import ZephyrBuildError as ZephyrModelError
from merlin.compiler.compilation import Artifact, Deployment
from merlin.runtime.backends.baremetal import spike_launcher as _spike
from merlin.runtime.backends.zephyr.application import _sustained_stats
from merlin.runtime.backends.zephyr.image_build import build_app
from merlin.runtime.execution.service import ExternalObservation, PreparedRun, collect, prepare

# ---- firesim_runner ----------------------------------------------------------------------------


class FireSimRunner(Protocol):
    """Historical launch signature retained only for import compatibility."""

    def __call__(self, elf: str, **options) -> str: ...


class FireSimRunnerError(ValueError):
    """An unsupported launch request or incomplete external observation."""


def select_runner(name: str | None = None) -> FireSimRunner:
    """Refuse the old launch route before discovering or importing a plugin."""
    raise FireSimRunnerError(
        "FireSim infrastructure execution is external: prepare an image with an explicit "
        "Deployment, then collect the operator's result package; installed runner selection "
        "and MERLIN_FIRESIM_RUNNER no longer launch work from Merlin"
    )


def prepare_image(
    elf: str | Path, *, output: str | Path, deployment: Deployment, inputs: dict[str, Path] | None = None
) -> PreparedRun:
    """Retain the exact ELF and caller-selected inputs in a portable request."""
    return prepare(
        Artifact.from_file(elf, kind="elf", role="entry"), output=output, deployment=deployment, inputs=inputs
    )


def collect_uart(prepared: PreparedRun | str | Path, *, results: str | Path) -> tuple[str, ExternalObservation]:
    """Retain hash-bound observations and return exactly one explicit UART stream.

    Matching file identity does not authenticate the external execution claim.
    The caller still owns parsing and independent numerical admission.
    """
    observed = collect(prepared, results=results)
    streams = [item for item in observed.artifacts if item["role"] == "uart"]
    if len(streams) != 1:
        raise FireSimRunnerError("external results must identify exactly one artifact with role 'uart'")
    if observed.reported_exit_code != 0:
        raise FireSimRunnerError(
            f"external execution reported exit code {observed.reported_exit_code}; observations retained"
        )
    return (observed.directory / streams[0]["path"]).read_text(encoding="utf-8"), observed


# ---- zephyr_execution --------------------------------------------------------------------------
# Runs a built Zephyr image on Spike, Verilator or FireSim and grades its console.

#: Base ISA string for a whole-model RVV spike run. VLEN is NOT part of it — see `spike_isa`.
DEFAULT_SPIKE_ISA = "rv64gcv_zfh_zvfh"


def spike_isa(vlen: int | None = None, base: str = DEFAULT_SPIKE_ISA) -> str:
    """``base`` ISA string with an explicit VLEN, expressed as the ``zvl<N>b`` extension.

    spike takes its vector length from the ISA string, and with no ``zvl`` extension it uses the V
    minimum of 128. That is a silent trap for a board whose vector unit is wider: the image is
    validated at a vector length the hardware does not have, and a fixed-width schedule compiled for
    one VLEN maps to a different LMUL on the other (the measured K1 case -- ``-march=rv64gcv`` assumes
    128 while the board is 256, which doubles LMUL and spills). Pinning it here lets a run state the
    vector length it actually exercised. ``None`` keeps spike's default.
    """
    if vlen is None:
        return base
    if vlen < 128 or vlen & (vlen - 1):
        raise ZephyrModelError(f"VLEN must be a power of two >= 128, got {vlen}")
    return f"{base}_zvl{int(vlen)}b"


def run_on_spike(
    elf: str | Path,
    *,
    dram_base: int,
    mem_bytes: int,
    harts: int = 2,
    isa: str = DEFAULT_SPIKE_ISA,
    vlen: int | None = None,
    timeout: int = 3600,
) -> dict[str, Any]:
    """Run the Zephyr ELF on spike ``-pN``; parse the OUT/ARGMAX/METRIC/DONE markers.

    ``dram_base`` and ``mem_bytes`` describe the selected board's mapped DRAM;
    ``vlen`` pins the simulated vector length (via :func:`spike_isa`).
    """
    isa = spike_isa(vlen, isa) if vlen is not None else isa
    cmd = [_spike.spike_path(), f"--isa={isa}", f"-p{harts}", f"-m{hex(dram_base)}:{hex(mem_bytes)}", str(elf)]
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    console = proc.stdout + proc.stderr
    return _parse_console(console, proc.returncode)


def _verilator_facts(board: str, config: str | None) -> tuple[str | None, str | None]:
    """``(config, env_name)`` for ``board``'s Verilator sim: the SoC config (``config`` if given, else the
    board's declared ``rtl_sim_config``) and the binary override ``MERLIN_<TARGET>_VERILATOR`` derived from
    the board's ``target``. Either is None when the registry does not declare it -- never a default."""
    from merlin.common.paths import target_env_name
    from merlin.runtime.boards.catalog import board as _board_desc

    desc = _board_desc(board)
    cfg = config or desc.rtl_sim_config
    env_name = target_env_name(desc.target, "VERILATOR") if desc.target else None
    return cfg, env_name


def verilator_sim(config: str | None = None, *, board: str) -> Path | None:
    """Path to the chipyard Verilator sim for ``board`` (or an explicit ``config``), or None when it is
    not built.

    Resolution order: the board target's ``MERLIN_<TARGET>_VERILATOR`` (an explicit binary), then the
    standard chipyard build location ``$MERLIN_CHIPYARD/sims/verilator/simulator-chipyard.harness-<cfg>``,
    where ``cfg`` is ``config`` or else the board's declared ``rtl_sim_config``. Returns None rather than
    raising so callers record an honest ``not_run`` -- also when the registry declares neither fact.
    """
    from merlin.common.paths import env as _env

    cfg, env_name = _verilator_facts(board, config)
    explicit = _env(env_name) if env_name else None
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    cy = _env("MERLIN_CHIPYARD")
    if not cy or not cfg:
        return None
    sim = Path(cy) / "sims" / "verilator" / f"simulator-chipyard.harness-{cfg}"
    return sim if sim.is_file() else None


def run_on_verilator(
    elf: str | Path,
    *,
    board: str,
    config: str | None = None,
    timeout: int = 7200,
    references: dict | None = None,
    reference: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run a Zephyr ELF built for ``board`` on the chipyard Verilator sim of that board's SoC.

    The Zephyr chipyard board's console is HTIF (``zephyr,console = &htif``), which is what
    the Rocket-chip emulator harness speaks, so the ELF runs unmodified and the SAME
    OUT/ARGMAX/METRIC/DONE parser is reused. The SoC config is ``config``, else the board's
    declared ``rtl_sim_config`` (see :func:`verilator_sim`).

    SCOPE — this is cycle-accurate RTL, i.e. roughly 10^4 simulated cycles/second. A whole
    22-layer TinyLlama inference is ~10^10 cycles and is NOT feasible here; that is a
    property of RTL simulation, not a configuration to tune. Use this to certify the
    multicore-RVV MECHANISM (a GEMM micro-benchmark, or a 1-2 layer slice) and take
    whole-model functional truth from spike and whole-model cycle truth from the K1 board.
    """
    cfg, env_name = _verilator_facts(board, config)
    sim = verilator_sim(config, board=board)
    if sim is None:
        how = [f"set {env_name}"] if env_name else []
        how.append(
            f"build it: make -C $MERLIN_CHIPYARD/sims/verilator CONFIG={cfg}"
            if cfg
            else f"declare `rtl_sim_config` for {board!r} in the board registry, or pass config="
        )
        raise ZephyrModelError(f"no Verilator sim for board {board!r} (config {cfg!r}); " + ", or ".join(how))
    proc = subprocess.run([str(sim), str(elf)], capture_output=True, text=True, timeout=timeout)
    res = _parse_console(proc.stdout + proc.stderr, proc.returncode)
    res["sim"] = str(sim)
    res["sim_config"] = cfg
    refs = references if references is not None else reference
    if refs is not None:
        res.update(_gate(res["prefix"], refs))
    return res


def _parse_console(console: str, rc: int) -> dict[str, Any]:
    out_line = next((l for l in console.splitlines() if l.startswith("OUT ")), None)
    if out_line is None or "DONE" not in console:
        raise ZephyrModelError(f"run did not produce OUT/DONE (rc={rc}):\n{console[-2000:]}")
    parts = out_line.split()
    n = int(parts[1])
    bits = [int(x) for x in parts[2 : 2 + n]]
    flat = np.array([struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0] for b in bits], dtype=np.float32)
    metrics, argmax, sumval = {}, None, None
    out_hash: dict[str, Any] | None = None
    iter_cycles: list[int] = []
    iter_wall_ns: list[int] = []
    stage_wall_ns: dict[str, list[int]] = {}
    for l in console.splitlines():
        if l.startswith("METRIC "):
            parts = l.split()
            if parts[1] == "iter_cycles":  # METRIC iter_cycles <i> <cycles>
                iter_cycles.append(int(parts[3]))
            elif parts[1] == "iter_wall_ns":  # K1: METRIC iter_wall_ns <i> <ns>
                iter_wall_ns.append(int(parts[3]))
            else:
                # Most metrics are counters, but not all: `build_hash` is a hex identity string. A
                # parser that assumes int() crashes on the first non-numeric metric and takes the whole
                # run with it, so keep the raw string when it is not a number.
                #
                # And some carry a LIST: `METRIC vector_harts 0 1 2` names every vector-capable hart.
                # Reading only the first token reported it as `0`, which says "zero vector harts" --
                # precisely the opposite of what a chip with three of them was reporting.
                if len(parts) > 3:
                    metrics[parts[1]] = " ".join(parts[2:])
                else:
                    try:
                        metrics[parts[1]] = int(parts[2])
                    except ValueError:
                        metrics[parts[1]] = parts[2]
        elif l.startswith("ARGMAX "):
            p = l.split()
            argmax = np.array([int(x) for x in p[2 : 2 + int(p[1])]], dtype=np.int64)
        elif l.startswith("SUM "):
            sumval = struct.unpack("<f", struct.pack("<I", int(l.split()[1]) & 0xFFFFFFFF))[0]
        elif l.startswith("HASH "):
            # `HASH <algo> <elems> <value>` -- a whole-tensor digest. `outputs` above is only the
            # first MERLIN_DUMP_CAP elements (0.2% of a 2,048,000-element logit tensor), so this is
            # the only thing in the protocol that speaks for the rest of the answer. The algorithm
            # is named rather than assumed, so a future image can change it without silently
            # comparing two different digests.
            p_ = l.split()
            if len(p_) >= 4:
                out_hash = {"algo": p_[1], "elems": int(p_[2]), "value": int(p_[3]) & 0xFFFFFFFF}
        elif l.startswith("STAGE "):
            # K1 multi-program protocol: STAGE <session-repeat> <program> <wall-ns>.
            # Keep the repeat index structural: missing/duplicate rows are evidence of an invalid
            # diagnostic series, not values that should be silently packed into a shorter list.
            p_ = l.split()
            if len(p_) == 4:
                repeat, name, value = int(p_[1]), p_[2], int(p_[3])
                values = stage_wall_ns.setdefault(name, [])
                if repeat != len(values):
                    raise ZephyrModelError(
                        f"non-contiguous STAGE series for {name}: expected {len(values)}, got {repeat}"
                    )
                values.append(value)
    res = {"outputs": flat, "prefix": flat, "argmax": argmax, "sum": sumval, "metrics": metrics, "console": console}
    if out_hash is not None:
        res["out_hash"] = out_hash
    # `PROF <id> <ticks> <hits>` from an op_profile build. The K1 path already reads these; this one
    # dropped them on the floor, so a Zephyr/FireSim console carried per-op cycles that no caller could
    # reach -- which is why the vector unit had no measured rate while the matrix unit had one.
    from merlin.compiler.lower.model import profile as _op_profile  # `op_profile` is a parameter name here

    prof = _op_profile.parse_prof_lines(console)
    if prof:
        res["op_prof"] = prof
    if iter_cycles:
        res["iter_cycles"] = iter_cycles
        res["sustained"] = _sustained_stats(iter_cycles)
    if iter_wall_ns:
        res["iter_wall_ns"] = iter_wall_ns
        res["sustained_wall_ns"] = _sustained_stats(iter_wall_ns)
    if stage_wall_ns:
        res["stage_wall_ns"] = stage_wall_ns
    return res


# Per-element relative-error ceiling for the strict tiers. The aggregate cos and global-max-
# normalized ``rel`` terms can BOTH stay near-perfect while a single output element is
# catastrophically wrong: a measured fp16-accumulate GEMM passed the old gate at
# cos=0.9999986 while being 1209% wrong (per-element rel ≈ 12.09) on one element, because the
# blow-up was localized and drowned by the aggregates. Folding a per-element ceiling into each
# conjunction closes that hole. Tunable via MERLIN_GATE_MAX_REL (default 0.05 rejects the
# 1209% case while clearing real recaptures, whose genuine per-element spread runs ~3e-3 —
# e.g. bitvla's fp32 baseline). 0 (or empty) DISABLES the per-element term entirely.
_GATE_MAX_REL = float(os.environ.get("MERLIN_GATE_MAX_REL", "0.05") or "0")

# Localized-error veto for the COSINE-ONLY tier (t3), expressed against the run's own quantization
# floor rather than a constant. Measured across the tracked recaptures, the deviation of a CORRECT
# host int8 reference from the fp32 golden spans 0.027 (small_llama) to 1.88 (gemma2_2b) of the
# output RMS, and 1.3 to 99.0 in per-element relative terms. So NO fixed bound -- relative or
# absolute -- is satisfiable by correct implementations across models: the 0.05 per-element ceiling
# above rejects every one of them, which is why t3 carries no per-element term at all today.
# What IS comparable is the run against the SAME MODEL's own quantization noise: golden_w8a8 is the
# host's int8 result, so max|golden_w8a8 - golden_fp32| is the deviation that correct W8A8 arithmetic
# already costs on this exact output. A conformant accelerator computes the same quantized math in a
# different (equally valid) rounding/accumulation order, so it should land within a small multiple of
# that floor; a localized blow-up lands far outside it. The bound is therefore DERIVED per run and
# needs no per-model tuning. Requires BOTH references (the floor is unmeasurable otherwise) and a
# non-degenerate floor; 0 (or empty) disables it.
_GATE_QUANT_EXCESS = float(os.environ.get("MERLIN_GATE_QUANT_EXCESS", "4.0") or "0")

# HOW MUCH OF THE OUTPUT THE VERDICT ACTUALLY COVERS. The run's output reaches this function as a
# CONSOLE PREFIX -- the bare-metal/Linux harness dumps at most ``MERLIN_DUMP_CAP`` (4096) elements
# (`merlin.runtime.backends.linux.harness_source.main_linux_c`) -- while the reference is the whole tensor, and the loop below
# truncates the reference to the prefix's length. For every model whose output fits under the cap
# that is the whole tensor and the distinction is empty; for the ones that do not it is not:
# tiny_llama compares 4096 of 256000 (1.6%), gemma2_2b 4096 of 2048000 (0.2%) or of 32768000
# (0.01%), deepjscc 4096 of 12288 (33%). MEASURED, and it is not a cosmetic difference -- on
# tiny_llama int8 the SAME output scores w8a8_cos 0.892504 on the prefix and 0.995999 on the whole
# tensor, because the prefix is exactly token 0's logits and token 0 is the position W8A8
# quantization destroys for everyone (the INDEPENDENT torchao reference scores cos -0.320 against
# fp32 there, while tokens 1-7 all sit above 0.988). A caller handed 0.892504 with nothing saying
# "of 1.6% of the output, all of it the worst position" reads it as the model's accuracy. That
# reading has already cost this repo one multi-hour hunt for a board defect that did not exist
# (`prefix 0.484 vs full 0.976`), so the coverage is now REPORTED on every verdict.
#
# ``MERLIN_GATE_MIN_COVERAGE`` turns the report into a veto: a fraction in (0, 1] below which no
# tier may be awarded, so a caller that needs a whole-output verdict declares it and gets a refusal
# instead of a prefix score. Default 0 = report only, because the existing board harness CANNOT
# produce more than the cap and flipping every large-output run to NOT_GATED is a measurement-policy
# decision, not a bug fix.
_GATE_MIN_COVERAGE = float(os.environ.get("MERLIN_GATE_MIN_COVERAGE", "0") or "0")


def _gate(prefix: np.ndarray, references, *, max_rel: float | None = None, min_coverage: float | None = None) -> dict:
    """Multi-tier accuracy gate for an int8 (W8A8) run. ``references`` is either a single
    fp32 reference array (legacy: one ``cos``/``rel``/``ok`` at the strict fp32 threshold) or
    a ``{tier: array}`` dict. Tiers (literature-backed for W8A8 transformers — I-BERT /
    SmoothQuant put W8A8 at cos 0.99–0.999 vs fp32):
      * ``w8a8`` — vs a torch/host W8A8 reference: T1 cos > 0.999 AND rel < 1e-2 AND per-element
        max-rel < ``max_rel`` (the int8 datapath is faithful to the W8A8 math AND no single
        element blew up);
      * ``fp32`` — vs the fp32 golden: T2 cos > 0.99 AND top-1 argmax matches AND per-element
        max-rel < ``max_rel`` (quant degradation within band, no localized blow-up).
    ``ok = T1 or T2``. The per-element term is ``perel = |pref - r| / (|r| + floor)`` with the
    denominator floored at 0.1% of ``|r|.max()`` so genuine near-zero reference elements don't
    manufacture spurious blow-ups, while a real localized error (large element, or a near-zero
    element that jumped) still trips it. ``max_rel`` defaults to the ``MERLIN_GATE_MAX_REL``
    module knob; a value <= 0 disables the per-element term (recovering the old aggregate-only
    behavior). Emits per-tier ``<tier>_cos``/``<tier>_rel``/``<tier>_argmax``/``<tier>_max_rel``.

    ``prefix`` is compared against the LEADING elements of each reference, so a run whose output the
    harness console truncated is scored on that truncation. Every verdict therefore also carries
    ``n_compared`` / ``n_reference`` / ``compared_fraction`` / ``comparison_complete``, and
    ``min_coverage`` (default: the ``MERLIN_GATE_MIN_COVERAGE`` knob) refuses every tier below that
    fraction rather than returning a partial-output score as if it were the model's verdict. See
    ``_GATE_MIN_COVERAGE``."""
    pref = np.asarray(prefix, dtype=np.float32).ravel()
    if not isinstance(references, dict):
        references = {"fp32": references}
    thresh = _GATE_MAX_REL if max_rel is None else float(max_rel)
    cover_min = _GATE_MIN_COVERAGE if min_coverage is None else float(min_coverage)
    out: dict[str, Any] = {}
    _arr: dict[str, np.ndarray] = {}
    _ref_n: dict[str, int] = {}
    k = len(pref)
    for tier, ref in references.items():
        if ref is None:
            continue
        _full = np.asarray(ref, dtype=np.float32).ravel()
        _ref_n[tier] = int(_full.size)
        r = _full[:k]
        _arr[tier] = r
        rmax = max(1e-9, float(np.abs(r).max()))
        rel = float(np.abs(pref - r).max()) / rmax
        cos = float((pref @ r) / (np.linalg.norm(pref) * np.linalg.norm(r) + 1e-12))
        # per-element relative error over SIGNIFICANT reference elements (|r_i| >= 1% of the RMS
        # scale). Masking by the TYPICAL scale (RMS), not the max, is what makes this robust on both
        # spiky classification logits AND whole-model regression outputs: a near-zero output element
        # with a large relative but negligible ABSOLUTE error (invisible to cos — measured: bitvla
        # raw max-rel 1.05 at cos 0.9999945) is below the RMS floor and does not trip the gate, while
        # a real error on a meaningful element (the fp16-accumulate 1209% disaster; the blow-up
        # constructions) sits above it and is caught. rmax-based flooring failed here because a spiky
        # output makes 1e-3*rmax too small to suppress the near-zero element.
        rms = max(1e-9, float(np.sqrt(np.mean(r.astype(np.float64) ** 2))))
        _sig = np.abs(r) >= 1e-2 * rms
        perel = float((np.abs(pref[_sig] - r[_sig]) / np.abs(r[_sig])).max()) if _sig.any() else 0.0
        out[f"{tier}_cos"] = cos
        out[f"{tier}_rel"] = rel
        out[f"{tier}_argmax"] = bool(int(np.argmax(pref)) == int(np.argmax(r)))
        out[f"{tier}_max_rel"] = perel

    # ---- derived localized-error veto (see _GATE_QUANT_EXCESS): judge the run against the
    # quantization noise THIS model's own references already exhibit, not against a constant.
    _excess_ok, _excess_applied = True, False
    _f, _q = _arr.get("fp32"), _arr.get("w8a8")
    if _GATE_QUANT_EXCESS > 0 and _f is not None and _q is not None:
        _floor = float(np.abs(_q.astype(np.float64) - _f.astype(np.float64)).max())
        _scale = max(1e-12, float(np.abs(_f.astype(np.float64)).max()))
        if _floor > 1e-9 * _scale:  # degenerate (identical references) => unmeasurable, veto stays off
            _run = float(np.abs(pref.astype(np.float64)[: len(_f)] - _f.astype(np.float64)).max())
            out["quant_floor_abs"] = _floor
            out["run_dev_abs"] = _run
            out["quant_excess"] = _run / _floor
            _excess_applied = True
            _excess_ok = out["quant_excess"] <= _GATE_QUANT_EXCESS

    def _perel_ok(tier: str, bound: float | None = None) -> bool:
        # bound <= 0 disables the term; missing tier => vacuously ok (that tier isn't in play).
        b = thresh if bound is None else bound
        return b <= 0 or out.get(f"{tier}_max_rel", 0.0) < b

    t1 = out.get("w8a8_cos", 0.0) > 0.999 and out.get("w8a8_rel", 1.0) < 1e-2 and _perel_ok("w8a8")
    t2 = out.get("fp32_cos", 0.0) > 0.99 and out.get("fp32_argmax", False) and _perel_ok("fp32")
    # legacy single-reference callers gate at the strict fp32 threshold (+ per-element ceiling)
    legacy = (
        "w8a8" not in out
        and out.get("fp32_cos", 0.0) > 0.9999
        and out.get("fp32_rel", 1.0) < 1e-3
        and _perel_ok("fp32")
    )
    # T3 fp32 REGRESSION tier: whole-model regression outputs (VLA policies) are cosine-tight but have
    # NO meaningful top-1 argmax and a global rel ~3e-3 (> the legacy 1e-3 bit-close bar), so t2/legacy
    # reject them despite cos ~0.9999945. This tier is the accepted
    # whole-model gate: cos >= 0.9999, COSINE-ONLY. Evidence forces this: a whole-model regression
    # output legitimately carries high PER-ELEMENT relative error on its many small (low-absolute)
    # elements -- bitvla's real seed measured per-element max-rel 1.1 (even RMS-masked) at cos
    # 0.9999945 / global rel 0.0031, i.e. numerically correct but per-element-noisy by nature. No
    # per-element RELATIVE bound can pass such an output, which is why the four-way authority (that has
    # always accepted these results) gates on cosine alone. A per-element ABSOLUTE bound CAN gate one,
    # and t3 now carries it whenever it is measurable -- see _GATE_QUANT_EXCESS. That veto is what
    # closes the localized-blow-up hole at this tier; without it a 1209%-style single-element error
    # reaching t3 passed untouched, since cosine is exactly what such an error hides from.
    # The per-element RELATIVE veto is retained where outputs are well-scaled and it is meaningful: the int8 (t1/w8a8) and classification
    # (t2/argmax) and bit-close (legacy) tiers, plus the fp16 driver's own gate. Whole-model fp16/int8
    # forks are still per-element-guarded via t1 (w8a8 ref) and the dtype drivers.
    t3 = out.get("fp32_cos", 0.0) > 0.9999 and "fp32_cos" in out and _excess_ok
    # ---- HOW MUCH OF THE OUTPUT THIS VERDICT COVERS (see `_GATE_MIN_COVERAGE`). Reported on every
    # verdict, because the truncation above is invisible in cos/rel and a partial score reads
    # exactly like a whole-output one. `n_reference` is the LARGEST reference offered: a tier whose
    # reference is itself short would otherwise make a truncated run look complete.
    _n_ref = max(_ref_n.values()) if _ref_n else 0
    _n_cmp = min(k, _n_ref) if _n_ref else k
    out["n_compared"] = int(_n_cmp)
    out["n_reference"] = int(_n_ref)
    out["compared_fraction"] = (float(_n_cmp) / _n_ref) if _n_ref else 0.0
    out["comparison_complete"] = bool(_n_ref) and _n_cmp >= _n_ref
    # A declared coverage floor VETOES every tier: below it there is no output-level verdict to give,
    # only a score on a slice the caller did not ask about.
    _coverage_ok = cover_min <= 0 or out["compared_fraction"] >= cover_min
    out["coverage_ok"] = bool(_coverage_ok)
    out["min_coverage"] = float(cover_min)
    out["cos"] = out.get("w8a8_cos", out.get("fp32_cos"))
    out["rel"] = out.get("w8a8_rel", out.get("fp32_rel"))
    out["max_rel"] = out.get("w8a8_max_rel", out.get("fp32_max_rel"))
    if not _coverage_ok:
        t1 = t2 = legacy = t3 = False
    out["ok"] = bool(t1 or t2 or legacy or t3)
    # Which tiers were actually in play, and which one carried the verdict. Without this the
    # fallback is INVISIBLE: grading a W8A8 run with no `golden_w8a8.npy` silently drops to the
    # fp32 tier, and the ordinary weight-only-vs-W8A8 gap then reads as a codegen defect. That
    # exact confusion cost a multi-hour hunt for a TinyLlama int8 "board bug" that did not exist
    # (the board matched the host W8A8 reference at rel 0.0). Callers that know the run is int8
    # should treat a missing `w8a8` tier as an incomplete measurement, not a failure.
    out["tiers"] = sorted(t for t in ("w8a8", "fp32") if f"{t}_cos" in out)
    out["tier_ok"] = "w8a8" if t1 else "fp32" if t2 else "fp32_legacy" if legacy else "fp32_cos_only" if t3 else None
    # DID A PER-ELEMENT VETO ACTUALLY APPLY? t3 is cosine-only only WHEN THE DERIVED VETO CANNOT BE
    # MEASURED (one reference, or two identical ones) -- a whole-model regression
    # output legitimately carries high per-element relative error on its many small elements, and no
    # per-element bound can pass one. But t3 fires for ANY output whose cosine clears 0.9999, including
    # one where the veto WOULD have been meaningful and WOULD have failed: a measured whole model passed
    # here at cos 0.99993 with per-element max-rel 2.82 and global rel 0.0148, having failed both t1 and
    # t2 on exactly those numbers. That is a real pass and also a much weaker one, and a caller quoting
    # `ok` alone cannot tell the two apart. Stated as a field so a certification can carry its own
    # strength instead of the reader having to reverse-engineer it from a tier name.
    # When both references are present the run IS per-element guarded -- by the quantization-excess
    # bound, the veto actually applicable to this output class (the relative one rejects correct
    # reference implementations, so its absence here was never evidence of a defect).
    out["per_element_guarded"] = bool(t1 or t2 or legacy or (t3 and _excess_applied))
    out["per_element_basis"] = (
        "relative" if (t1 or t2 or legacy) else "quantization_excess" if (t3 and _excess_applied) else None
    )
    return out


def run_on_firesim(
    elf: str | Path,
    *,
    reference: np.ndarray | None = None,
    references: dict | None = None,
    mode: str | None = None,
    output: str | Path | None = None,
    deployment=None,
    inputs: dict[str, Path] | None = None,
    results: str | Path | None = None,
    timeout: int | None = None,
    queue: bool | None = None,
    firesim_root: str | None = None,
    firesim_env: str | None = None,
    runner: FireSimRunner | None = None,
    runner_name: str | None = None,
) -> dict[str, Any]:
    """Prepare an ELF or collect external UART through the historical model API.

    ``mode="prepare"`` requires output and an explicit Deployment. No numerical
    verdict is produced. ``mode="collect"`` interprets the positional path as a
    prepared package and requires results; original UART parsing and numerical
    gates apply. Legacy execution keywords refuse before loading any provider.
    """

    if mode not in {"prepare", "collect"} or any(
        value is not None for value in (timeout, queue, firesim_root, firesim_env, runner, runner_name)
    ):
        raise FireSimRunnerError(
            "FireSim launch is externally managed; use mode='prepare' with output/deployment, "
            "then mode='collect' with the prepared package and results; remove launch options"
        )
    if mode == "prepare":
        if output is None or deployment is None or results is not None:
            raise FireSimRunnerError("prepare requires output and deployment, without results")
        if reference is not None or references is not None:
            raise FireSimRunnerError("prepare cannot gate output; retain reference files through inputs")
        prepared = prepare_image(elf, output=output, deployment=deployment, inputs=inputs)
        return {
            "status": "prepared",
            "prepared": str(prepared.directory),
            "request_sha256": prepared.request_sha256,
            "execution": "external_operator_required",
        }
    if results is None or any(value is not None for value in (output, deployment, inputs)):
        raise FireSimRunnerError("collect requires results and cannot change the prepared deployment or inputs")
    if reference is not None and references is not None:
        raise FireSimRunnerError("pass reference or references, not both")
    uart, observed = collect_uart(elf, results=results)
    res = _parse_console(uart, observed.reported_exit_code)
    refs = references if references is not None else reference
    if refs is not None:
        res.update(_gate(res["prefix"], refs))
    res.update(
        status="collected",
        request_sha256=observed.request_sha256,
        observation=str(observed.directory),
        execution_authority="external_operator_report",
    )
    return res


def build_and_run(
    model_dir: str | Path,
    work: str | Path,
    *,
    board: str | None = None,
    backend: str = "rvv",
    rvv_hart: int = 0,
    harts: int = 2,
    reference: np.ndarray | None = None,
    references: dict | None = None,
    timeout: int = 3600,
    int8_compute: bool = False,
    quant_passes: list[str] | None = None,
    n_harts: int = 1,
    iters: int = 1,
    warmup: int = 0,
    omp_threads: int | None = None,
    rvv_schedule: str | None = None,
    cflags_override: list[str] | None = None,
    features: frozenset[str] | None = None,
    vlen: int | None = None,
) -> dict[str, Any]:
    """Build the Zephyr image and (for spike) run + gate. ``references`` (a ``{tier: array}``
    dict, e.g. ``{"w8a8": ..., "fp32": ...}``) drives the multi-tier int8 gate; a single
    ``reference`` array keeps the legacy strict fp32 gate.

    ``n_harts``/``iters``/``warmup`` select the multicore and sustained-inference images;
    ``rvv_schedule``/``cflags_override``/``features`` pass an RVV package's codegen through
    (the same seam :mod:`merlin.analysis.mining.package_application` uses) so a package can be measured here.
    """
    # spike needs at least as many harts as the image will use, or the pinned workers have
    # nowhere to run and the pool silently clamps to fewer threads.
    harts = max(harts, n_harts, rvv_hart + 1)
    b = build_app(
        model_dir,
        work,
        board=board,
        backend=backend,
        rvv_hart=rvv_hart,
        cpus=max(harts, rvv_hart + 1),
        int8_compute=int8_compute,
        quant_passes=quant_passes,
        n_harts=n_harts,
        iters=iters,
        warmup=warmup,
        omp_threads=omp_threads,
        rvv_schedule=rvv_schedule,
        cflags_override=cflags_override,
        features=features,
        vlen=vlen,
    )
    result: dict[str, Any] = {
        "elf": str(b["elf"]),
        "app_dir": str(b["app_dir"]),
        "ram_bytes": b["ram_bytes"],
        "n_harts": n_harts,
        "iters": iters,
        # WHICH image produced this run. A caller that gates here and ships a
        # differently-configured build (a board whose console spike cannot
        # service) needs to be able to say so, rather than leave the recipient
        # to discover that the console beside their ELF names another binary.
        "build_hash": b.get("build_hash", ""),
        "quantization_policy": b.get("quantization_policy"),
        # the vector length the build AND the run agreed on (None = spike's
        # default 128); recorded so a result cannot be read as another VLEN's
        "vlen": vlen,
    }
    from merlin.runtime.boards.catalog import board as _board_desc

    if board is None or _board_desc(board).simulator != "spike":
        return result  # FireSim path runs the elf separately (firesim_runner / queue)
    run = run_on_spike(
        b["elf"],
        dram_base=_board_desc(board).dram_base,
        harts=harts,
        mem_bytes=b["ram_bytes"],
        timeout=timeout,
        vlen=vlen,
    )
    result.update(run)
    refs = references if references is not None else reference
    if refs is not None:
        result.update(_gate(run["prefix"], refs))
    return result
