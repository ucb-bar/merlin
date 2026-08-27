"""Does this backend EMIT A PROGRAM for a shape, and if not, does it say so?

A structural probe, deliberately not a numerical one. The numerical generalization difftest can only run
where a CPU reference exists for the operand format, which excludes every MX/fp8 datapath -- i.e. exactly
the targets whose shape coverage is most in doubt. Measured while building this: all four multi-tile
probes on an fp8 target were skipped for want of a golden, and the suite reported ``0 graded``, which
reads as "nothing to report" rather than "could not look".

But shape COVERAGE needs no golden. "Did you emit a program for 2 M-tiles" is answerable from the
command buffer alone, on any dtype, at no oracle cost. That is also the precise question a shape-keyed
backend fails: one submission chained twelve builders each guarded on a literal public-capsule shape and
fell through to a bare terminator, so a declined shape produced an output of zeros that the grader could
only describe as wrong arithmetic.

Three outcomes, and the third is the bug this exists to name:

``lowered``   the backend emitted commands for this shape.
``declined``  the backend said it does not handle this shape (see ``BackendDeclined``). Honest.
``empty``     the backend emitted a program with no commands and said nothing -- indistinguishable, at
              the numeric tier, from arithmetic that ran and was wrong.

Target-agnostic: the extents come from the target's DERIVED tile edge and the dtypes from its own corpus
binding. Nothing here knows which accelerator it is looking at.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from .capability_probes import tile_edge
from .capsule_common import make_run_paths, run_entrypoints
from .oot_runner import BackendDeclined, CertFailure

#: The corners a coverage sweep asks for, as multiples of the derived tile edge on (M, K, N).
#: ``tile`` is the BASELINE and is load-bearing: a multi-tile verdict means nothing unless the single
#: tile lowers, because whatever breaks at one tile breaks at two and would be misattributed to shape.
CORNERS: dict[str, tuple[int, int, int]] = {
    "tile": (1, 1, 1),
    "m_2tiles": (2, 1, 1),
    "k_2tiles": (1, 2, 1),
    "n_2tiles": (1, 1, 2),
}


@dataclass(frozen=True)
class CoverageResult:
    corner: str
    shape: tuple[int, int, int]
    outcome: str                  # lowered | declined | collapsed | empty | error
    detail: str | None = None
    work: int = 0                 # size of the emitted target artifact (see `sweep`)

    def to_dict(self) -> dict:
        d = {"corner": self.corner, "shape": list(self.shape), "outcome": self.outcome,
             "emitted_work": self.work}
        if self.detail:
            d["detail"] = self.detail
        return d


def contraction_interface(m: int, k: int, n: int, *, target: str, operand_mlir: str,
                          accum_mlir: str) -> str:
    """The single-contraction interface module, at the extents asked for.

    Byte-identical in structure to what the corpus generator emits for a contraction capsule, so a
    backend that lowers its own corpus lowers this too -- the ONLY difference is the extents, which is
    what makes the probe a clean shape experiment rather than a new op.
    """
    return (f'module attributes {{merlin_iface.version = "0.1", merlin_iface.target = "{target}", '
            f'merlin_iface.abi_version = "0.1"}} {{\n'
            f'  %W = merlin_iface.tensor {{name = "W", role = "weight"}} : tensor<{k}x{n}x{operand_mlir}>\n'
            f'  %A0 = merlin_iface.tensor {{name = "A0", role = "input"}} : tensor<{m}x{k}x{operand_mlir}>\n'
            f'  %W_res = merlin_iface.resident_pack %W {{layout = "packed_rhs"}} : '
            f'(tensor<{k}x{n}x{operand_mlir}>) -> !merlin_iface.resident\n'
            f'  %acc0 = merlin_iface.matmul %A0, %W_res : (tensor<{m}x{k}x{operand_mlir}>, '
            f'!merlin_iface.resident) -> !merlin_iface.acc<{accum_mlir}>\n'
            f'  %Y0 = merlin_iface.commit %acc0 {{name = "Y0", epilogue = [], '
            f'output_dtype = "{accum_mlir}"}} : (!merlin_iface.acc<{accum_mlir}>) -> '
            f'tensor<{m}x{n}x{accum_mlir}>\n'
            f'  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()\n}}\n')


def _binding(target: str):
    """The target's own corpus binding — the operand/accum dtypes its capsules are actually written in.

    Resolved through :func:`corpora.experiment_for` -- the one module allowed to know the per-target
    descriptor layout, and the only resolver that honours ``MERLIN_TARGET_EXPERIMENT``.
    """
    from .corpora import experiment_for
    from .corpus_spec import derive_binding
    te = experiment_for(target)
    if te is None:
        raise ValueError(f"no experiment descriptor for target {target!r}: cannot derive its dtypes")
    return derive_binding(te, {})


def probe_shape(package: str | Path, *, target: str, m: int, k: int, n: int,
                operand_mlir: str, accum_mlir: str, contract: str | Path | None = None,
                timeout: int = 300) -> tuple[str, str | None, int]:
    """Run ONLY the emit half of the contract for one shape; classify what came back.

    No oracle, no golden, no simulator -- the four entrypoints and the artifact they produce. That is
    what makes this cheap enough to run every round and usable on a datapath whose operand format has
    no CPU reference.

    Returns ``(outcome, detail, work)`` where ``work`` is the size of the emitted target artifact. The
    caller compares that across shapes; see :func:`sweep` for why the comparison is the real instrument.
    """
    with tempfile.TemporaryDirectory(prefix="lowcov_") as td:
        cdir = Path(td) / "capsule"
        cdir.mkdir()
        (cdir / "capsule.interface.mlir").write_text(
            contraction_interface(m, k, n, target=target, operand_mlir=operand_mlir,
                                  accum_mlir=accum_mlir), encoding="utf-8")
        capsule = {"name": f"lowcov_{m}x{k}x{n}", "__dir__": str(cdir),
                   "interface_mlir": "capsule.interface.mlir"}
        paths = make_run_paths(Path(td) / "runs", capsule["name"], suite=f"{target}-lowering-coverage",
                               target=target, dtype=operand_mlir, benchmark="matmul")
        try:
            _pkg, cb, _art = run_entrypoints(None, package, capsule, paths, contract=contract,
                                             timeout=timeout, fourth_output_name="lowered.llvm.mlir")
        except BackendDeclined as bd:
            return "declined", bd.reason, 0
        except CertFailure as cf:
            return "error", f"{cf.plane}: {cf.detail[:200]}", 0
        except Exception as e:  # noqa: BLE001 -- a harness fault must not read as a coverage verdict
            return "error", f"{type(e).__name__}: {str(e)[:200]}", 0
        # EMITTED NOTHING AND SAID NOTHING. The schema now forbids this (an empty command list requires
        # a stated `declined`), so reaching it means a backend produced a terminator-only program some
        # other way -- still a silent refusal, and still named as one rather than scored as bad math.
        if not (cb.get("commands") or []):
            return "empty", ("emitted a program with no commands and did not declare `declined` -- a "
                             "silent refusal, which at the numeric tier is indistinguishable from "
                             "arithmetic that ran and was wrong"), 0
        # THE COMMAND BUFFER IS NOT THE PROGRAM on every endpoint. A self-hosted-ISA backend emits its
        # kernel as the FOURTH artifact and can build a perfectly well-formed command buffer beside a
        # kernel that is a bare terminator -- measured: identical 4-command buffers for a shape it
        # lowered in 418 instruction words and one it "lowered" in 5. So the work is counted from the
        # artifact, not the buffer.
        return "lowered", None, len([ln for ln in (_art or "").splitlines() if ln.strip()])


def sweep(package: str | Path, *, target: str, contract: str | Path | None = None,
          corners: dict[str, tuple[int, int, int]] | None = None,
          timeout: int = 300) -> dict:
    """Probe every corner and summarise, PER AXIS.

    Per axis because a lowering that loops over K and N but not M covers two of three, and only naming
    the axis turns the result into an instruction ("add a loop over M") rather than a grade ("does not
    generalize").
    """
    b = _binding(target)
    tile = tile_edge(target)
    operand_mlir, accum_mlir = b.mlir_dtype(b.operand_dtype), b.mlir_dtype(b.accum_dtype)
    corners = corners or CORNERS
    results: list[CoverageResult] = []
    work: dict[str, int] = {}
    for name, (fm, fk, fn) in corners.items():
        m, k, n = tile * fm, tile * fk, tile * fn
        outcome, detail, w = probe_shape(package, target=target, m=m, k=k, n=n,
                                         operand_mlir=operand_mlir, accum_mlir=accum_mlir,
                                         contract=contract, timeout=timeout)
        work[name] = w
        results.append(CoverageResult(name, (m, k, n), outcome, detail, w))

    # WORK MUST NOT SHRINK WHEN THE PROBLEM GROWS. This is the instrument, and it is target-agnostic:
    # it needs no ISA knowledge, no golden and no simulator, only the observation that a program which
    # computes a 2x larger contraction cannot be SMALLER than the one that computes the 1x case. A
    # backend that silently declines emits its terminator and nothing else, so the artifact collapses.
    #
    # Measured on a real submission, from the emitted artifact alone: 418 instruction words at one tile,
    # 5 at two M-tiles (a bare ECALL), 1187 at two K-tiles and 1205 at two N-tiles. That is the same
    # M-versus-K/N boundary a post-freeze holdout took a paid run to find, reproduced here for the cost
    # of four calls to the emit path -- and it says WHICH axis, which is the actionable part.
    #
    # Necessary, not sufficient: a program that grows may still compute the wrong thing. This probe is
    # about COVERAGE (did you write code for this shape), and the numeric tiers keep their own job.
    base_work = work.get("tile", 0)
    for i, r in enumerate(results):
        if r.corner != "tile" and r.outcome == "lowered" and base_work and r.work < base_work:
            results[i] = CoverageResult(
                r.corner, r.shape, "collapsed",
                (f"emitted {r.work} instruction word(s) for a problem {r.shape} that is LARGER than the "
                 f"{tile}x{tile}x{tile} baseline, which took {base_work}. A program cannot compute more "
                 f"by doing less -- this is a silent refusal. Declare `declined` instead, or lower it."),
                r.work)

    by = {r.corner: r.outcome for r in results}
    baseline_ok = by.get("tile") == "lowered"
    out = {
        "target": target, "package": str(package), "tile_edge": tile,
        "operand_dtype": b.operand_dtype, "accum_dtype": b.accum_dtype,
        "corners": [r.to_dict() for r in results],
        "baseline_tile_lowered": baseline_ok,
        "n_declined": sum(1 for r in results if r.outcome == "declined"),
        "n_empty": sum(1 for r in results if r.outcome == "empty"),
        "n_collapsed": sum(1 for r in results if r.outcome == "collapsed"),
        "emitted_work": work,
    }
    if baseline_ok:
        out["multi_tile_axes_uncovered"] = sorted(
            {c[0] for c, o in by.items() if c.endswith("_2tiles") and o != "lowered"})
        out["all_covered"] = not out["multi_tile_axes_uncovered"]
    else:
        # Refusing to answer beats answering wrongly: with the baseline down, every multi-tile corner
        # fails for a reason that has nothing to do with shape, and reporting "M, K and N all uncovered"
        # would be a confident, specific, wrong attribution.
        out["multi_tile_axes_uncovered"] = []
        out["all_covered"] = False
        out["unmeasured"] = ("the single-tile baseline did not lower, so nothing here can be attributed "
                             "to shape generalization -- fix the baseline, then re-read this")
    return out
