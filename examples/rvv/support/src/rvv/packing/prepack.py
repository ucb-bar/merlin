"""Apply a model's weight layout ONCE, at build time, instead of on every inference.

A `linalg.transpose` reading a `@forward` argument is not computation: it is the model paying, every
single run, to convert a weight into the layout its consumer wanted all along. The packer could have
stored it that way once. :mod:`merlin.xdsl_dialects.lowering.weight_layout` finds those re-layouts
and :func:`merlin.baselines.bundle_rewrite.hoist_weight_transposes` applies them offline; what did
not exist was a way for the COMPILER to do it, so the only bundles that had it were three
hand-driven `_pretransposed` directories nobody could reproduce.

MEASURED, and it is the largest single whole-model lever this repo has found. Interleaved in ONE K1
board session, alternating bundles, three rounds each, both arms gating ok=True:

    stock          3,548,286 / 3,574,361 / 3,561,602 ns
    prepacked      2,125,388 / 2,086,712 / 2,127,671 ns   ->  1.70x  (+70.0%)

against a 2.6% K1 noise band. The saving is not the byte traffic of the transposes themselves --
those are 0.4 MiB per inference on this model, well inside the band -- it is that 15 `linalg.transpose`
ops and their `tensor.empty` destinations stop being materialized at all, taking their buffers,
their bufferization allocs and their copies with them.

THE SEAM, which is the whole trick. :func:`merlin.llvmlower.c_runtime.generate` is handed a
**bundle directory**, not the prepared IR: it re-parses `model.mlir` for the argument signature and
copies `weights.safetensors`' payload verbatim into `weights.bin`. So the compiled object follows
the PREPARED module while the ABI table and the weight blob follow the BUNDLE. Rewriting the IR
alone would leave the two describing different layouts -- the object indexing a transposed weight
that the blob stores untransposed. Handing BOTH `prepare_for_lowering` and `c_runtime.generate` the
same rewritten bundle is what keeps them consistent.

DEFAULT-OFF and BASELINE BYTE-IDENTICAL. The feature carries no pipeline, schedule or cflags hook --
it changes the INPUT bundle, not the compiler -- so a build naming no feature is byte-for-byte the
build that existed before.

FAIL CLOSED. A bundle whose layout cannot be pre-applied soundly (a stubbed weight with no bytes,
two arguments naming one tensor, an aliased byte range, a non-2-D weight) is REFUSED, not built
stock: a lever that silently declines is indistinguishable from one that did nothing, and would be
measured as if it had been applied.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

#: the compiler-feature name a package/fork lists in `compiler_features`
FEATURE = "prepack_weight_layout"

#: bump when the produced bundle's BYTES would change, so cached bundles from an older rewrite are
#: not reused under the new semantics
REWRITE_VERSION = "3"

#: files whose content defines the rewrite's input; the cache key is (name, size, mtime_ns) of each
_KEY_FILES = ("model.mlir", "weights.safetensors", "weights.safetensors.manifest.json")


class PrepackRefused(RuntimeError):
    """The weight layout of this bundle cannot be pre-applied soundly. Never downgraded to a
    warning: a build that quietly kept the stock bundle would report the lever as applied."""


@dataclass(frozen=True)
class PrepackPlan:
    """What pre-applying `bundle`'s weight layout would do, without doing it."""

    bundle: str
    hoistable: int
    blocked: int
    #: weight bytes the model re-lays-out per inference today, and would stop moving
    bytes_per_inference: int
    #: why it cannot be done; empty means it can
    problems: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.hoistable > 0 and not self.problems


def plan(src: Path | str, func_name: str = "forward") -> PrepackPlan:
    """The hoistable/blocked split for `src` plus the storage-level safety checks. Writes nothing.

    Shares its analysis and its checks with :func:`prepacked_bundle`, so a plan and a build can never
    disagree about what is hoistable or why it was refused.
    """
    from merlin.capture.rewrite import hoist_safety_problems
    from merlin.common import mlir_query as mq
    from merlin.common.ir_lock import IR_LOCK
    from merlin.xdsl_dialects.lowering.weight_layout import weight_layout_report

    src = Path(src)
    with IR_LOCK:
        # Give the parser the file path: captured MLIR can contain custom-form
        # tensor ops, for which mq.parse uses the audited generic-form fallback.
        report = weight_layout_report(mq.parse(src / "model.mlir"), func_name)

    man = json.loads((src / "weights.safetensors.manifest.json").read_text())
    weight_args = {
        int(k) for k, entry in man.items() if k.isdigit() and "weight" in entry
    }
    # `weight_layout_report` deliberately knows only IR structure, so it also reports transposed
    # runtime inputs.  At the bundle seam we can distinguish them: only manifest-declared weights
    # are candidates for an offline storage rewrite.  A non-weight transpose stays in the graph; it
    # is not a soundness failure for the weights that *can* be hoisted.
    eligible = [r for r in report.hoistable if r.arg in weight_args]
    problems: list[str] = [
        problem
        for problem in report.unpriceable
        if any(problem.startswith(f"arg {arg}:") for arg in weight_args)
    ]
    want: dict[str, int] = {}
    for r in eligible:
        entry = man.get(str(r.arg))
        assert entry is not None and "weight" in entry
        want[entry["weight"]] = r.arg
    if want:
        problems.extend(
            hoist_safety_problems(src, man, want, {r.arg for r in eligible})
        )
    return PrepackPlan(
        bundle=src.name,
        hoistable=len(eligible),
        blocked=len(report.blocked) + len(report.hoistable) - len(eligible),
        bytes_per_inference=sum(r.bytes_moved for r in eligible),
        problems=tuple(problems),
    )


def cache_key(src: Path | str) -> str:
    """Identity of (this bundle's inputs, this rewrite). Content-addressed by size + mtime rather
    than by digest: the weight blob is gigabytes on the models this matters most for, and hashing it
    on every fork of a beam search would cost more than the rewrite it guards."""
    src = Path(src).resolve()
    parts = [FEATURE, REWRITE_VERSION, str(src)]
    for name in _KEY_FILES:
        p = src / name
        st = p.stat()
        parts.append(f"{name}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def prepacked_bundle(
    src: Path | str, *, cache_root: Path | str | None = None, func_name: str = "forward"
) -> tuple[Path, dict]:
    """`(bundle_dir, effect)` for a bundle storing `src`'s weights in their consumers' layout.

    NEVER mutates `src`: the recapture tree is shared by every other session and every other
    measurement made from it. The result is a separate, cached directory keyed by
    :func:`cache_key`, published by an atomic rename so two concurrent builds cannot observe a
    half-written bundle (the loser reuses the winner's).
    """
    from merlin.capture.rewrite import (
        RewriteRefused,
        hoist_weight_transposes,
        read_rewrites,
    )

    src = Path(src).resolve()
    root = Path(cache_root) if cache_root is not None else _default_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    dst = root / f"{src.name}__{cache_key(src)}"
    if dst.is_dir():
        recs = [r for r in read_rewrites(dst) if r.name == "hoist_weight_transposes"]
        if recs:
            return dst, {"cached": True, **recs[-1].effect}
        shutil.rmtree(dst)  # a directory without its record is not a result

    p = plan(src, func_name)
    if not p.ok:
        raise PrepackRefused(
            f"{src.name}: {'; '.join(p.problems) if p.problems else 'no hoistable weight transposes'}"
        )

    tmp = root / f".tmp-{src.name}-{os.getpid()}-{cache_key(src)}"
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        rec = hoist_weight_transposes(src, tmp, func_name)
    except (
        RewriteRefused
    ) as exc:  # the checks agree with `plan`; surface it as a refusal
        shutil.rmtree(tmp, ignore_errors=True)
        raise PrepackRefused(str(exc)) from exc
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    try:
        os.replace(tmp, dst)
    except OSError:  # another build published first -- use theirs
        shutil.rmtree(tmp, ignore_errors=True)
        if not dst.is_dir():
            raise
        return dst, {"cached": True, **rec.effect}
    # The rewrite stamped `prov.weights_file` with the path it WROTE to, which was the staging
    # directory. Re-point it at the published one: provenance naming a path that no longer exists is
    # the same defect as provenance naming the source blob, one step further along.
    from merlin.capture.rewrite import retarget_weights_file

    text, retargeted = retarget_weights_file(
        (dst / "model.mlir").read_text(), dst / "weights.safetensors"
    )
    if retargeted:
        (dst / "model.mlir").write_text(text)
    return dst, {"cached": False, **rec.effect}


def _default_cache_root() -> Path:
    from merlin.common.artifacts import cache_dir

    return cache_dir("weight_prepack")


def prepare_build_bundle(
    src: Path | str, work: Path | str, features: frozenset[str] | None
) -> Path:
    """Select ONE bundle for both lowering and runtime ABI generation.

    Backends call this before reading model.mlir or selecting default inputs. No feature means no
    parsing, copying or receipt. The receipt records the deletion, not a predicted speedup.
    """
    src = Path(src).resolve()
    if FEATURE not in (features or frozenset()):
        return src
    destination, effect = prepacked_bundle(src)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    (work / "bundle_preparation.json").write_text(
        json.dumps(
            {
                "schema": "merlin.bundle-preparation.v1",
                "feature": FEATURE,
                "source_bundle": str(src),
                "prepared_bundle": str(destination),
                "effect": effect,
                "consumers": ["model_lowering", "runtime_abi_and_weights"],
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    return destination
