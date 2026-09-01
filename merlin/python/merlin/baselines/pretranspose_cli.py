"""``merlin-bundle-pretranspose`` — store weights in the layout their consumer wants.

The analysis and the rewrite both already existed and were complete
(:func:`merlin.xdsl_dialects.lowering.weight_layout.weight_layout_report` splits hoistable from
blocked and fails closed via ``.unpriceable``; :func:`..bundle_rewrite.hoist_weight_transposes`
applies it, asserts each weight's bytes equal ``stored.T``, and records a ``RewriteRecord``). What did
not exist was any way to RUN them: no CLI, no pass registration, no caller in any compile path. The
three ``*_pretransposed`` bundles on disk were produced by hand-driving the library from a REPL, which
is why nobody could reproduce them.

WHY THIS IS A RAM-WALL LEVER, NOT A SPEED LEVER. Be precise about the size of the prize: on the whole
model the runtime transposes measure **0.4 %** of spectformer int8 and **1.2 %** of whisper int8 — at
or inside the K1's 2.6 % noise band, so this is not where the wall time is. What it does buy is
capacity and correctness: on Gemma 2 2B, 183 argument transposes moved **2,493 MiB per inference**
against a 2,505 MiB weight blob (essentially every weight, every time), and the largest of them — the
562.5 MiB int8 tied head — killed a whole-model FireSim run outright with
``FAIL alloc bytes=589824064`` at op 11,494 of 11,526. Pre-transposing needs no runtime change at all:
weights are ``@forward`` arguments bound as offsets into one flat blob, and a transpose changes neither
the offset nor the byte count.

PRECEDENCE vs ``fuse_transpose_b`` — decided, because the two solve the same measured problem and
CONFLICT (fusing first removes the very transposes this looks for):

  Prefer THIS. It is an offline, bit-exact bundle rewrite with a measured capacity payoff and no
  measured regression, and it is the only one of the two that changes the STORED layout — which is what
  an accelerator wanting K-major weights in DRAM actually needs. ``fuse_transpose_b`` is a default-off
  schedule lever that folds the transpose into the matmul's access pattern, leaves the stored layout
  alone, and measured **-6.53 % on openvla** the one time it was run. So: hoist first, then consider
  fusion only for the re-layouts the hoist reports as BLOCKED (an argument with readers other than the
  transpose, where pre-applying the layout would change what those readers see).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _report(src: Path, func: str):
    """The hoistable/blocked split for a bundle, without writing anything.

    Parses under ``IR_LOCK`` and via ``mlir_query.parse``, the same way ``hoist_weight_transposes``
    does, so a --dry-run and the real run can never disagree about what is hoistable.
    """
    from ..common import mlir_query as mq
    from ..common.ir_lock import IR_LOCK
    from ..xdsl_dialects.lowering.weight_layout import weight_layout_report

    with IR_LOCK:
        return weight_layout_report(mq.parse((src / "model.mlir").read_text()), func)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        prog="merlin-bundle-pretranspose",
        description="Pre-apply weight transposes to a capture bundle (offline, bit-exact).")
    ap.add_argument("bundle", help="source capture bundle directory")
    ap.add_argument("--out", default=None,
                    help="destination bundle (default: <bundle>_pretransposed, a SIBLING of the source)")
    ap.add_argument("--func", default="forward", help="function to analyse (default: forward)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the hoistable/blocked split and the bytes it would save; write nothing")
    a = ap.parse_args(argv)

    src = Path(a.bundle).resolve()
    if not (src / "model.mlir").is_file():
        raise SystemExit(f"[pretranspose] {src} is not a capture bundle (no model.mlir)")

    rep = _report(src, a.func)
    mib = rep.hoistable_bytes / 2 ** 20
    print(f"[pretranspose] {src.name}: {len(rep.hoistable)} hoistable, {len(rep.blocked)} blocked"
          f"{f', {len(rep.unpriceable)} UNPRICEABLE' if rep.unpriceable else ''}")
    print(f"[pretranspose] moved per inference today: {mib:,.1f} MiB ({rep.hoistable_bytes:,} bytes)")
    for r in rep.blocked:
        print(f"[pretranspose]   BLOCKED arg {r.arg}: the argument has readers besides the transpose, "
              "so pre-applying the layout would change what they see")
    if rep.unpriceable:
        # fail-closed: an unpriceable re-layout is one the analysis could not size, and a plan that
        # silently omits it would under-report the saving. Say so rather than proceeding quietly.
        print(f"[pretranspose] REFUSING to claim a total: {rep.unpriceable}")

    if a.dry_run:
        return 0
    if not rep.hoistable:
        raise SystemExit(f"[pretranspose] nothing to hoist in {src.name} (already pre-transposed?)")

    dst = Path(a.out).resolve() if a.out else src.parent / f"{src.name}_pretransposed"
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"[pretranspose] {dst} exists and is not empty; pass --out or remove it")

    from .bundle_rewrite import hoist_weight_transposes

    rec = hoist_weight_transposes(src, dst, a.func)
    print(f"[pretranspose] wrote {dst}")
    print(json.dumps({"name": rec.name, "source_bundle": rec.source_bundle,
                      "effect": rec.effect, "caveats": rec.caveats}, indent=2))
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
