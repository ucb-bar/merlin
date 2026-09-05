"""The two non-pytest layers the historical replay runs. Exit 0 = accepted, 1 = REJECTED.

Split out so `replay.py` invokes every layer the same way (a subprocess against a shadowed package)
and so each layer can be run by hand to see what it says. Both distinguish "rejected the input" from
"could not run": the replay scores only exit 1 as a detection, so an exception here must not exit 1.

* `lit` — the static layer: llvm-lit over `merlin/tests/data/lit`, pointed at the shadow through
  `MERLIN_LIT_PYTHONPATH`. Without that override the suite would test the CURRENT tree and report a
  clean pass for every historical defect.
* `oracle` — the numeric layer: for each tracked capsule that lowers, compare the independent golden
  against `reference_outputs` of the command buffer and against `simulate`. This is the pre-existing
  dynamic check the formal layers sit beside, and it is in the instrument so a detection can be
  ATTRIBUTED: a defect both catch is not evidence for the new layer.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

#: How many capsules the oracle layer checks. The replay runs this once per sampled commit, so the
#: bound is what keeps the sweep affordable; capsules are taken in sorted order, never sampled, so two
#: runs check the same ones.
ORACLE_CAPSULES = 12


def _lit(repo: Path) -> int:
    lit = repo / "third_party" / "llvm-build" / "bin" / "llvm-lit"
    suite = repo / "merlin" / "tests" / "data" / "lit"
    if not lit.is_file() or not suite.is_dir():
        print("lit or the suite is unavailable; this layer did not run", file=sys.stderr)
        return 3
    env = dict(os.environ)
    shadow = os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]
    if shadow:
        env["MERLIN_LIT_PYTHONPATH"] = shadow
    proc = subprocess.run((str(lit), "-s", str(suite)), capture_output=True, text=True, env=env)
    if proc.returncode == 0:
        return 0
    print(proc.stdout[-4000:], file=sys.stderr)
    return 1


def _oracle(repo: Path) -> int:
    """Golden vs command buffer, on real tracked capsules.

    A capsule that does not lower here is SKIPPED rather than failed: the replay pins old library files
    over the package, and an old lowering that cannot build a buffer for a capsule written later is a
    mismatch between the two, not a numeric disagreement. Counting it as a rejection would credit this
    layer with catching defects it never evaluated. If nothing at all lowers, the layer reports that it
    could not run (exit 3), because a zero-capsule pass is the "check that skipped and reported
    success" shape.
    """
    import yaml

    from merlin.runtime import simulate
    from merlin.runtime.reference import reference_outputs
    from merlin.targetgen.capsule_golden import golden

    root = repo / "merlin" / "contract" / "capsules"
    disagreements: list[str] = []
    checked = 0
    for cdir in sorted(p.parent for p in root.rglob("capsule.yaml")):
        if checked >= ORACLE_CAPSULES:
            break
        try:
            cap = yaml.safe_load((cdir / "capsule.yaml").read_text(encoding="utf-8")) or {}
            want = golden(cap, cdir)
            if not want:
                continue
            cb = _lower(cdir, cap)
            if cb is None:
                continue
            got_ref, got_sim = reference_outputs(cb), simulate(cb)["outputs"]
        except Exception:
            continue                      # this capsule is not evaluable here; not a disagreement
        checked += 1
        for name, expected in want.items():
            if name in got_ref and got_ref[name] != expected:
                disagreements.append(f"{cdir.name}:{name} golden != reference")
            if name in got_sim and got_sim[name] != expected:
                disagreements.append(f"{cdir.name}:{name} golden != simulate")
    if not checked:
        print("no capsule was evaluable; this layer did not run", file=sys.stderr)
        return 3
    if disagreements:
        print(f"{len(disagreements)} disagreement(s) over {checked} capsule(s): "
              f"{disagreements[:6]}", file=sys.stderr)
        return 1
    return 0


def _lower(cdir: Path, cap: dict):
    """The capsule's command buffer, or None when this tree cannot produce one.

    The capsule's own `capsule.interface.mlir` is the input, parsed by the same
    `parse_interface_mlir` the backends use, so this layer sees the bytes a submission sees rather than
    a synthetic module built in-process.
    """
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir

    src = cdir / "capsule.interface.mlir"
    if not src.is_file():
        return None
    return parse_interface_mlir(src.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    from merlin.common.paths import repo_root

    argv = list(sys.argv[1:] if argv is None else argv)
    which = argv[0] if argv else ""
    if which == "lit":
        return _lit(repo_root())
    if which == "oracle":
        return _oracle(repo_root())
    print(f"usage: python -m merlin.verify.replay_layers <lit|oracle>", file=sys.stderr)
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
