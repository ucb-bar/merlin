"""Production wiring for the pass slot's four gate checks.

:mod:`mining.pass_slot` defines the ordered gate but INJECTS its four checks, so it is testable with
no toolchain, board or agent. This module supplies the real ones. Keeping them apart is deliberate:
the gate is the part that must be reviewable at a glance, and it should not drag a compiler toolchain
into every test that touches it.

Two properties matter more than the plumbing.

**The proposal is applied in an isolated OVERLAY, never in the working tree.** A pass proposal is
replacement source for a library module, and this repo is developed by several agents against one
shared checkout -- writing a module in place would change what every other session is building, and a
crash would leave it changed. So the source is written to a temp directory that shadows the package on
``PYTHONPATH`` for the child build only. Nothing in the checkout is touched, and there is nothing to
roll back if the gate rejects.

**Every check is a comparison against a control measured in the SAME session.** The host and board are
both shared and loaded, so a wall or a digest recorded on a different day is not a comparand. Each
check here either compares two runs of this call, or compares against a digest computed from the bytes
actually read.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from .pass_slot import PassProposal


def module_to_relpath(module: str) -> Path:
    """``merlin.llvmlower.act_poly`` -> ``merlin/llvmlower/act_poly.py``."""
    parts = module.split(".")
    if not parts or any(not p.isidentifier() for p in parts):
        raise ValueError(f"not a dotted module path: {module!r}")
    return Path(*parts).with_suffix(".py")


def _mirror_except(real: Path, dst: Path, chain: list[str], source: str) -> None:
    """Mirror ``real`` into ``dst`` with symlinks, descending ``chain`` and writing ``source`` at its
    leaf as the one REAL file. Symlinks mean nothing is copied and nothing in the checkout is opened
    for writing; the leaf is the single divergence, which is what makes the blast radius checkable."""
    dst.mkdir(parents=True, exist_ok=True)
    head = chain[0]
    for entry in real.iterdir():
        if entry.name == head:
            continue                     # replaced, or descended into
        (dst / entry.name).symlink_to(entry)
    if len(chain) == 1:
        (dst / head).write_text(source, encoding="utf-8")
    else:
        _mirror_except(real / head, dst / head, chain[1:], source)


@contextmanager
def overlay_for(proposal: PassProposal, *, package_root: Path | None = None):
    """Yield an env mapping whose ``PYTHONPATH`` shadows ``proposal.module`` with its new source.

    The overlay MIRRORS the real package tree with symlinks and replaces exactly one file, so the
    proposal's blast radius is precisely the module it claims to replace and everything else still
    resolves to the real checkout's bytes. Copying just the ``__init__.py`` chain does NOT work and
    was the first attempt: an overlay ``merlin/__init__.py`` shadows the whole package, and since the
    overlay has no ``merlin/kernels/`` the child process cannot import it at all -- every build under
    the overlay fails for a reason that has nothing to do with the proposal.

    The working tree is never written. This is not tidiness -- other sessions are building from it.
    """
    from ..common.paths import merlin_dir
    root = Path(package_root) if package_root is not None else merlin_dir() / "python"
    rel = module_to_relpath(proposal.module)
    if not (root / rel).is_file():
        raise FileNotFoundError(
            f"{proposal.module} does not exist at {root/rel}; the slot replaces an EXISTING pass, so "
            f"a proposal naming a new module has to add it to the checkout under review instead")
    tmp = Path(tempfile.mkdtemp(prefix="merlin_passslot_", dir=os.environ.get("TMPDIR") or None))
    try:
        _mirror_except(root, tmp, list(rel.parts), proposal.source)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(tmp), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        env["MERLIN_PASS_SLOT_OVERLAY"] = str(tmp)
        yield env
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def emitted_digest_of(run_dir: Path) -> str | None:
    """Digest of a run's emitted MNEMONIC stream, or None when there is no objdump.

    Reuses the beam's digest so 'inert' means the same thing in both places. Mnemonics rather than raw
    text, so register-allocation noise and symbol offsets cannot mask a no-op as a change.
    """
    from .beam import _emitted_digest
    return _emitted_digest(Path(run_dir))


def lift_run_cca(run_dir: Path, *, op: str):
    """Lift OUR CCA from a finished run, threading the object's undefined symbols.

    The undefined symbols are not optional. Without them the envelope cannot NAME the runtime helper
    it escapes to and ``compute.activation_vectorization`` cannot see a scalar ``expf`` at all (an
    unlinked ``model.o`` has no ``<expf>`` label) -- so the two axes whose CODEGEN rungs this slot
    exists to serve would both read as absent, and the gate would credit a proposal that changed
    nothing about them.
    """
    from ..kernels import cca
    from ..kernels.decode import rvv
    from .beam import _undef_syms
    run_dir = Path(run_dir)
    objd = run_dir / "generated" / "objdump.txt"
    if not objd.is_file():
        return None
    return cca.lift_asm(rvv.decode_text(objd.read_text()), op=op, source="ours",
                        undefined_symbols=_undef_syms(run_dir))


def _certify(env: dict[str, str] | None, *, package_dir: Path, model_dir: Path, runs_root: Path,
             run_id: str, targets: tuple[str, ...], timeout: int) -> dict[str, Any]:
    """Run ``certify_rvv`` in a CHILD process, so an overlaid module is actually imported fresh.

    In-process would not work: the parent has already imported the module the proposal replaces, and
    ``sys.modules`` would keep serving the old one. Running out-of-process also means a proposal that
    crashes the compiler is a recorded refusal rather than the end of the slot.
    """
    import json
    script = (
        "import json,sys\n"
        "from pathlib import Path\n"
        "from merlin.mining.runner import certify_rvv\n"
        "a=json.loads(sys.argv[1])\n"
        "r=certify_rvv(a['package_dir'], a['model_dir'], runs_root=a['runs_root'],\n"
        "              run_id=a['run_id'], targets=tuple(a['targets']), timeout=a['timeout'])\n"
        "print('__MERLIN_RESULT__'+json.dumps({'status':r.get('status'),\n"
        "      'correctness':r.get('correctness'),'measurement':r.get('measurement'),\n"
        "      'failure':r.get('failure')}, default=str))\n")
    arg = json.dumps({"package_dir": str(package_dir), "model_dir": str(model_dir),
                      "runs_root": str(runs_root), "run_id": run_id,
                      "targets": list(targets), "timeout": timeout})
    proc = subprocess.run([sys.executable, "-c", script, arg], env=env, capture_output=True,
                          text=True, timeout=timeout + 120)
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("__MERLIN_RESULT__"):
            return json.loads(line[len("__MERLIN_RESULT__"):])
    return {"status": "error",
            "failure": f"certify produced no result (rc={proc.returncode}): "
                       f"{(proc.stderr or '')[-600:]}"}


def _cos_of(rec: dict) -> float | None:
    c = rec.get("correctness") or {}
    for k in ("fp32_cos", "w8a8_cos"):
        if c.get(k) is not None:
            return float(c[k])
    return None


def production_gate_checks(
    *, frozen_pkg: Path, work_pkg: Path, model_dir: Path, runs_root: Path,
    heldout_model_dirs: tuple[Path, ...] = (), op: str = "matmul",
    targets: tuple[str, ...] = ("spike",), timeout: int = 3600,
    certify: Callable[..., dict] | None = None,
) -> dict[str, Callable]:
    """Build the four real gate checks as ``gate(**checks)`` keyword arguments.

    ``frozen_pkg`` is the frozen baseline package (empty features), ``work_pkg`` the package the
    proposal is meant to improve. Both are measured twice -- once WITHOUT the overlay and once with --
    inside this call, so every verdict rests on a same-session control.

    ``targets=("spike",)`` by default: the gate's job is correctness and facet acquisition, which spike
    answers deterministically and without the board lock. Ranking on wall is the beam's job, and it is
    the caller's choice to add ``"k1"``.

    ``certify`` is injectable so the whole wiring is testable without a toolchain.
    """
    runs_root = Path(runs_root)
    cert = certify if certify is not None else _certify
    state: dict[str, Any] = {}

    def _run(env, pkg: Path, mdir: Path, tag: str) -> dict:
        rec = cert(env, package_dir=Path(pkg), model_dir=Path(mdir), runs_root=runs_root,
                   run_id=tag, targets=targets, timeout=timeout)
        rec["_run_dir"] = str(runs_root / tag)
        return rec

    def frozen_baseline_ok(proposal: PassProposal) -> bool:
        """Empty features must still lower byte-identically WITH the proposal applied.

        This is the invariant every measurement in the repo is read against: if the control moves,
        every recorded delta silently moves with it. Compared against a control run in this same call,
        never a stored digest.
        """
        base = _run(None, frozen_pkg, model_dir, "gate_frozen_control")
        with overlay_for(proposal) as env:
            cand = _run(env, frozen_pkg, model_dir, "gate_frozen_overlay")
        a, b = emitted_digest_of(base["_run_dir"]), emitted_digest_of(cand["_run_dir"])
        state["frozen"] = {"control_digest": a, "overlay_digest": b,
                           "control_status": base.get("status"), "overlay_status": cand.get("status")}
        if a is None or b is None:
            return False              # no emitted code to compare -> cannot assert the invariant
        return a == b

    def bit_exact_ok(proposal: PassProposal) -> tuple[bool, str]:
        """The proposal's build must still match the golden on the work package."""
        with overlay_for(proposal) as env:
            rec = _run(env, work_pkg, model_dir, "gate_bitexact_overlay")
        state["bit_exact"] = rec
        if rec.get("status") == "error":
            return False, f"build/run failed: {str(rec.get('failure'))[:300]}"
        c = rec.get("correctness") or {}
        if not c.get("gate_ok"):
            return False, f"correctness gate failed (cos={_cos_of(rec)})"
        return True, f"cos={_cos_of(rec)}"

    def lift_cca(proposal: PassProposal):
        """The achieved CCA, from the SAME run bit_exact_ok measured -- not a fresh build.

        Re-building would let a nondeterministic proposal show one object to the numeric check and a
        different one to the facet check, which is precisely the seam a promise audit must not have.
        """
        rec = state.get("bit_exact")
        if rec is None:
            with overlay_for(proposal) as env:
                rec = _run(env, work_pkg, model_dir, "gate_bitexact_overlay")
            state["bit_exact"] = rec
        return lift_run_cca(Path(rec["_run_dir"]), op=op)

    def heldout_ok(proposal: PassProposal) -> tuple[bool, str]:
        """The same numeric check on captures the proposer was not shown.

        Fails CLOSED when there are none: 'held out' has to mean something was held out. A slot run
        with no held-out bundle is honest only if it says so, and the caller passes ``heldout_ok=None``
        to say it deliberately.
        """
        if not heldout_model_dirs:
            return False, "no held-out captures supplied, so generalisation was never tested"
        bad = []
        for i, mdir in enumerate(heldout_model_dirs):
            with overlay_for(proposal) as env:
                rec = _run(env, work_pkg, mdir, f"gate_heldout_{i}")
            c = rec.get("correctness") or {}
            if rec.get("status") == "error" or not c.get("gate_ok"):
                bad.append(f"{Path(mdir).name}(cos={_cos_of(rec)}, {str(rec.get('failure'))[:120]})")
        state["heldout"] = {"n": len(heldout_model_dirs), "failed": bad}
        return (not bad), ("all held-out captures passed" if not bad else f"failed on {bad}")

    checks = {"frozen_baseline_ok": frozen_baseline_ok, "bit_exact_ok": bit_exact_ok,
              "lift_cca": lift_cca}
    if heldout_model_dirs:
        checks["heldout_ok"] = heldout_ok
    checks["_state"] = state       # the caller records this; not consumed by gate()
    return checks


def gate_kwargs(checks: dict[str, Callable]) -> dict[str, Callable]:
    """``checks`` minus the bookkeeping key, ready to splat into ``pass_slot.gate``."""
    return {k: v for k, v in checks.items() if not k.startswith("_")}


def digest_source(source: str) -> str:
    """Digest of the proposal source itself, for the run record -- what was actually gated."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
