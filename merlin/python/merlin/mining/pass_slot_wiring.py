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


def checkout_pythonpath(existing: str | None = None) -> str:
    """``PYTHONPATH`` that makes a child process import THIS checkout's ``merlin``.

    Necessary, not defensive. This venv installs ``merlin`` editable via a ``.pth`` file, and that file
    names whichever checkout last ran the install -- several checkouts of this repo share the venv, and
    one of them repointed it mid-session. pytest is unaffected (it inserts the rootdir), which is
    exactly why the shadowing is easy to miss: the suites keep passing while a plain
    ``python script.py`` imports a different tree.

    For the gate that is not cosmetic. The control arm runs with no overlay, so without this it would
    build with whatever tree the ``.pth`` names while the overlay arm builds from a mirror of THIS one
    -- and "the frozen baseline still lowers byte-identically" would be comparing two different
    compilers. Both arms must start from the checkout under test.
    """
    from ..common.paths import merlin_dir
    parts = [str(merlin_dir() / "python")]
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


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


class SeamNotActionable(RuntimeError):
    """The escalated action names no module the slot can overlay, with the declared reason why."""


def module_for_action(action) -> str:
    """The dotted module an escalated action's seam points at, or raise with the declared reason.

    This is the join between the ladder and the leaf. The catalog used to say WHAT to fix without
    saying WHERE: of the six blocked routes exactly one named a module that exists, so a caller had to
    know by hand which seam it could act on. Raising with the catalog's own declared reason keeps the
    two honest answers distinct -- "here is the module" and "there is no module, and here is why" --
    instead of collapsing both into None.
    """
    from ..kernels import action_catalog as ac
    seam = getattr(action, "target_seam", "") or ""
    mod = ac.seam_module(seam)
    if mod is not None:
        return mod
    why = ac.seam_needs_new_module(seam)
    raise SeamNotActionable(
        f"seam {seam!r} names no module the pass slot can overlay. "
        + (f"Declared reason: {why}" if why
           else "No reason is declared either, which is a catalog bug -- see "
                "test_action_catalog.test_every_blocked_route_says_WHERE_the_fix_goes."))


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
        # overlay first, then THIS checkout, then whatever the caller had. The overlay mirrors the
        # checkout, so the second entry is belt-and-braces -- but the base must never be the venv's
        # `.pth`, which may name a different tree entirely.
        env["PYTHONPATH"] = os.pathsep.join(
            [str(tmp), checkout_pythonpath(os.environ.get("PYTHONPATH"))])
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
    # env=None means "the control arm, no overlay" -- but it must still import THIS checkout, or the
    # frozen-baseline comparison is between two different compilers. See checkout_pythonpath.
    env = dict(env) if env is not None else dict(os.environ)
    env.setdefault("PYTHONPATH", "")
    if not env["PYTHONPATH"].startswith(str(Path(__file__).resolve().parents[3])):
        env["PYTHONPATH"] = checkout_pythonpath(env["PYTHONPATH"] or None)
    # -X importtime so we can tell whether the module under test was ACTUALLY imported. Without that
    # the numeric checks can pass vacuously: a package with empty features never imports a
    # feature-gated pass, so a module whose body is `raise` builds clean and the gate credits it.
    proc = subprocess.run([sys.executable, "-X", "importtime", "-c", script, arg], env=env,
                          capture_output=True, text=True, timeout=timeout + 120)
    out = {"status": "error",
           "failure": f"certify produced no result (rc={proc.returncode}): "
                      f"{(proc.stderr or '')[-600:]}"}
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("__MERLIN_RESULT__"):
            out = json.loads(line[len("__MERLIN_RESULT__"):])
            break
    out["_imported"] = sorted(imported_modules(proc.stderr or ""))
    return out


def imported_modules(importtime_stderr: str) -> set[str]:
    """Module names from ``python -X importtime`` output.

    Parsed structurally on the ``|`` column separator -- ``import time: self [us] | cumulative |
    imported package`` -- never by pattern. The last column is the dotted name, indented by import
    depth, so it is stripped.
    """
    names: set[str] = set()
    for line in importtime_stderr.splitlines():
        if not line.startswith("import time:"):
            continue
        cols = line.split("|")
        if len(cols) < 3:
            continue
        name = cols[-1].strip()
        if name and name != "imported package":
            names.add(name)
    return names


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
        # A check that could not run must not report success. `work_pkg` has to carry the features
        # that route through the proposed pass, or this build never imported it and "numerics are
        # bit-exact" says nothing about the proposal. MEASURED: a proposal whose entire body was
        # `raise RuntimeError` passed both the frozen and bit-exact checks against the empty-feature
        # baseline, because a feature-gated pass is never imported when its feature is off.
        imported = rec.get("_imported")
        if imported is not None and proposal.module not in set(imported):
            return False, (
                f"{proposal.module} was never imported by this build, so the numeric check did not "
                f"exercise the proposal. work_pkg={Path(work_pkg).name} must enable the feature that "
                f"routes through this pass.")
        return True, f"cos={_cos_of(rec)}"

    def inert_ok(proposal: PassProposal) -> tuple[bool, str]:
        """Did the proposal change the emitted code at all, against the SAME package unpatched?

        Without this the gate cannot tell "your pass ran and was not enough" from "your pass never
        fired", and it reported the second as the first -- so the next turn improves code that was
        never reached. The control is the work package built WITHOUT the overlay in this same call,
        so the comparison is not against a stored digest.
        """
        cand = state.get("bit_exact")
        if cand is None:
            bit_exact_ok(proposal)
            cand = state["bit_exact"]
        ctrl = _run(None, work_pkg, model_dir, "gate_inert_control")
        a, b = emitted_digest_of(ctrl["_run_dir"]), emitted_digest_of(cand["_run_dir"])
        state["inert"] = {"control_digest": a, "candidate_digest": b}
        if a is None or b is None:
            return False, ("no emitted code to compare, so a change could not be established "
                           f"(control={a}, candidate={b})")
        if a == b:
            return False, (f"byte-identical to the unpatched build (digest {a}); the pass was "
                           f"imported but its matching never fired, so nothing downstream of it ran")
        return True, f"emitted code changed ({a} -> {b})"

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
              "inert_ok": inert_ok, "lift_cca": lift_cca}
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


def make_pass_slot_fn(*, frozen_pkg: Path, model_dir: Path, runs_root: Path,
                      targets_root: Path, op: str = "matmul", max_turns: int = 2,
                      targets: tuple[str, ...] = ("spike",), timeout: int = 3600,
                      heldout_model_dirs: tuple[Path, ...] = (), model: str = "opus",
                      agent_timeout: int = 2400, certify: Callable[..., dict] | None = None,
                      propose_fn: Callable | None = None) -> Callable:
    """A ``pass_slot_fn(action, parent_run_id=...)`` for :func:`mining.beam.run_beam`.

    This is the join that closes the loop. The beam already walks the ladder: the CCA lifts the loss,
    the router picks the cheapest action, the fork is built and measured, ``achieved_residual`` says
    the promise went unmet, and ``route_escalated`` returns the next rung. When that rung is not
    forkable the ladder used to stop and record a work-item, and a human carried it to the slot by
    hand. Now the beam hands it over directly.

    The work package is the FORK that produced the residual, not the frozen baseline. That is the only
    correct choice: a feature-gated pass is never imported when its feature is off, so gating against
    the baseline would measure a build that never ran the proposal (the guard in ``bit_exact_ok``
    catches it, but as a refusal rather than a result).

    Returns a serializable record for every escalation it is handed, INCLUDING the ones it cannot act
    on -- "this rung needs a module that does not exist, and here is the declared reason" is the
    honest outcome for four of the six blocked seams, and dropping it would make the ladder look
    complete when it is not.
    """
    from ..common.paths import merlin_dir
    from . import pass_agent
    from .pass_slot import iterate_pass_slot

    def _slot(action, *, parent_run_id: str | None = None) -> dict[str, Any]:
        seam = getattr(action, "target_seam", "") or ""
        rec: dict[str, Any] = {"seam": seam, "axis": getattr(action, "divergence_axis", None),
                               "action_class": getattr(action, "action_class", None),
                               "parent_run_id": parent_run_id}
        try:
            module = module_for_action(action)
        except SeamNotActionable as e:
            rec.update(actionable=False, reason=str(e))
            return rec
        work_pkg = Path(targets_root) / str(parent_run_id) if parent_run_id else Path(frozen_pkg)
        if not work_pkg.is_dir():
            rec.update(actionable=False,
                       reason=f"the fork package {work_pkg} that produced this residual is not on "
                              f"disk, so there is nothing to gate the proposal against")
            return rec
        src_path = merlin_dir() / "python" / module_to_relpath(module)
        ws = Path(runs_root) / f"pass_slot_{parent_run_id or 'seed'}_{module.replace('.', '_')}"
        propose = propose_fn
        attempts: list = []
        if propose is None:
            propose, attempts = pass_agent.proposer_for(
                action, current_source=src_path.read_text(), workspace=ws,
                model=model, timeout=agent_timeout)
        checks = production_gate_checks(
            frozen_pkg=Path(frozen_pkg), work_pkg=work_pkg, model_dir=Path(model_dir),
            runs_root=ws / "gate", heldout_model_dirs=heldout_model_dirs, op=op,
            targets=targets, timeout=timeout, certify=certify)
        turns = iterate_pass_slot(action, propose=propose, max_turns=max_turns,
                                  **gate_kwargs(checks))
        rec.update(
            actionable=True, module=module, work_pkg=str(work_pkg), workspace=str(ws),
            turns=[{"accepted": v.accepted, "stage": v.stage, "reason": v.reason,
                    "residual": list(v.residual),
                    "source_digest": (digest_source(p.source) if p else None)}
                   for p, v in turns],
            accepted=any(v.accepted for _p, v in turns),
            agent=[a.to_dict() for a in attempts])
        return rec

    return _slot
