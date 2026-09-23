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

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

#: How many capsules the oracle layer checks. The replay runs this once per sampled commit, so the
#: bound is what keeps the sweep affordable; capsules are taken in sorted order, never sampled, so two
#: runs check the same ones.
ORACLE_CAPSULES = 12


def _qualified_pytest(argv: list[str]) -> str:
    """Count executed comparisons; pytest exit 1 also covers import/setup/runtime crashes."""
    try:
        import pytest
    except ModuleNotFoundError as exc:
        if exc.name != "pytest":
            raise
        raise RuntimeError("pytest is unavailable; install merlin-analysis[replay] in the replay interpreter") from exc

    class Evidence:
        passed = 0
        rejected = 0
        broken = 0

        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_makereport(self, item, call):
            report = (yield).get_result()
            if report.skipped:
                return
            if report.failed:
                if (
                    call.when == "call"
                    and call.excinfo
                    and isinstance(call.excinfo.value, (AssertionError, pytest.fail.Exception))
                ):
                    self.rejected += 1
                else:
                    self.broken += 1
            elif call.when == "call" and report.passed:
                self.passed += 1

    evidence = Evidence()
    code = pytest.main(argv, plugins=[evidence])
    if code not in (0, 1) or evidence.broken:
        return "error"
    if evidence.rejected:
        return "red"
    return "green" if code == 0 and evidence.passed else "error"


def qualify_layer(argv: list[str]) -> int:
    """Trusted live bootstrap, invoked by absolute filename outside the historical shadow.

    A completed receipt, not process exit alone, authorizes a measured verdict. Imports and crashes
    outside the actual assertion/numeric comparison never create a rejection receipt. This is an
    instrument qualification boundary, not a sandbox for executing hostile Python.
    """
    import hashlib

    receipt, policy, *command = argv
    path = Path(receipt)
    path.write_text(json.dumps({"qualification_policy": policy, "status": "running"}))
    status = "error"
    detail = ""
    try:
        context_path = Path(os.environ["MERLIN_REPLAY_IMPORT_CONTEXT"])
        if not context_path.is_file():
            raise RuntimeError("replay import context is not a regular file")
        context_bytes = context_path.read_bytes()
        if hashlib.sha256(context_bytes).hexdigest() != os.environ["MERLIN_REPLAY_IMPORT_CONTEXT_SHA256"]:
            raise RuntimeError("replay import context hash mismatch")
        context = json.loads(context_bytes)
        if list(sys.version_info[:2]) != context["python_version"]:
            raise RuntimeError("replay interpreter major/minor must match the parent's dependency environment")
        helper = Path(context["helper"])
        if not helper.is_file():
            raise RuntimeError("replay frozen import helper is not a regular file")
        source = helper.read_bytes()
        if hashlib.sha256(source).hexdigest() != context["helper_sha256"]:
            raise RuntimeError("replay frozen import helper hash mismatch")
        namespace = {"__file__": str(helper), "__name__": "_replay_frozen_imports"}
        exec(compile(source, str(helper), "exec"), namespace)
        namespace["activate"](
            snapshot_root=context["snapshot_root"],
            import_roots=context["import_roots"],
            sources=context["sources"],
        )
        sys.path.extend(entry for entry in context["dependencies"] if entry not in sys.path)
        if command[:2] == ["-m", "pytest"]:
            status = _qualified_pytest(command[2:])
        elif command[:2] == ["-m", "merlin.verify.replay_layers"]:
            # The instrument stays live; only the compiler/runtime/oracle bytes are shadowed.
            repo = Path(os.environ["MERLIN_REPO_ROOT"])
            if command[2] == "lit":
                # A lit FAIL can itself be a Python import crash in %merlin-opt, and its children
                # do not inherit this process's finder. Keep standalone lit available, but do not
                # credit replay with a detection until those children produce qualified receipts.
                raise RuntimeError("lit child commands lack qualified exception/import receipts")
            code = {"oracle": _oracle}[command[2]](repo)
            status = {0: "green", 1: "red"}.get(code, "error")
        else:
            raise ValueError(f"unknown replay layer invocation: {command}")
    except BaseException as exc:
        # SystemExit(1), dependency failures and interrupts are not numeric/assertion evidence.
        detail = f"{type(exc).__name__}: {exc}"
        print(f"replay layer unavailable: {detail}", file=sys.stderr)
    record = {"qualification_policy": policy, "status": status}
    if detail:
        record["detail"] = detail
    path.write_text(json.dumps(record))
    return {"green": 0, "red": 1}.get(status, 3)


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
    with tempfile.TemporaryDirectory(prefix="merlin-replay-lit-") as tmp:
        report = Path(tmp) / "results.json"
        proc = subprocess.run(
            (str(lit), "-s", "--output", str(report), str(suite)), capture_output=True, text=True, env=env
        )
        if proc.returncode not in (0, 1) or not report.is_file():
            return 3
        try:
            tests = json.loads(report.read_text())["tests"]
            codes = [test["code"] for test in tests]
        except (OSError, ValueError, KeyError, TypeError):
            return 3
    if not codes or any(code not in {"PASS", "FAIL", "UNSUPPORTED", "XFAIL"} for code in codes):
        return 3
    if "FAIL" in codes:
        print(proc.stdout[-4000:], file=sys.stderr)
        return 1 if proc.returncode == 1 else 3
    return 0 if proc.returncode == 0 and "PASS" in codes else 3


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
            continue  # this capsule is not evaluable here; not a disagreement
        checked += 1
        for name, expected in want.items():
            if name not in got_ref:
                disagreements.append(f"{cdir.name}:{name} missing reference output")
            elif got_ref[name] != expected:
                disagreements.append(f"{cdir.name}:{name} golden != reference")
            if name not in got_sim:
                disagreements.append(f"{cdir.name}:{name} missing simulate output")
            elif got_sim[name] != expected:
                disagreements.append(f"{cdir.name}:{name} golden != simulate")
    if not checked:
        print("no capsule was evaluable; this layer did not run", file=sys.stderr)
        return 3
    if disagreements:
        print(f"{len(disagreements)} disagreement(s) over {checked} capsule(s): {disagreements[:6]}", file=sys.stderr)
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
    argv = list(sys.argv[1:] if argv is None else argv)
    which = argv[0] if argv else ""
    try:
        from merlin.common.paths import repo_root

        if which == "lit":
            return _lit(repo_root())
        if which == "oracle":
            return _oracle(repo_root())
    except Exception as exc:
        print(f"layer unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print("usage: python -m merlin.verify.replay_layers <lit|oracle>", file=sys.stderr)
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
