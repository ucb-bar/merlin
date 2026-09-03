"""Production wiring for the pass slot's gate checks.

The gate itself (mining.pass_slot) injects its four checks so it stays testable with no toolchain.
These tests cover the wiring that supplies the real ones, and the two properties that matter more than
the plumbing: the proposal is applied in an isolated OVERLAY that never touches the shared working
tree, and every verdict rests on a control measured in the same call.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir
from merlin.mining import pass_slot_wiring as w
from merlin.mining.pass_slot import PassProposal, gate

_REAL_MODULE = "merlin.llvmlower.act_poly"


def _prop(source="X = 1\n", module=_REAL_MODULE):
    return PassProposal(module=module, source=source, rationale="test")


# --------------------------------------------------------------------------- module paths

def test_module_to_relpath_round_trips_a_dotted_module():
    assert w.module_to_relpath("merlin.llvmlower.act_poly") == Path("merlin/llvmlower/act_poly.py")


@pytest.mark.parametrize("bad", ["", "merlin..act", "merlin/llvmlower", "1bad.mod", "merlin.a-b"])
def test_module_to_relpath_rejects_anything_that_is_not_a_dotted_module(bad):
    """A path that is not a module name must raise, not be coerced -- the overlay writes a file at
    whatever this returns, and coercing '../x' into a path is how an overlay escapes its temp dir."""
    with pytest.raises(ValueError):
        w.module_to_relpath(bad)


# --------------------------------------------------------------------------- the overlay

def test_the_overlay_never_writes_the_working_tree():
    """The property this repo requires: several agents build from one shared checkout, so applying a
    proposal in place would change what every other session compiles -- and a crash would leave it
    changed. Assert the real module's bytes are untouched, during AND after."""
    target = merlin_dir() / "python" / w.module_to_relpath(_REAL_MODULE)
    before = target.read_bytes()
    with w.overlay_for(_prop("SENTINEL = 'overlaid'\n")) as env:
        assert target.read_bytes() == before, "the checkout was modified inside the overlay"
        assert Path(env["MERLIN_PASS_SLOT_OVERLAY"]).is_dir()
        overlay_copy = Path(env["MERLIN_PASS_SLOT_OVERLAY"]) / w.module_to_relpath(_REAL_MODULE)
        assert "SENTINEL" in overlay_copy.read_text()
    assert target.read_bytes() == before, "the checkout was modified after the overlay"


def test_the_overlay_is_removed_even_when_the_body_raises():
    with pytest.raises(RuntimeError):
        with w.overlay_for(_prop()) as env:
            tmp = Path(env["MERLIN_PASS_SLOT_OVERLAY"])
            raise RuntimeError("boom")
    assert not tmp.exists(), "a failed gate must not leave overlay dirs behind"


def test_a_child_process_actually_imports_the_overlaid_module():
    """The overlay is worth nothing if it does not shadow. Checked in a CHILD process because the
    parent has already imported the real module and sys.modules would keep serving it."""
    with w.overlay_for(_prop("SENTINEL = 'overlaid'\n")) as env:
        out = subprocess.run(
            [sys.executable, "-c",
             f"import {_REAL_MODULE} as m; print(getattr(m, 'SENTINEL', 'NOT_SHADOWED'))"],
            env=env, capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "overlaid", f"overlay did not shadow: {out.stdout} {out.stderr[-300:]}"


def test_the_rest_of_the_package_still_resolves_from_the_real_checkout():
    """Blast radius: only the named module is replaced. A proposal that silently needs a second file
    edited must fail here rather than pass on a change nobody reviewed."""
    with w.overlay_for(_prop("SENTINEL = 1\n")) as env:
        out = subprocess.run(
            [sys.executable, "-c",
             "import merlin.kernels.cca as c; import merlin.common.paths as p; print('ok', bool(c.CCA and p.repo_root()))"],
            env=env, capture_output=True, text=True, timeout=120)
    assert out.stdout.startswith("ok"), out.stderr[-400:]


def test_overlay_refuses_a_module_that_does_not_exist():
    """The slot REPLACES an existing pass. A proposal naming a new module is adding a file to the
    checkout, which is a reviewed change, not something a gate should conjure into a temp dir."""
    with pytest.raises(FileNotFoundError):
        with w.overlay_for(_prop(module="merlin.llvmlower.not_a_real_pass_module")):
            pass


# --------------------------------------------------------------------------- the checks

class _FakeCertify:
    """Records each certify call and returns a scripted result, so the wiring is testable with no
    toolchain. `digests` maps run_id -> the mnemonic digest that run should appear to emit."""

    def __init__(self, *, digests=None, gate_ok=True, status="ok", failure=None, cos=0.9999999,
                 imported=(_REAL_MODULE,)):
        self.imported = imported
        self.calls = []
        self.digests = digests or {}
        self.gate_ok, self.status, self.failure, self.cos = gate_ok, status, failure, cos

    def __call__(self, env, *, package_dir, model_dir, runs_root, run_id, targets, timeout):
        self.calls.append({"run_id": run_id, "overlaid": env is not None,
                           "model_dir": Path(model_dir).name, "package_dir": Path(package_dir).name})
        d = Path(runs_root) / run_id / "generated"
        d.mkdir(parents=True, exist_ok=True)
        mnem = self.digests.get(run_id, "vfmacc.vv v1,v2,v3")
        (d / "objdump.txt").write_text(
            "0000000000000000 <forward>:\n"
            f"   0:\t02b7f0d7          \t{mnem}\n")
        return {"status": self.status, "failure": self.failure,
                "correctness": {"gate_ok": self.gate_ok, "fp32_cos": self.cos},
                "measurement": [{"target": "spike", "cycles": 1000}],
                "_imported": list(self.imported) if self.imported is not None else None}


def _checks(tmp_path, cert, **kw):
    return w.production_gate_checks(
        frozen_pkg=tmp_path / "frozen", work_pkg=tmp_path / "work",
        model_dir=tmp_path / "bundle", runs_root=tmp_path / "runs",
        certify=cert, **kw)


def test_frozen_baseline_check_compares_overlay_against_a_same_session_control(tmp_path):
    """The invariant every recorded delta is read against. It must be checked against a control run
    in THIS call -- the host and board are shared and loaded, so a stored digest is not a comparand."""
    cert = _FakeCertify()
    checks = _checks(tmp_path, cert)
    assert checks["frozen_baseline_ok"](_prop()) is True
    ids = [c["run_id"] for c in cert.calls]
    assert ids == ["gate_frozen_control", "gate_frozen_overlay"]
    assert [c["overlaid"] for c in cert.calls] == [False, True], \
        "one run must be WITHOUT the overlay, or there is no control"


def test_frozen_baseline_check_fails_when_the_proposal_moves_the_control(tmp_path):
    cert = _FakeCertify(digests={"gate_frozen_overlay": "vfmul.vv v1,v2,v3"})
    checks = _checks(tmp_path, cert)
    assert checks["frozen_baseline_ok"](_prop()) is False


def test_frozen_baseline_fails_closed_when_there_is_no_emitted_code_to_compare(tmp_path):
    """No objdump means the invariant was not asserted. Reporting True would be crediting a check
    that could not run -- the exact failure this repo has hit three times."""
    class _NoObjdump(_FakeCertify):
        def __call__(self, env, **kw):
            _FakeCertify.__call__(self, env, **kw)
            (Path(kw["runs_root"]) / kw["run_id"] / "generated" / "objdump.txt").unlink()
            return {"status": "error", "failure": "compile failed", "correctness": {}}
    checks = _checks(tmp_path, _NoObjdump())
    assert checks["frozen_baseline_ok"](_prop()) is False


def test_bit_exact_check_reports_the_cos_and_fails_on_a_broken_gate(tmp_path):
    ok, why = _checks(tmp_path, _FakeCertify())["bit_exact_ok"](_prop())
    assert ok and "cos=" in why
    ok, why = _checks(tmp_path, _FakeCertify(gate_ok=False, cos=0.31))["bit_exact_ok"](_prop())
    assert not ok and "0.31" in why


def test_bit_exact_check_fails_on_a_build_that_crashed(tmp_path):
    """A proposal that breaks the compiler is a recorded refusal, not an exception out of the slot."""
    ok, why = _checks(tmp_path, _FakeCertify(status="error", failure="PipelineError: bad schedule")
                      )["bit_exact_ok"](_prop())
    assert not ok and "PipelineError" in why


def test_lift_cca_reuses_the_run_the_numeric_check_measured(tmp_path):
    """Re-building would let a nondeterministic proposal show one object to the numeric check and a
    different one to the facet check -- precisely the seam a promise audit must not have."""
    cert = _FakeCertify()
    checks = _checks(tmp_path, cert)
    checks["bit_exact_ok"](_prop())
    n_before = len(cert.calls)
    c = checks["lift_cca"](_prop())
    assert len(cert.calls) == n_before, "lift_cca rebuilt instead of reusing the measured run"
    assert c is not None and c.compute is not None


def test_heldout_is_absent_rather_than_vacuously_true_when_nothing_is_held_out(tmp_path):
    """"held out" has to mean something was held out. The check is omitted entirely, so the caller
    passing it must have supplied bundles; a vacuous pass would credit untested generalisation."""
    checks = _checks(tmp_path, _FakeCertify())
    assert "heldout_ok" not in checks
    with_held = _checks(tmp_path, _FakeCertify(), heldout_model_dirs=(tmp_path / "other",))
    assert "heldout_ok" in with_held


def test_heldout_check_runs_every_supplied_capture_and_names_the_failures(tmp_path):
    cert = _FakeCertify(gate_ok=False, cos=0.5)
    checks = _checks(tmp_path, cert,
                     heldout_model_dirs=(tmp_path / "held_a", tmp_path / "held_b"))
    ok, why = checks["heldout_ok"](_prop())
    assert not ok
    assert "held_a" in why and "held_b" in why, f"a failing held-out capture must be named: {why}"


def test_gate_kwargs_drops_the_bookkeeping_key(tmp_path):
    checks = _checks(tmp_path, _FakeCertify())
    kw = w.gate_kwargs(checks)
    assert "_state" not in kw and "frozen_baseline_ok" in kw


def test_the_wired_gate_runs_end_to_end_and_refuses_an_unverifiable_action(tmp_path):
    """The whole path: real gate, real wiring, fake toolchain. An action with no machine-readable
    promise must be REFUSED, since 'accepted' would credit a change nothing checked."""
    class _NoPromise:
        intended_facet = None
        action_class = "CODEGEN"
    v = gate(_prop(), _NoPromise(), **w.gate_kwargs(_checks(tmp_path, _FakeCertify())))
    assert not v.accepted and v.stage == "unverifiable"


def test_a_cheating_proposal_is_refused_before_anything_is_built(tmp_path):
    """The cheat scan is first because it is free. Assert no certify call happened at all."""
    cert = _FakeCertify()
    checks = _checks(tmp_path, cert)
    v = gate(_prop("golden = open('golden.npy')\n"), object(), **w.gate_kwargs(checks))
    assert not v.accepted and v.stage == "cheat"
    assert cert.calls == [], "a cheating proposal must be rejected before any build"


def test_digest_source_is_stable_and_short():
    a, b = w.digest_source("x = 1\n"), w.digest_source("x = 1\n")
    assert a == b and len(a) == 16 and a != w.digest_source("x = 2\n")


def test_exactly_one_file_in_the_overlay_is_not_a_symlink():
    """The blast-radius property, stated precisely.

    The overlay mirrors the real tree with symlinks so nothing is copied and nothing in the checkout
    is opened for writing. The ONE real file is the module the proposal claims to replace -- so
    "only this pass changed" is a checkable fact about the overlay, not a promise about the proposal.
    """
    rel = w.module_to_relpath(_REAL_MODULE)
    with w.overlay_for(_prop("SENTINEL = 1\n")) as env:
        root = Path(env["MERLIN_PASS_SLOT_OVERLAY"])
        real_files = [p.relative_to(root) for p in root.rglob("*")
                      if p.is_file() and not p.is_symlink()]
        assert real_files == [rel], f"expected only {rel} to be a real file, got {real_files}"


def test_the_overlaid_module_resolves_while_its_siblings_stay_real():
    """Both halves at once, in a child process: the replaced module is the proposal's, and a sibling
    module in the SAME package is still the checkout's."""
    with w.overlay_for(_prop("SENTINEL = 'overlaid'\n")) as env:
        out = subprocess.run(
            [sys.executable, "-c",
             f"import {_REAL_MODULE} as m, merlin.llvmlower.impr_features as f;"
             " print(getattr(m,'SENTINEL','NOT_SHADOWED'), hasattr(f,'normalize'))"],
            env=env, capture_output=True, text=True, timeout=180)
    assert out.stdout.strip() == "overlaid True", f"{out.stdout!r} {out.stderr[-400:]}"


def test_bit_exact_refuses_when_the_proposed_module_was_never_imported(tmp_path):
    """A check that could not run must not report success -- this repo's most expensive recurring
    failure. MEASURED: a proposal whose entire body was `raise RuntimeError` passed both the frozen
    and bit-exact checks against the empty-feature baseline, because a feature-gated pass is simply
    never imported when its feature is off. So `work_pkg` has to enable the feature that routes
    through the pass, and the wiring says so instead of crediting the build."""
    cert = _FakeCertify(imported=("merlin.kernels.cca", "merlin.common.paths"))
    ok, why = _checks(tmp_path, cert)["bit_exact_ok"](_prop())
    assert not ok
    assert "never imported" in why and _REAL_MODULE in why


def test_bit_exact_still_passes_when_the_module_was_imported(tmp_path):
    ok, why = _checks(tmp_path, _FakeCertify(imported=(_REAL_MODULE, "merlin.kernels.cca"))
                      )["bit_exact_ok"](_prop())
    assert ok and "cos=" in why


def test_an_import_list_that_is_absent_does_not_block(tmp_path):
    """A certify seam that cannot report imports (an injected fake, an older runner) must not turn
    every proposal into a refusal -- unknown is not the same as absent."""
    ok, _ = _checks(tmp_path, _FakeCertify(imported=None))["bit_exact_ok"](_prop())
    assert ok


def test_importtime_output_is_parsed_on_its_column_separator():
    """Structural parse of the `|` columns, never a pattern. Indentation marks import depth and is
    stripped; non-importtime lines and the header are ignored."""
    sample = ("import time:       123 |        456 | merlin.llvmlower.act_poly\n"
              "import time:        11 |         11 |   merlin.kernels.cca\n"
              "import time: self [us] | cumulative | imported package\n"
              "some unrelated stderr line\n")
    assert w.imported_modules(sample) == {"merlin.llvmlower.act_poly", "merlin.kernels.cca"}
    assert w.imported_modules("") == set()


def test_module_for_action_joins_the_ladder_to_the_leaf():
    """The escalated activation action resolves to the module the slot then overlays -- the join the
    catalog was missing."""
    from merlin.kernels import action_catalog as ac
    from merlin.kernels.cca_compare import Divergence
    d = Divergence(axis="compute.activation_vectorization", expert="vectorized_polynomial",
                   ours="scalar_libm_call", backend="rvv")
    esc = ac.route_escalated(d, ac.route(d).action_class)
    assert esc is not None and esc.action_class == "CODEGEN"
    assert w.module_for_action(esc) == _REAL_MODULE


def test_module_for_action_raises_with_the_declared_reason_when_there_is_no_module():
    """"There is no module, and here is why" must stay distinct from "here is the module". Collapsing
    both into None is what let the ladder dead-end in prose unnoticed."""
    class _A:
        target_seam = "pass:tile-epilogue-store-once (eliminate the rank-generic copy, not erase it)"
    with pytest.raises(w.SeamNotActionable, match="rank-generic"):
        w.module_for_action(_A())


def test_module_for_action_says_it_is_a_catalog_bug_when_no_reason_is_declared():
    class _A:
        target_seam = "pass:something-nobody-declared"
    with pytest.raises(w.SeamNotActionable, match="catalog bug"):
        w.module_for_action(_A())


def test_the_checkout_under_test_is_pinned_for_child_builds():
    """This venv installs merlin editable via a .pth naming whichever checkout last ran the install,
    and several checkouts share it -- one repointed it mid-session. pytest is unaffected (it inserts
    the rootdir), which is why the shadowing is easy to miss: the suites keep passing while a plain
    `python script.py` imports a different tree.

    For the gate that is not cosmetic: the control arm runs with no overlay, so without pinning it
    would build with whatever tree the .pth names while the overlay arm builds from a mirror of THIS
    one, and "the frozen baseline still lowers byte-identically" would compare two different
    compilers."""
    from merlin.common.paths import merlin_dir
    base = w.checkout_pythonpath()
    assert base.split(os.pathsep)[0] == str(merlin_dir() / "python")
    chained = w.checkout_pythonpath("/somewhere/else")
    assert chained.split(os.pathsep) == [str(merlin_dir() / "python"), "/somewhere/else"]


def test_the_control_arm_gets_the_checkout_even_with_no_overlay_env(tmp_path, monkeypatch):
    """The control arm is invoked with env=None. It must still receive a PYTHONPATH pinning this
    checkout, or the frozen-baseline digests come from two different compilers."""
    from merlin.common.paths import merlin_dir
    seen = {}

    class _Proc:
        returncode = 0
        stdout = "__MERLIN_RESULT__" + json.dumps({"status": "pass", "correctness": {}})
        stderr = ""

    monkeypatch.setattr(w.subprocess, "run", lambda argv, **kw: (seen.update(kw), _Proc())[1])
    w._certify(None, package_dir=tmp_path, model_dir=tmp_path, runs_root=tmp_path,
               run_id="r", targets=("spike",), timeout=10)
    pp = (seen.get("env") or {}).get("PYTHONPATH", "")
    assert str(merlin_dir() / "python") in pp, f"control arm not pinned to this checkout: {pp!r}"
