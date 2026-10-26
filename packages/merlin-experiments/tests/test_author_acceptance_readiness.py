"""The acceptance suite refuses whenever the launch's tooling-readiness preflight would.

Two launches on the withholding SDK inputs (then numbered V20, now V22) refused with "NO_GO: selected authoring tools are not ready" after the suite
had passed: the grant check ignored the bundle's recorded denials of the withheld kernels modules, and
the selected-sandbox ``cca_spine`` probe imported the withheld ``merlin.kernels``. The bundle here is
composed the way the release composes it (V22 author SDK, substrate v15, the arm's own grants); each
pre-fix code path is restored by monkeypatch and the suite's ``tooling-readiness`` row must carry the
launch's refusal, while the current code passes. The probe text is the one ``public_probe_session``
builds from the bundle; it runs in the suite's author interpreter (only the V22 SDK importable, every
withheld identity refused) in place of the bwrap argv, which needs a descriptor-selected toolchain.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import author_acceptance as AA
from merlin_experiments.phase1 import full_compiler_workspace as FCW
from merlin_experiments.phase1 import tooling_readiness as R

from merlin.common.paths import repo_root
from merlin.targetgen import tool_registry as TR
from merlin.targetgen.generate_bundles import _select_python_grants
from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V15, attach_common_substrate
from merlin.targetgen.tool_profile_shared import prepare_shared_source_release

ARM = "merlin_rtlchecks"
TOOLS = TR.ARM_TOOLS[ARM]
TARGET = "phantom5"
FACTS = repo_root() / "merlin/tests/fixtures/phantom_target/phantom5/contracts/rtl_facts/facts.json"
REFUSED = sorted({*FCW.DENIED_PREFIXES, *(m for e in FCW.MODULE_ACCESS for m in e.modules)})
#: The seven grants the first refused launch named.
WITHHELD_GRANTS = (
    "grant:merlin_infra:merlin/python/merlin/kernels/endpoints.py",
    "grant:merlin_infra:merlin/python/merlin/kernels/roles.py",
    "grant:cca_spine:merlin/python/merlin/kernels/cca.py",
    "grant:cca_spine:merlin/python/merlin/kernels/cca_compare.py",
    "grant:cca_spine:merlin/python/merlin/kernels/cca_contract.py",
    "grant:cca_spine:merlin/python/merlin/kernels/action_catalog.py",
    "grant:cca_spine:merlin/python/merlin/kernels/microkernel.py",
)


@pytest.fixture(scope="module")
def v22(tmp_path_factory):
    sdk = tmp_path_factory.mktemp("inputs-v22") / "author-library" / "base-sdk"
    prepare_shared_source_release(
        sdk,
        python_source_root=repo_root() / "src",
        license_directory=repo_root(),
        tools=TOOLS,
        version=22,
        source_binding=True,
    )
    allowed = [{"path": p, "mode": "ro", "note": "tool"} for t in TOOLS for p in TR.spec(t).bundle_paths]
    bundle = dict(
        bundle_id="merlin_assisted_rtlchecks_public_v0",
        arm=ARM,
        tools=list(TOOLS),
        allowed=allowed,
        denied=[],
        host_inputs=[],
    )
    _select_python_grants(bundle, sdk)
    attach_common_substrate(bundle, version=COMMON_SUBSTRATE_V15, python_source_root=sdk, host_index_bits=64)
    _select_python_grants(bundle, sdk)
    return sdk, bundle


def _inputs(tmp_path):
    context = SimpleNamespace(target=TARGET, repo=tmp_path, descriptor=tmp_path / "descriptor.yaml")
    te = SimpleNamespace(target=TARGET, rtl_facts_pin="target/contracts/rtl_facts/", sim_via="")
    ws = tmp_path / "run" / "workspace"
    ws.mkdir(parents=True)
    return context, te, ws


def _author_probe(sdk: Path):
    """``_live_probe`` with the suite's author interpreter standing in for the bwrap argv."""

    def live(context, te, ws, bundle, tools):
        probe_ws = ws.parent / "probe"
        probe_ws.mkdir(exist_ok=True)
        # The cca_spine probe is the one the second refused launch failed; it needs no broker.
        with R.public_probe_session(context, probe_ws, bundle, ("cca_spine",), facts_workspace=ws) as probe:
            proc, _ = AA.run_child(probe, [sdk], REFUSED, cwd=probe_ws, env={"MERLIN_RTL_FACTS": str(FACTS)})
        ok = proc.returncode == 0 and "AUTHORING_AND_BROKER_ROUNDTRIPS_OK" in proc.stdout
        output = (proc.stdout + "\n" + proc.stderr).strip()[-1600:]
        return R._check("selected_sandbox_authoring", ok, f"rc={proc.returncode}; output={output}")

    return live


@pytest.fixture
def launch(v22, tmp_path, monkeypatch):
    """The fresh launch's assessment path: snapshot-external compiler execution, V22 bundle."""
    sdk, bundle = v22
    context, te, ws = _inputs(tmp_path)
    bundle = {**bundle, "allowed": [*bundle["allowed"], {"path": te.rtl_facts_pin, "mode": "ro"}]}
    monkeypatch.setenv("MERLIN_COMPILER_EXECUTION_MODE", "snapshot_external_v1")
    # RTL check compilation needs a frozen corpus and is independent of both refusals.
    monkeypatch.setattr(R, "_frozen_rtl_checks", lambda *args, **kwargs: [])
    monkeypatch.setattr(R, "_live_probe", _author_probe(sdk))

    def verdict():
        result = R.assess(context, te, ws, bundle, TOOLS, tmp_path)
        return AA.readiness_check({**result, "bundle_id": bundle["bundle_id"], "resolved_tools": list(TOOLS)})

    return bundle, verdict


def test_v22_bundle_records_the_withheld_kernels_modules_as_denials(v22):
    _, bundle = v22
    denied = {Path(row["path"]).name for row in bundle["denied"]}
    assert {Path(name.split(":", 2)[2]).name for name in WITHHELD_GRANTS} <= denied
    assert R._cca_withheld(bundle)


def test_readiness_reproduces_the_ignored_denial_no_go(launch, monkeypatch):
    _, verdict = launch
    original = R._grant_checks
    # Before the fix the grant check never read ``denied``: the same as a bundle that records none.
    monkeypatch.setattr(R, "_grant_checks", lambda te, bundle, tools: original(te, {**bundle, "denied": []}, tools))
    check = verdict()
    assert check.status == AA.FAIL, check.cause
    assert check.cause.startswith(AA.NO_GO + ": ")
    failed = {row["name"] for row in check.detail["checks"] if not row["ok"]}
    assert failed == set(WITHHELD_GRANTS)
    assert "selected_sandbox_authoring" not in {row["name"] for row in check.detail["checks"]}


def test_readiness_reproduces_the_withheld_cca_import_no_go(launch, monkeypatch):
    _, verdict = launch
    # Before the fix the cca_spine probe imported merlin.kernels whatever the bundle withheld.
    monkeypatch.setattr(R, "_cca_withheld", lambda bundle: False)
    check = verdict()
    assert check.status == AA.FAIL, check.cause
    assert check.cause.startswith(AA.NO_GO + ": selected_sandbox_authoring: rc=1")
    assert "merlin.kernels" in check.cause
    assert all(row["ok"] for row in check.detail["checks"] if row["name"].startswith("grant:"))


def test_readiness_passes_on_v22_with_the_current_code(launch):
    _, verdict = launch
    check = verdict()
    assert check.status == AA.PASS, check.cause
    rows = {row["name"]: row for row in check.detail["checks"]}
    assert rows["selected_sandbox_authoring"]["ok"]
    assert all(rows[name]["ok"] and "recorded denial" in rows[name]["detail"] for name in WITHHELD_GRANTS)
    assert "7 withheld by recorded denial" in check.cause


def test_readiness_verdicts_other_than_ready_never_pass_silently():
    skipped = AA.readiness_check({"status": "skipped", "reason": "raw_baseline has no assisted contract"})
    assert skipped.status == AA.PASS and "skipped" in skipped.cause
    assert AA.readiness_check({"status": "unknown", "checks": []}).status == AA.FAIL
    contradictory = {"status": "ready", "checks": [{"name": "x", "ok": False, "detail": "d"}]}
    assert AA.readiness_check(contradictory).status == AA.FAIL


def test_the_check_runs_the_assessment_in_the_exports_interpreter(tmp_path, monkeypatch):
    framework = tmp_path / "export"
    (framework / "src").mkdir(parents=True)
    descriptor = tmp_path / "release" / "experiment" / "target_experiment.yaml"
    staged = {"workspace": str(tmp_path / "ws"), "descriptor": str(descriptor), "bundle": "b"}
    ctx = AA.Context(framework, tmp_path, tmp_path, staged, [], {}, None, "medium", 3, "snapshot_external_v1")
    seen = {}

    def child(script, roots, refused, *args, cwd, timeout, env):
        seen.update(script=script, roots=roots, args=args, env=env)
        failed = {"name": "selected_sandbox_authoring", "ok": False, "detail": "rc=1"}
        return subprocess.CompletedProcess([], 0, "", ""), {"status": "no_go", "checks": [failed]}

    monkeypatch.setattr(AA.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(AA, "run_child", child)
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "cas"))
    check = AA.check_tooling_readiness(ctx)
    assert check.status == AA.FAIL and check.cause == f"{AA.NO_GO}: selected_sandbox_authoring: rc=1"
    assert seen["roots"] == AA.operator_roots(framework)
    assert seen["env"]["PYTHONPATH"] == str(framework / "src")
    assert seen["env"]["MERLIN_COMPILER_EXECUTION_MODE"] == "snapshot_external_v1"
    assert seen["env"]["MERLIN_BUNDLE_CAS"] == str(tmp_path / "cas")  # the launch environment is inherited
    assert '"manifest": "' + str(descriptor.parent / "input_bundles/b/input_bundle_manifest.yaml") in seen["args"][0]
    compile(AA._PRELUDE + seen["script"], "readiness", "exec")  # the shipped source is self-contained
    assert "def assess_staged_bundle(" in seen["script"]
    assert AA.check_tooling_readiness(AA.Context(framework, None, tmp_path, {}, [], {}, None, "m", 3)).status == AA.SKIP
