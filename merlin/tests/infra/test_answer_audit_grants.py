"""The answer-access audit must respect the input bundle's OWN grants.

A bundle grants and denies at MODULE granularity inside a single package: the arm-4 gemmini bundle
hands the agent ``merlin/python/merlin/runtime/commandbuffer.py`` and ``.../runtime/tensor.py`` while
withholding ``.../runtime/reference.py`` and ``.../runtime/simulator.py``.  A package-PREFIX substring
test (``"from merlin.runtime" in text``) cannot tell those apart, so it recorded ``oracle_use`` -- and
an unclean, unwaivable answer-access verdict -- for an agent importing a module it was given.  The same
asymmetry bit the path audit: a withheld GRADER module is identified by a bare stem, which also matches
that stem used as a grep PATTERN over a granted source file.

These tests pin both directions: what the bundle grants is clean, what it (or the declared oracle
registry) denies is still a violation.  Bundles are synthesised in ``tmp_path`` so the assertions are
about the DERIVATION, not about any one target's shipped grant list; a final test then checks that a
real shipped bundle exhibiting the grant/deny split behaves the same way.
"""
from __future__ import annotations

import json
import sys

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen.sandbox.answer_surfaces import ORACLE_MODULES, module_name_for

_HARNESS = merlin_dir() / "experiments/capsule_bench/harness"
_DESCRIPTOR = merlin_dir() / "experiments/capsule_bench/targets/gemmini/target_experiment.yaml"

# The two package-siblings the split is about, DERIVED rather than spelled out: the denied one is the
# first declared oracle module that is an importable module inside a package, and the granted one is a
# module in that same package that the registry does NOT name.
_DENIED_REL = next(rel for rel in ORACLE_MODULES if rel.endswith(".py"))
_DENIED_MOD = module_name_for(_DENIED_REL)
_PACKAGE = _DENIED_MOD.rsplit(".", 1)[0]
_GRANTED_MOD = f"{_PACKAGE}.commandbuffer"
_GRANTED_REL = "merlin/python/" + _GRANTED_MOD.replace(".", "/") + ".py"


@pytest.fixture()
def harness(monkeypatch):
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(_DESCRIPTOR))
    if str(_HARNESS) not in sys.path:
        sys.path.insert(0, str(_HARNESS))
    import run_baseline_qa_loop as L  # noqa: PLC0415
    return L


def _bundle(tmp_path, allowed=(), denied=()):
    """A minimal input bundle: just the two grant lists the audit derives its policy from."""
    d = tmp_path / "bundle"
    d.mkdir(exist_ok=True)
    (d / "allowed_files.txt").write_text("\n".join(allowed) + "\n")
    (d / "denied_files.txt").write_text("\n".join(denied) + "\n")
    return str(d)


def _arm4_bundle(tmp_path):
    return _bundle(tmp_path, allowed=[_GRANTED_REL], denied=[_DENIED_REL])


def _transcript(tmp_path, cmd, stdout="ok", tid="t1", tool="Bash", extra_input=None):
    inp = {"command": cmd} if tool == "Bash" else (extra_input or {"file_path": cmd})
    lines = [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": tid, "name": tool, "input": inp}]}},
        {"type": "user", "tool_use_result": {"stdout": stdout},
         "message": {"content": [{"type": "tool_result", "tool_use_id": tid,
                                  "content": stdout or "(Bash completed with no output)"}]}},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(line) for line in lines))
    return p


def _audit(harness, tmp_path, cmd, bundle, **kw):
    tp = _transcript(tmp_path, cmd, **kw)
    return harness.audit_transcript(tp, arm="merlin_assisted", bundle=bundle)


# --------------------------------------------------------------------- GRANTED modules stay clean
@pytest.mark.parametrize("stmt", [
    "from {granted} import dataflow_operands, validate_command_buffer",
    "from {package} import commandbuffer",
    "import {granted}",
    "import {granted} as cb",
])
def test_importing_a_granted_sibling_is_not_oracle_use(harness, tmp_path, stmt):
    body = stmt.format(granted=_GRANTED_MOD, package=_PACKAGE)
    res = _audit(harness, tmp_path, f'python3 -c "\nimport json\n{body}\nprint(1)\n"',
                 _arm4_bundle(tmp_path))
    assert res["hits"] == []
    assert res["clean"] is True


def test_importing_the_root_package_is_a_self_scan_not_oracle_use(harness, tmp_path):
    """``import merlin`` is how an agent inspects its own environment; every denied module is a
    descendant of it, so treating it as oracle use flags every integrity self-scan."""
    res = _audit(harness, tmp_path, 'python3 -c "import merlin, os; print(merlin.__file__)"',
                 _arm4_bundle(tmp_path))
    assert res["clean"] is True
    assert res["hits"] == []


# ------------------------------------------------------------------ DENIED modules still flagged
@pytest.mark.parametrize("stmt", [
    "from {denied} import golden_for",
    "from {package} import reference",
    "import {denied}",
    "import json, {denied}",
])
def test_importing_a_denied_oracle_module_is_still_oracle_use(harness, tmp_path, stmt):
    body = stmt.format(denied=_DENIED_MOD, package=_PACKAGE,
                       reference=_DENIED_MOD.rsplit(".", 1)[-1])
    body = body.replace("import reference", f"import {_DENIED_MOD.rsplit('.', 1)[-1]}")
    res = _audit(harness, tmp_path, f'python3 -c "\n{body}\n"', _arm4_bundle(tmp_path))
    assert res["clean"] is False
    assert [h["kind"] for h in res["hits"]] == ["oracle_use"]


def test_importing_the_package_that_holds_the_oracle_is_still_flagged(harness, tmp_path):
    """A bare sub-package import resolves to nothing more specific but names the package the oracle
    lives in -- it stays a violation (the pre-existing strictness this fix must not relax)."""
    res = _audit(harness, tmp_path, f'python3 -c "import {_PACKAGE}"', _arm4_bundle(tmp_path))
    assert res["clean"] is False


def test_a_bundle_denied_module_outside_the_registry_is_oracle_use(harness, tmp_path):
    """The bundle's own deny list adds to the declared registry, so the audit tracks the grant."""
    denied_extra = "merlin/python/merlin/targetgen/generate/runtime_adapter.py"
    bundle = _bundle(tmp_path, allowed=[_GRANTED_REL], denied=[_DENIED_REL, denied_extra])
    res = _audit(harness, tmp_path,
                 f'python3 -c "from {module_name_for(denied_extra)} import adapt"', bundle)
    assert res["clean"] is False
    assert res["hits"][0]["token"] == module_name_for(denied_extra)


def test_the_declared_registry_outranks_a_mis_authored_grant(harness, tmp_path):
    """A bundle that erroneously ALLOWS an oracle module cannot launder it: the declared registry is
    the harness's own identity of 'the oracle' and wins over any per-arm grant."""
    bundle = _bundle(tmp_path, allowed=[_GRANTED_REL, _DENIED_REL], denied=[])
    res = _audit(harness, tmp_path, f'python3 -c "import {_DENIED_MOD}"', bundle)
    assert res["clean"] is False


def test_a_bundle_without_grant_lists_still_flags_the_declared_oracle(harness, tmp_path):
    """Fail closed: a legacy bundle with no allowed/denied files falls back to the declared registry."""
    res = _audit(harness, tmp_path, f'python3 -c "import {_DENIED_MOD}"',
                 str(tmp_path / "no_such_bundle"))
    assert res["clean"] is False


def test_authored_python_importing_the_oracle_is_still_flagged(harness, tmp_path):
    """Write/Edit of a .py file is audited the same way as inline python."""
    tp = _transcript(tmp_path, "", tool="Write",
                     extra_input={"file_path": "submission/selfgrade.py",
                                  "content": f"import json\nfrom {_DENIED_MOD} import golden_for\n"})
    res = harness.audit_transcript(tp, arm="merlin_assisted", bundle=_arm4_bundle(tmp_path))
    assert res["clean"] is False
    assert [h["kind"] for h in res["hits"]] == ["oracle_use"]


def test_authored_python_importing_a_granted_sibling_is_clean(harness, tmp_path):
    tp = _transcript(tmp_path, "", tool="Write",
                     extra_input={"file_path": "submission/emit.py",
                                  "content": f"from {_GRANTED_MOD} import dataflow_operands\n"})
    res = harness.audit_transcript(tp, arm="merlin_assisted", bundle=_arm4_bundle(tmp_path))
    assert res["clean"] is True
    assert res["hits"] == []


def test_an_oracle_CALL_is_still_flagged_without_any_import(harness, tmp_path):
    """The call tokens are independent of the grant lists -- calling the oracle is use, however it got
    into scope."""
    res = _audit(harness, tmp_path, 'python3 -c "print(reference_outputs(cb))"',
                 _arm4_bundle(tmp_path))
    assert res["clean"] is False


# ------------------------------------------------ a grader STEM used as a grep pattern is advisory
def test_grep_pattern_over_a_granted_file_is_not_a_path_read(harness, tmp_path):
    """MEASURED (merlincirct_g4p1_20260905 round 1): grepping a GRANTED source for a token that happens
    to be a withheld module's stem was recorded as a content read of a withheld path."""
    grader_stem = next(t for t in harness._GRADER_TOKENS if "/" not in t and "." not in t)
    cmd = f'grep -n "{grader_stem}\\|pad\\|DIM" commandbuffer.py | head -60'
    res = _audit(harness, tmp_path, cmd, _arm4_bundle(tmp_path), stdout="24:POOL_ATTR_ARITY = {}")
    assert res["clean"] is True
    assert [h["kind"] for h in res["hits"]] == ["pattern_mention"]


def _tool_granted_but_token_flagged(harness):
    """Shipped bundles that GRANT a file whose path carries one of the coarse merlin-tool tokens.

    ``_path_tokens`` flags those tokens for every arm whose NAME is not ``merlin_assisted`` -- but the
    ``cpp_merlininfra`` bundles do grant ``merlin/python/merlin/targetgen/generate/*.py``. So the coarse
    arm-name gate accuses those arms of reading a withheld tool they were in fact handed. Derived by
    walking the bundles, so this stays true as the ladder changes."""
    out = []
    targets = merlin_dir() / "experiments/capsule_bench/targets"
    for allowed_f in sorted(targets.glob("*/input_bundles/*/allowed_files.txt")):
        if "merlin_assisted" in allowed_f.parent.name:
            continue
        arm = allowed_f.parent.name.rsplit("_", 2)[0]
        for line in allowed_f.read_text().splitlines():
            rel = line.strip()
            if not rel or rel.endswith("/"):
                continue
            if any(tok in rel for tok in harness._MERLIN_TOOL_TOKENS):
                out.append(pytest.param(str(allowed_f.parent), arm, rel,
                                        id=f"{allowed_f.parent.name}-{rel.rsplit('/', 1)[-1]}"))
    return out


def test_some_shipped_bundle_grants_a_tool_the_arm_gate_would_flag(harness):
    """Guard the guard: if no bundle exhibits the mismatch any more, the cases below are vacuous."""
    assert _tool_granted_but_token_flagged(harness), "no bundle exercises the tool-grant mismatch"


def test_reading_a_granted_tool_file_is_advisory_not_a_violation(harness, tmp_path):
    """MEASURED-shape case: a non-``merlin_assisted`` arm reading a generator file its OWN bundle
    grants was recorded as a content read of a withheld tool. Every shipped instance must be advisory."""
    cases = _tool_granted_but_token_flagged(harness)
    assert cases, "no bundle exercises the tool-grant mismatch"
    for case in cases:
        bundle_dir, arm, rel = case.values
        tp = _transcript(tmp_path, f"cat {rel}", stdout="def emit(): ...")
        res = harness.audit_transcript(tp, arm=arm, bundle=bundle_dir)
        assert res["clean"] is True, f"{arm} reading its granted {rel} was flagged"
        assert [h["kind"] for h in res["hits"]] == ["granted_read"], (arm, rel, res["hits"])


def test_the_same_arm_reading_an_UNGRANTED_tool_is_still_a_violation(harness, tmp_path):
    """The protection this must not weaken: the arm-policy tokens still bite for a tool path the
    bundle does NOT grant."""
    cases = _tool_granted_but_token_flagged(harness)
    bundle_dir, arm, rel = cases[0].values
    sibling = rel.rsplit("/", 1)[0] + "/definitely_not_granted.py"
    tp = _transcript(tmp_path, f"cat {sibling}", stdout="def emit(): ...")
    res = harness.audit_transcript(tp, arm=arm, bundle=bundle_dir)
    assert res["clean"] is False
    assert any(h["kind"] == "path_read" for h in res["hits"])


def test_a_real_answer_file_read_is_still_a_violation(harness, tmp_path):
    """The protection this fix must not weaken: a content read of a golden that RETURNED data."""
    res = _audit(harness, tmp_path, "cat isa/AT0_config_smoke/golden.yaml",
                 _arm4_bundle(tmp_path), stdout="expected_out: [1.0, 2.0]")
    assert res["clean"] is False
    assert any(h["kind"] == "path_read" for h in res["hits"])


def test_a_granted_directory_never_exempts_a_masked_golden_inside_it(harness, tmp_path):
    """The broad ``merlin/contract/`` grant is exactly the bind that re-exposes goldens; only FILE
    grants may exempt a read, so a golden under a granted DIR is still a violation."""
    bundle = _bundle(tmp_path, allowed=["merlin/contract/"], denied=[_DENIED_REL])
    res = _audit(harness, tmp_path, "cat merlin/contract/capsules/isa/X/golden.yaml", bundle,
                 stdout="expected_out: [7]")
    assert res["clean"] is False
    assert any(h["kind"] == "path_read" for h in res["hits"])


# ------------------------------------------------------------------- the real, shipped grant lists
def _bundles_with_a_split():
    """Every shipped bundle that grants a module in a package whose sibling it denies -- the exact
    configuration the package-prefix test could not represent. Derived by walking the bundles."""
    out = []
    targets = merlin_dir() / "experiments/capsule_bench/targets"
    for allowed_f in sorted(targets.glob("*/input_bundles/*/allowed_files.txt")):
        denied_f = allowed_f.with_name("denied_files.txt")
        if not denied_f.is_file():
            continue
        allowed = [module_name_for(ln.strip()) for ln in allowed_f.read_text().splitlines()]
        denied = [module_name_for(ln.strip()) for ln in denied_f.read_text().splitlines()]
        for g in [m for m in allowed if m]:
            for d in [m for m in denied if m]:
                if g != d and g.rsplit(".", 1)[0] == d.rsplit(".", 1)[0]:
                    out.append(pytest.param(str(allowed_f.parent), g, d,
                                            id=f"{allowed_f.parent.name}-{g.rsplit('.', 1)[-1]}"))
    return out


@pytest.mark.parametrize("bundle_dir, granted, denied", _bundles_with_a_split())
def test_shipped_bundles_split_grant_and_deny_in_one_package(harness, tmp_path, bundle_dir,
                                                             granted, denied):
    clean = _audit(harness, tmp_path, f'python3 -c "from {granted} import thing"', bundle_dir)
    assert clean["clean"] is True, f"{granted} is granted by {bundle_dir} but was flagged"
    dirty = _audit(harness, tmp_path, f'python3 -c "from {denied} import thing"', bundle_dir)
    assert dirty["clean"] is False, f"{denied} is denied by {bundle_dir} but was not flagged"
