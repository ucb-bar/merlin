"""A tool an agent is never told about cannot be measured.

An experiment arm's treatment is the tools it grants. If the arm's generated documentation lists the
PATHS it may read but never says what the tool answers or how to invoke it, a null result cannot
distinguish "the tool does not help" from "the agent never knew it was there" — and that ambiguity
destroys the arm-to-arm comparison the whole experiment exists to make.

Measured 2026-09-05: the freshly generated verify-arm STARTER_PROMPT was byte-identical to arm-4's,
and `ALLOWED_MERLIN_TOOLS.md` rendered only `ToolSpec.note` — a policy label like "ALLOWED (verify
arm): the compiler-verification seam" — while `ToolSpec.blurb`, which carries the invocation and the
exit-code contract, was rendered nowhere. The arm's single treatment was bound into the sandbox and
unadvertised.
"""
from __future__ import annotations

import pytest


def test_a_granted_tools_blurb_reaches_the_agent_facing_doc():
    from merlin.targetgen.generate_bundles import _allowed_merlin_tools_doc
    from merlin.targetgen.tool_registry import TOOLS

    name = "verify_seam"
    spec = TOOLS.get(name)
    if spec is None:
        pytest.skip("verify_seam is not registered in this checkout")

    doc = _allowed_merlin_tools_doc(
        {"bundle_id": "b", "arm": "merlin_verify", "allowed": [], "denied": [], "tools": [name]})
    assert name in doc, "the doc does not name the granted tool"
    assert spec.blurb.split(".")[0] in doc, (
        "the tool's blurb — what it answers and how to invoke it — is not in the agent-facing doc; "
        "the arm's treatment would be unadvertised and a null result uninterpretable")


def test_the_manifest_records_which_tools_it_granted():
    """tools.txt and the manifest must not be two half-descriptions of the same grant."""
    from merlin.targetgen.generate_bundles import _allowed_merlin_tools_doc

    doc = _allowed_merlin_tools_doc(
        {"bundle_id": "b", "arm": "a", "allowed": [], "denied": [], "tools": []})
    assert "predates tool recording" in doc, (
        "a manifest with no tools list must say so, not render an empty section that reads as "
        "'this arm grants nothing'")


def test_every_tool_blurb_names_an_invocation_that_exists():
    """A blurb telling an agent to run a console script absent from the venv is a broken instruction.

    Measured: this checkout has ZERO `merlin-*` console scripts — the package is used via PYTHONPATH,
    so `merlin-verify ...` is `command not found` even though pyproject declares it.
    """
    import shutil

    from merlin.targetgen.tool_registry import TOOLS

    for name, spec in sorted(TOOLS.items()):
        blurb = spec.blurb or ""
        for token in blurb.replace("`", " ").split():
            if not token.startswith("merlin-"):
                continue
            script = token.rstrip(".,;:")
            if shutil.which(script) is None:
                pytest.fail(
                    f"tool {name!r} tells the agent to run {script!r}, which is not on PATH in this "
                    f"environment. Use `python -m <module>` instead, or install the console scripts.")
