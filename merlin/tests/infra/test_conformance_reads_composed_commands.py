"""Composed shell commands must not hide honest arm-4 evidence, or invent a false edit boundary.

Agents batch work: a redirect, a heredoc, several commands in one Bash call. `_executable` returns
None the moment it sees a shell control operator, so every such call yielded NO evidence -- and, for
`_submission_mutation`, "cannot be proved read-only" was read as "is the first edit", which pinned
the discovery boundary at call 0 and made arm4_discovery_before_submission_mutation unsatisfiable
however well the agent behaved. Measured on the atlas arm-4 run of 2026-09-07.

Anti-forgery is the invariant these tests protect: decomposition must credit REAL invocations only,
never a quoted string.
"""

from __future__ import annotations

from merlin_experiments.phase1 import conformance as CF


def _call(command: str, *, result: str = "", ok: bool = True) -> CF.ToolCall:
    return CF.ToolCall(
        name="Bash", input={"command": command}, tool_use_id="t0", result_present=True, succeeded=ok, result_text=result
    )


def _bijection(call) -> bool:
    return CF._cca_evidence(call, script="cca_contract.py", subcommand="check-bijection", api="check_bijection")


def test_a_redirect_does_not_hide_the_cca_invocation():
    call = _call('/bin/bash -lc "python3 cca_contract.py check-bijection atlas > /tmp/b.json"')
    assert _bijection(call)


def test_a_command_after_a_heredoc_is_still_seen():
    # The heredoc body merges following commands into one bogus segment unless we split on newlines.
    call = _call("/bin/bash -lc \"python3 - <<'PY'\nprint(1)\nPY\npython3 cca_contract.py check-bijection atlas\"")
    assert _bijection(call)


def test_a_quoted_string_is_not_evidence():
    """The whole point of structural resolution: echo cannot forge an invocation.

    The argv must be spelled so that ONLY the interpreter check can reject it -- `echo python3 ...`
    would fail merely because argv[0] is `python3` rather than the script, which passes even if the
    interpreter check is deleted. Here argv[0] IS the script, so the executable is the only guard.
    """
    call = _call('/bin/bash -lc "echo cca_contract.py check-bijection atlas"')
    assert not _bijection(call)


def test_listing_the_submission_is_not_an_edit():
    """An agent's first call is typically a read-only survey; it must not become the edit boundary."""
    call = _call("/bin/bash -lc \"pwd && rg --files -g 'submission/**' | sort\"")
    assert not CF._submission_mutation(call)


def test_a_composed_write_is_still_an_edit():
    call = _call('/bin/bash -lc "echo hi && cp seed.mlir submission/kernel.mlir"')
    assert CF._submission_mutation(call)


def test_a_redirect_into_the_submission_is_still_an_edit():
    call = _call('/bin/bash -lc "python3 gen.py > submission/kernel.mlir"')
    assert CF._submission_mutation(call)


def test_an_unresolvable_command_touching_the_submission_stays_the_boundary():
    """Nothing unprovable is waved through: if no segment resolves, it is still the boundary."""
    call = _call('/bin/bash -lc "$MYSTERY submission/kernel.mlir"')
    assert CF._submission_mutation(call)
