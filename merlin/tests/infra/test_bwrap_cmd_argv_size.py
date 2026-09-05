"""``bwrap_cmd`` must obey the execve per-argument limit, not merely intend to.

THE FAILURE THIS PINS. The string ``run_baseline_qa_loop.bwrap_cmd`` returns is handed to
``bash -c`` as ONE argv element, and a single execve argument may not exceed ``MAX_ARG_STRLEN``
(32 pages, 128 KiB on every Linux this runs on). The answer-mask pass appends one
``--ro-bind /dev/null <path>`` per golden/expected file and one ``--tmpfs <dir>`` per hidden
directory; at the 1,172 surfaces this checkout carries those masks alone are ~167 KB. So the
inline join does not merely risk the cap, it exceeds it by 27% before any base argv, toolchain
bind, or payload — and every arm dies with ``OSError: [Errno 7] Argument list too long`` naming
``bash``, before the first round, with the sandbox never assembled.

That is exactly what ``34e0296f`` ("fix(sandbox): one composer for the argv size rule") fixed, by
routing this caller through :func:`bwrap.compose_command` — the one place a bwrap argv becomes a
shell string, which moves the bind list behind ``--args <fd>`` once it would not fit. The fix was
then LOST: a later merge took the pre-fix side of ``run_baseline_qa_loop.py``, the line reverted to
the raw ``" ".join(parts) + ...``, and nothing noticed. Readiness section F4 stayed green the whole
time because it invokes bwrap as an argv LIST (``subprocess.run(argv + [...])``) and so never
constructs the oversized single argument the live path constructs. A check that cannot fail
reported success while the launch path could not spawn.

WHY THESE TESTS AND NOT A COMMENT. A comment cannot detect a merge. The first test asserts the
composer is what produces the string (it is the only mechanism that can honour the cap); the second
asserts the actual behaviour at real corpus scale, so it fails if the composer is ever bypassed or
its threshold is raised past the kernel's.
"""
from __future__ import annotations

import os
import sys

import pytest

from merlin.common.paths import merlin_dir

_HARNESS = merlin_dir() / "experiments/capsule_bench/harness"
if str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))

#: The kernel's per-argument ceiling, derived rather than written down: MAX_ARG_STRLEN is 32 pages.
MAX_ARG_BYTES = 32 * os.sysconf("SC_PAGE_SIZE")


def _compose_source() -> str:
    import inspect

    import run_baseline_qa_loop as L
    return inspect.getsource(L.bwrap_cmd)


def test_bwrap_cmd_goes_through_the_shared_composer():
    """The live launch path must build its shell string with the composer that knows the size rule.

    Asserted on the source rather than by calling it, because assembling a real bundle snapshot needs
    a workspace this test has no business creating — and the property being pinned is precisely that
    this caller does not grow a second, unbounded join of its own.
    """
    src = _compose_source()
    assert "compose_command" in src, (
        "bwrap_cmd no longer routes through bwrap.compose_command; a raw join of the bind list "
        "exceeds MAX_ARG_STRLEN at corpus scale and every arm dies with E2BIG before round 1 "
        "(regression of 34e0296f, which a merge already reverted once)")
    assert '" ".join(parts)' not in src, (
        "bwrap_cmd joins its argv inline again — that is the exact pre-34e0296f line")


def test_composer_keeps_a_corpus_scale_mask_set_under_the_execve_limit(tmp_path):
    """At the real number of answer surfaces, the composed command must fit in one execve argument.

    This is the behavioural half: it fails if the composer is bypassed OR if its own threshold drifts
    above the kernel's. The argv is synthesised from the DECLARED answer surfaces, so the test tracks
    the corpus instead of a snapshot of it.
    """
    from merlin.targetgen.sandbox import bwrap as BW
    from merlin.targetgen.sandbox.answer_surfaces import answer_surfaces

    import run_baseline_qa_loop as L

    surfaces = answer_surfaces(L._te())
    if len(surfaces) < 200:
        pytest.skip(f"only {len(surfaces)} answer surfaces declared; the size rule is not exercised")

    argv = ["bwrap", "--dev-bind", "/", "/"]
    for s in surfaces:
        argv += (["--ro-bind", "/dev/null", str(s.path)] if s.kind == "file"
                 else ["--tmpfs", str(s.path)])
    inline_bytes = len(" ".join(argv).encode("utf-8"))
    assert inline_bytes > MAX_ARG_BYTES, (
        f"the inline join is {inline_bytes} B, under the {MAX_ARG_BYTES} B cap — this test is no "
        f"longer exercising the rule it exists for; lower the surface threshold or drop it")

    # The composer writes its args file BESIDE `ws`, so `ws` must be a throwaway directory — pointing
    # it at a repo path would leave a 160 KB dotfile in the tree on every run of this test.
    composed = BW.compose_command(argv, " bash -c 'true'", tmp_path / "ws_size_probe")
    assert len(composed.encode("utf-8")) <= MAX_ARG_BYTES, (
        f"composed command is {len(composed.encode('utf-8'))} B, over the {MAX_ARG_BYTES} B execve "
        f"per-argument limit; bash will refuse it with E2BIG")
