"""Keep worker failure causes intact in bounded capture diagnostics."""

from __future__ import annotations


def stderr_cause(stderr: str, *, limit: int = 1800) -> str:
    """The part of ``stderr`` that explains a failure, rather than whatever came last.

    A python subprocess interleaves warnings with tracebacks, so the tail of stderr is often a
    warning emitted after the real error -- which is exactly how a capture failure came to be
    attributed to a benign LSTM contiguity UserWarning. So: prefer the LAST traceback, and fall back
    to the tail only when there is none, saying which was used either way.
    """
    text = str(stderr or "")
    if not text.strip():
        return "(the worker wrote nothing to stderr)"
    marker = "Traceback (most recent call last)"
    idx = text.rfind(marker)
    if idx < 0:
        return "--- stderr tail (no traceback found) ---\n" + text[-limit:]

    # ⚠️ A `SystemExit` TRACEBACK IS NOT A DIAGNOSIS. A worker that ends `raise SystemExit(main())`
    # produces a traceback whose only frame is that line: the process CHOSE its exit code and printed
    # its reason itself, earlier in stderr. Preferring the last traceback then reports the wrapper and
    # discards the message -- measured on the m2m capture worker, where three capsules failed with
    # `rc=1` and the recorded cause was four lines of `raise SystemExit(main())`, naming nothing.
    #
    # Same shape as the warning bug this function already guards: the informative text is not where the
    # obvious rule looks. So when the tail of the traceback names SystemExit, or names no exception at
    # all, keep the tail as well rather than instead -- the traceback still locates the exit, and the
    # tail is where the worker said why.
    tail_lines = [ln for ln in text[idx:].splitlines() if ln.strip()]
    closing = tail_lines[-1].strip() if tail_lines else ""
    uninformative = closing.startswith("SystemExit") or closing.startswith("raise SystemExit")
    if uninformative:
        return (
            "--- last traceback names only SystemExit, so the worker's own message is kept too ---\n"
            + text[idx:][: limit // 3]
            + "\n--- stderr tail ---\n"
            + text[:idx][-(limit - limit // 3) :]
        )

    # ⚠️ TRUNCATE FROM THE MIDDLE, NEVER FROM THE HEAD. `text[idx:idx + limit]` keeps the header and
    # the stack frames and cuts off the LAST line -- which is the only line that names the error. A
    # deep traceback therefore recorded a failure reason containing no error at all, just frames, and
    # every diagnosis that started from one of these was slower than it needed to be. The exception
    # line is the point of a traceback, so it is the part that is guaranteed to survive.
    tb = text[idx:]
    if len(tb) <= limit:
        return "--- last traceback ---\n" + tb
    head = limit // 3
    tail = limit - head
    return (
        "--- last traceback (middle elided; the exception line is kept) ---\n"
        + tb[:head]
        + f"\n    ... {len(tb) - limit} characters of frames elided ...\n"
        + tb[-tail:]
    )
