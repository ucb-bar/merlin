"""Own-process POSIX alarm scope for a single main-thread DATA fetch.

Host signal delivery/CPython interruption are trusted; no hard real-time or
uninterruptible-kernel guarantee. Imports do not install a handler or timer.
"""
from contextlib import contextmanager
import math
import signal
import threading


class FetchDeadline(TimeoutError):
    pass


@contextmanager
def fetch_deadline(seconds):
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("fetch deadline requires own main-thread process")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 1800:
        raise ValueError("finite positive deadline at most 1800 seconds required")
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer != (0.0, 0.0):
        raise ValueError("preexisting timer; no timer takeover")
    blocked = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    if signal.SIGALRM in blocked or signal.SIGALRM in signal.sigpending():
        raise ValueError("blocked/pending SIGALRM; no signal takeover")
    previous_handler = signal.getsignal(signal.SIGALRM)
    if previous_handler not in (signal.SIG_DFL, signal.SIG_IGN) and not callable(previous_handler):
        raise ValueError("unrestorable prior SIGALRM handler; no signal takeover")

    def expired(signum, frame):
        raise FetchDeadline(f"outer fetch deadline expired after {seconds} seconds")

    primary = None
    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds, 0.0)
        yield
    except BaseException as error:
        primary = error
        raise
    finally:
        # Cancel before restoring the old handler. Never replace a body failure
        # with a cleanup failure; attach cleanup evidence to the original object.
        errors = []
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.0, 0.0)
        except BaseException as error:
            errors.append({"operation": "cancel_timer", "type": type(error).__name__, "error": str(error)})
        try:
            signal.signal(signal.SIGALRM, previous_handler)
        except BaseException as error:
            errors.append({"operation": "restore_handler", "type": type(error).__name__, "error": str(error)})
        if errors:
            if primary is not None:
                primary.deadline_cleanup_errors = errors
                primary.add_note("deadline cleanup errors: " + "; ".join(x["error"] for x in errors))
            else:
                failure = RuntimeError("deadline cleanup failed: " + "; ".join(x["error"] for x in errors))
                failure.deadline_cleanup_errors = errors
                raise failure
