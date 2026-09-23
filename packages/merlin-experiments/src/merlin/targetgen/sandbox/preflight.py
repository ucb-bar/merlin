"""Is the isolation sandbox actually USABLE on this host — a named condition, not a `which` hit.

WHY THIS EXISTS. Every refusal in this repo that guards an unsandboxed agent run asks
``shutil.which("bwrap") is None``. That question is about a FILE, and the property the refusal needs is
about an EXECUTION. The two came apart on this host: ``/usr/bin/bwrap`` is present and on ``PATH`` (so
``which`` succeeds and no refusal fires) but it is not setuid and
``kernel.apparmor_restrict_unprivileged_userns`` is 1, so every invocation dies at
``bwrap: setting up uid map: Permission denied`` before the inner command starts.

The consequence is a silent-failure hazard, not merely a missing guard. A run launched that way
proceeds, every agent round exits nonzero because its whole command was the bwrap wrapper, and the
harness reports the round outcome it has a name for — "no seed reached" — which reads as *the model
could not write a correct kernel* rather than *the sandbox never started*. A campaign is on record as
having died exactly this way with zero treatment rounds ever begun, and nothing in its artifacts said
so. Two other harnesses in this repo already classify this exact stderr as a harness fault after the
fact; this module makes it detectable BEFORE the run, and gives the condition a name.

THE CONDITIONS ARE DISTINGUISHABLE ON PURPOSE. "not installed", "installed but cannot execute" and
"could not be determined" are three different operator actions (install it; fix host policy or use a
different containment mechanism; investigate) and three different things to record in a run artifact.
Collapsing them into a boolean is what let the second one masquerade as the first for as long as it
did.

FAIL CLOSED. :attr:`SandboxProbe.usable` is true for exactly one status. A probe that timed out, raised,
or produced output this module does not understand is :data:`SANDBOX_UNKNOWN`, and UNKNOWN is not usable
— a sandbox that has not been shown to work has not been shown to work. Callers that must not run
unsandboxed use :func:`require_working_sandbox`, which raises on anything but OK.

This module never changes a host setting and never tries to work around one; it observes and reports.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

#: The sandbox binary is installed and a trivial command ran inside it.
SANDBOX_OK = "ok"
#: No such binary on ``PATH`` (or at the explicit path given).
SANDBOX_ABSENT = "absent"
#: The binary exists but cannot construct a sandbox on this host.
SANDBOX_INOPERABLE = "inoperable"
#: The probe itself could not be carried out. NOT a pass.
SANDBOX_UNKNOWN = "unknown"

#: The one status that permits an isolated run. Anything else — including UNKNOWN — does not.
_USABLE_STATUS = SANDBOX_OK

#: Substrings of the binary's own stderr that IDENTIFY a known inoperable cause, mapped to a stable
#: reason code. These are diagnostic refinements only: a nonzero exit is inoperable whether or not its
#: stderr is recognised, so an unrecognised message can never be mistaken for success. They are
#: properties of the sandbox tool and the kernel, not of any hardware target.
_INOPERABLE_CAUSES: tuple[tuple[str, str], ...] = (
    ("setting up uid map", "userns_uid_map_denied"),
    ("Creating new namespace failed", "userns_creation_denied"),
    ("No permissions to creating new namespace", "userns_creation_denied"),
    ("loopback: Failed RTM_NEWADDR", "netns_denied"),
    ("Operation not permitted", "operation_not_permitted"),
    ("Permission denied", "permission_denied"),
)

#: What the probe runs inside the sandbox. Deliberately the cheapest possible true-returning program,
#: so a failure is attributable to sandbox CONSTRUCTION rather than to the payload.
_PROBE_PAYLOAD: tuple[str, ...] = ("/bin/true",)


def _probe_argv(binary: str) -> list[str]:
    """A minimal but REPRESENTATIVE sandbox: the same operation classes the real argv uses.

    It is not enough to ask the binary for its version — a version print does not unshare anything, so
    it succeeds on precisely the host where the real sandbox cannot be built. The probe therefore
    performs the operations whose denial is the observed failure: a user namespace with a uid map
    (implied by any bwrap invocation), read-only binds of the system tree, a tmpfs, and a fresh
    ``/proc``. ``--ro-bind-try`` is used for the paths that legitimately vary between distributions, so
    an absent ``/lib64`` is not read as a broken sandbox.
    """
    argv = [binary, "--die-with-parent", "--unshare-pid"]
    for required in ("/usr", "/bin"):
        argv += ["--ro-bind", required, required]
    for optional in ("/lib", "/lib64", "/etc"):
        argv += ["--ro-bind-try", optional, optional]
    argv += ["--tmpfs", "/tmp", "--proc", "/proc", "--dev", "/dev", "--"]
    argv += list(_PROBE_PAYLOAD)
    return argv


@dataclass(frozen=True)
class SandboxProbe:
    """What was observed about the sandbox binary, in a form a run artifact can record verbatim."""

    status: str
    reason: str
    binary: str | None = None
    returncode: int | None = None
    stderr: str = ""
    argv: tuple[str, ...] = field(default_factory=tuple)

    @property
    def usable(self) -> bool:
        """Whether an isolated run may proceed. True for OK ONLY — UNKNOWN is never a pass."""
        return self.status == _USABLE_STATUS

    def describe(self) -> str:
        """A one-line operator-facing explanation naming the condition and its cause."""
        head = f"sandbox {self.status} ({self.reason})"
        if self.binary:
            head += f" [{self.binary}]"
        if self.returncode is not None:
            head += f" rc={self.returncode}"
        tail = " ".join(self.stderr.split())
        return f"{head}: {tail[:400]}" if tail else head

    def as_record(self) -> dict[str, object]:
        """The block a run artifact embeds, so "was this run sandboxed?" is answerable afterwards."""
        return {
            "status": self.status,
            "reason": self.reason,
            "usable": self.usable,
            "binary": self.binary,
            "returncode": self.returncode,
            "stderr": self.stderr[:2000],
            "probe_argv": list(self.argv),
        }


class SandboxUnavailable(RuntimeError):
    """Raised when a caller requires a working sandbox and there is not one.

    Carries the probe so the caller records WHICH condition refused it, rather than a bare message.
    """

    def __init__(self, probe: SandboxProbe, context: str = "") -> None:
        prefix = f"{context}: " if context else ""
        super().__init__(f"{prefix}{probe.describe()}")
        self.probe = probe


#: The binary this repo's sandbox is built on. Named once here rather than spelled at every call site.
SANDBOX_BINARY_NAME = "bwrap"

_CACHE: dict[tuple[str | None, tuple[str, ...]], SandboxProbe] = {}


def sandbox_failure_reason(stderr: str) -> str | None:
    """A reason code when ``stderr`` carries a known "the sandbox could not be built" message, else None.

    Exposed so a caller that has already RUN the sandbox and got a nonzero exit can classify it with the
    same table the preflight probe uses, rather than keeping its own partial list of messages. A partial
    list is how the condition goes unrecognised: the one detector that existed matched only "Operation
    not permitted", while the message on a host with unprivileged user namespaces restricted is
    "setting up uid map: Permission denied" — so the failure was never classified at all.
    """
    for needle, code in _INOPERABLE_CAUSES:
        if needle in stderr:
            return code
    return None


def _classify_stderr(stderr: str) -> str:
    return sandbox_failure_reason(stderr) or "nonzero_exit"


def probe_sandbox(
    binary: str | Path | None = None,
    *,
    timeout: float = 20.0,
    use_cache: bool = True,
) -> SandboxProbe:
    """Determine whether a sandbox can actually be CONSTRUCTED on this host, right now.

    Runs a trivial command inside a minimal sandbox and reports the outcome as one of the four named
    conditions. Cached per (binary, argv) for the life of the process, because the answer is a host
    property and the probe forks — pass ``use_cache=False`` to force a fresh observation.
    """
    resolved = str(binary) if binary is not None else shutil.which(SANDBOX_BINARY_NAME)
    if not resolved:
        return SandboxProbe(status=SANDBOX_ABSENT, reason="binary_not_on_path", binary=None)
    if not Path(resolved).exists():
        return SandboxProbe(status=SANDBOX_ABSENT, reason="binary_missing", binary=resolved)

    argv = _probe_argv(resolved)
    key = (resolved, tuple(argv))
    if use_cache and key in _CACHE:
        return _CACHE[key]

    try:
        completed = subprocess.run(  # noqa: S603 — fixed argv, no shell
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
    except subprocess.TimeoutExpired:
        probe = SandboxProbe(status=SANDBOX_UNKNOWN, reason="probe_timed_out", binary=resolved, argv=tuple(argv))
    except OSError as exc:
        # The binary is there but could not be executed at all (no exec permission, bad interpreter,
        # wrong architecture). That is not "absent" and it is not a sandbox that works.
        probe = SandboxProbe(
            status=SANDBOX_INOPERABLE,
            reason="binary_not_executable",
            binary=resolved,
            stderr=str(exc),
            argv=tuple(argv),
        )
    else:
        stderr = completed.stderr or ""
        if completed.returncode == 0:
            probe = SandboxProbe(
                status=SANDBOX_OK,
                reason="probe_command_ran",
                binary=resolved,
                returncode=0,
                stderr=stderr,
                argv=tuple(argv),
            )
        else:
            probe = SandboxProbe(
                status=SANDBOX_INOPERABLE,
                reason=_classify_stderr(stderr),
                binary=resolved,
                returncode=completed.returncode,
                stderr=stderr,
                argv=tuple(argv),
            )
    if use_cache:
        _CACHE[key] = probe
    return probe


def require_working_sandbox(
    binary: str | Path | None = None, *, context: str = "", timeout: float = 20.0
) -> SandboxProbe:
    """Return the probe if the sandbox works; otherwise raise :class:`SandboxUnavailable`.

    The raising form exists so a caller cannot accidentally treat a falsy-but-not-OK probe as usable.
    """
    probe = probe_sandbox(binary, timeout=timeout)
    if not probe.usable:
        raise SandboxUnavailable(probe, context)
    return probe


def reset_probe_cache() -> None:
    """Forget cached observations (tests, and an operator who has just changed host policy)."""
    _CACHE.clear()


def _main() -> int:
    """``python -m merlin.targetgen.sandbox.preflight`` — is the sandbox usable here, right now?

    Exists for the hand-off this module was written for. The fix for an INOPERABLE host is an OPS
    action (a host policy change), and the person who makes it is not the person who reads the run
    artifact; asking them to confirm it worked should not require writing a snippet. Exit status is the
    answer -- 0 usable, 1 not -- so it also drops straight into a launch script, and the reason code is
    printed so a failure names a condition rather than a mood.
    """
    probe = probe_sandbox(use_cache=False)
    print(probe.describe())
    return 0 if probe.usable else 1


if __name__ == "__main__":  # pragma: no cover — operator entry point
    raise SystemExit(_main())
