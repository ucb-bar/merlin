"""Which authored source tree one self-check request grades.

``submission/`` is the only tree the harness grades officially: the background grader, the freeze and
the formal grade read it and nothing else. An author may also self-check an alternative candidate kept
at ``variants/<name>/`` in the same workspace, with the same layout as ``submission/``, so that several
approaches can be checked side by side while the official one stays untouched. The winner is adopted by
copying it into ``submission/``, where the official grade then grades those bytes.

A variant request is served exactly like a ``submission/`` request: its tree is digested, copied into a
private snapshot, re-verified against the digest and graded by the same self-check under the same
sandbox, masking and tier rules. What differs is only what a result may be used for:

* every response names the tree it graded (``selfcheck_submission``), and a variant's response also
  carries a ``variant`` block with its name and digest and ``official: false``;
* a variant result triggers no promotion, freezes no promotion source and publishes no foreground
  marker, so it can never enter the promotion ledger that convergence and certificates are read from;
* exact-byte reuse still applies, because it is keyed by bytes rather than by tree: the simulator
  certificate store keys a verdict by the emitted ELF bytes, the capsule's stimulus and policy, the
  compiler profile, the engine and the grading-path instrument
  (``merlin.targetgen.simulation_certificate``; ``tier_cache`` likewise keys by execution identity).
  A variant whose emitted program for a capsule is byte-identical to ``submission/``'s produces the
  same key, so either side may carry that verdict; any differing byte is a different key.

The tree is opened component by component without following symlinks and read through the opened
directory, so a ``variants`` entry swapped for a symlink between the check and the copy cannot redirect
the broker -- which runs outside the sandbox -- to a host path.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import string
from collections.abc import Callable, Iterator
from pathlib import Path

SUBMISSION = "submission"
VARIANTS = "variants"
#: Longest accepted variant name. Names are labels, never operator paths.
NAME_MAX = 64
_NAME_FIRST = frozenset(string.ascii_letters + string.digits)
_NAME_CHARS = _NAME_FIRST | frozenset("._-")
#: Tells the self-check worker which tree it grades, for its operator-side telemetry line.
LABEL_ENV = "MERLIN_SELFCHECK_SUBMISSION_LABEL"
#: Operator bound on concurrent ordinary variant requests (default: half the ordinary lane).
VARIANT_JOBS_ENV = "MERLIN_SELFCHECK_VARIANT_JOBS"

VARIANT_NOTE = (
    "SELF-CHECK OF A VARIANT, not of submission/. Only submission/ is graded officially: this result is "
    "never counted toward convergence, certifies nothing for submission/ and starts no promotion. To adopt "
    "this variant, copy its files into submission/ so the official grade grades those bytes."
)


def valid_name(name: object) -> bool:
    """A variant name: one path component of letters, digits, ``.``, ``_`` or ``-``, starting alphanumeric."""
    return (
        isinstance(name, str)
        and 0 < len(name) <= NAME_MAX
        and name[0] in _NAME_FIRST
        and all(character in _NAME_CHARS for character in name)
    )


def request_label(request: dict) -> str:
    """The tree a request names: ``submission`` (also when absent) or ``variants/<name>``.

    Anything else -- another workspace path, an absolute path, ``..``, a nested or empty name -- is
    refused rather than normalised, so a request can never name an operator path.
    """
    raw = request.get("submission", SUBMISSION) if isinstance(request, dict) else SUBMISSION
    if raw == SUBMISSION:
        return SUBMISSION
    parent, slash, name = raw.partition("/") if isinstance(raw, str) else ("", "", "")
    if parent != VARIANTS or not slash or not valid_name(name):
        raise ValueError(
            "self-check source must be 'submission' or 'variants/<name>' (name: letters, digits, '.', '_', "
            f"'-', at most {NAME_MAX} characters, starting with a letter or digit)"
        )
    return f"{VARIANTS}/{name}"


def is_variant_request(request_bytes: bytes) -> bool:
    """Whether a raw request names anything but ``submission`` (refused names count: they are not it)."""
    try:
        request = json.loads(request_bytes)
    except (ValueError, TypeError):
        return False
    return isinstance(request, dict) and request.get("submission", SUBMISSION) != SUBMISSION


@contextlib.contextmanager
def pinned_source(ws: Path, label: str) -> Iterator[Path]:
    """Yield a path bound to the selected directory, opened without following any symlink.

    Each component below the workspace is opened with ``O_NOFOLLOW``, and the yielded
    ``/proc/self/fd/<n>`` path names the opened directory itself, so later reads cannot be redirected
    by renaming or replacing a component. Inner symlinks are the caller's to refuse.
    """
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    opened: list[int] = []
    try:
        opened.append(os.open(Path(ws), os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC))
        for part in label.split("/"):
            try:
                opened.append(os.open(part, flags, dir_fd=opened[-1]))
            except FileNotFoundError:
                raise FileNotFoundError(f"self-check source {label}/ does not exist in the workspace") from None
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise ValueError(
                        f"self-check source contains a symlink: {label}/ must be a real directory in the workspace"
                    ) from None
                raise
        yield Path(f"/proc/self/fd/{opened[-1]}")
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)


def source_digest(ws: Path, label: str, digest: Callable[[Path], str]) -> str:
    """Digest of the selected tree. ``submission`` keeps its historical digest of the live path."""
    if label == SUBMISSION:
        return digest(Path(ws) / SUBMISSION)
    with pinned_source(ws, label) as source:
        return digest(source)


def response_fields(request: dict, sha256: str | None) -> dict:
    """What every response states about the tree it graded, including a refused one's request."""
    raw = request.get("submission", SUBMISSION) if isinstance(request, dict) else SUBMISSION
    label = raw[:200] if isinstance(raw, str) and raw else SUBMISSION
    fields: dict = {"selfcheck_submission": label}
    if label != SUBMISSION:
        name = label.partition("/")[2]
        fields["official"] = False
        fields["variant"] = {
            "name": name if valid_name(name) else None,
            "path": label,
            "sha256": sha256,
            "official": False,
            "note": VARIANT_NOTE,
        }
    return fields


def newer_bytes_note(label: str) -> str:
    """The note for a verdict graded on bytes edited while the request was queued."""
    if label == SUBMISSION:
        return (
            "you edited the submission while this check was queued, so it graded your CURRENT bytes "
            "rather than the ones present when you asked. The verdict is about the newer code."
        )
    return (
        f"you edited {label}/ while this check was queued, so it graded its CURRENT bytes rather than "
        "the ones present when you asked. The verdict is about the newer code."
    )


def variant_capacity(ordinary_capacity: int) -> int:
    """Concurrent ordinary variant requests: by default half the ordinary lane, at least one.

    Variants share the ordinary lane and the shared CPU pool's existing lease priority; the bound keeps
    the rest of the lane free for ``submission/`` checks however many variants are queued.
    """
    raw = os.environ.get(VARIANT_JOBS_ENV, "").strip()
    if not raw:
        return max(1, ordinary_capacity // 2)
    if not raw.isascii() or not raw.isdecimal() or not 1 <= int(raw) <= ordinary_capacity:
        raise ValueError(f"{VARIANT_JOBS_ENV} must be an integer from 1 to the ordinary self-check capacity")
    return int(raw)
