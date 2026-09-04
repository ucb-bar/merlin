"""The ENVIRONMENT the compiler ran under — recorded, because it changes what gets emitted.

A configuration that does not determine the binary is not a configuration. Two beam nodes carrying
BYTE-IDENTICAL ``knobs.yaml`` (same ``compiler_features``, same ``dtype_strategy``) on the same
capture bundle emitted two different binaries -- digests ``210dbfe9a01c44aa`` and
``2efd837676ff75cd`` -- and ran 2,555,462 ns against 4,151,146 ns, a 1.61x difference that no
recorded field could explain. Nothing in either run's artifacts named the environment.

The lowering path reads a couple of dozen ``MERLIN_*`` variables, and several of them steer codegen
directly rather than merely selecting a host or a path: the per-op block cap, the vectorize-rank
tagging, the OPU packing block and alignment, the worker stack size. An unset-vs-set difference in
any of them changes the emitted code while leaving every recorded field identical.

So the snapshot is taken by PREFIX rather than from a list of names. A curated list is exactly the
thing that goes stale -- the next variable someone adds is the one that is not on it, and its
absence is silent -- and the cost of capturing a few irrelevant variables is nothing next to the
cost of a binary nobody can reproduce.
"""
from __future__ import annotations

import os

#: Prefix that marks a variable as this project's. Everything set under it is captured.
PREFIX = "MERLIN_"

#: Substrings whose VALUE must never be written into an artifact. The variable's presence is still
#: recorded -- that it was set can be the thing that explains a difference -- but the value is not.
_REDACT = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")

#: A value this long is a payload, not a setting; record its size instead of inlining it.
_MAX_VALUE = 512


def _redacted(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in _REDACT)


def snapshot(environ: "dict[str, str] | None" = None) -> dict[str, str]:
    """Every ``MERLIN_*`` variable that is SET, with sensitive values redacted.

    Unset variables are omitted rather than recorded as empty: "not set" and "set to empty string"
    are different inputs to the compiler, and conflating them would make the record lie about which
    one produced the binary.
    """
    env = os.environ if environ is None else environ
    out: dict[str, str] = {}
    for name in sorted(env):
        if not name.startswith(PREFIX):
            continue
        value = env[name]
        if _redacted(name):
            out[name] = "<redacted>"
        elif len(value) > _MAX_VALUE:
            out[name] = f"<{len(value)} chars>"
        else:
            out[name] = value
    return out


def digest(environ: "dict[str, str] | None" = None) -> str:
    """A short stable digest of the snapshot, for comparing two runs at a glance."""
    import hashlib

    snap = snapshot(environ)
    payload = "\n".join(f"{k}={v}" for k, v in sorted(snap.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def differences(a: dict[str, str], b: dict[str, str]) -> dict[str, tuple[str | None, str | None]]:
    """``{name: (a_value, b_value)}`` for every variable the two snapshots disagree on.

    ``None`` on a side means the variable was NOT SET there, which is the difference that is easiest
    to miss and the one most likely to explain two binaries from one recorded configuration.
    """
    out: dict[str, tuple[str | None, str | None]] = {}
    for name in sorted(set(a) | set(b)):
        av, bv = a.get(name), b.get(name)
        if av != bv:
            out[name] = (av, bv)
    return out
