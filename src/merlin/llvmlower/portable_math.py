"""The ``portable-transcendentals`` numeric policy: one erf/exp/tanh/log on every target.

THE RULE. The golden a device is graded against and the device image must evaluate the SAME function,
bit for bit. ``convert-math-to-libm`` broke that rule silently for every transcendental: ``math.erf``
became a call to whatever ``erff`` the link supplied, which is glibc's on the host that writes
``golden_w8a8.npy`` and picolibc/newlib's in a RISC-V image. The two differ by one ulp on 76 of 8001
inputs in GELU's range, and requantization amplifies that ulp into whole int8 steps: spectformer int8
missed the W8A8 gate's per-element bound (rel 0.0064, max_rel 0.16) while a host build of the same
model reproduced the golden bit for bit.

THE POLICY. Two values, chosen by name and recorded in every build that lowers through
:func:`merlin.llvmlower.lower.lower_model`:

* ``portable`` (default) -- the lowered model calls ``merlin_erff``/``merlin_expf``/``merlin_tanhf``/
  ``merlin_logf`` from ``merlin/runtime/abi/portable_math.h``, compiled into the MLIR runtime object
  that is linked into the host ``.so`` and into every device image alike. Each evaluates in double
  with an explicit, contraction-free sequence of IEEE-754 operations and rounds once to float, so
  the result is a function of the source alone. Measured exhaustively over all float inputs:
  correctly rounded for erf, exp and tanh, and for log on all but 5 of 2^31 inputs (max error
  0.5000000009 ulp) -- see ``docs/design/portable_transcendentals.md``.
* ``libm`` -- the previous behaviour, each target's C library. Selected with
  ``--no-pass portable-transcendentals`` (``MERLIN_PASSES=-portable-transcendentals``) or
  ``MERLIN_PORTABLE_TRANSCENDENTALS=0``.

Why ``portable`` is the default: a gate comparing a device against a host reference is only
meaningful when both run the same program, and under ``libm`` they never did for these functions.
The change of numerics is small and one-directional in accuracy -- glibc's ``erff`` is off by up to
0.97 ulp on 3% of inputs, ``tanhf`` by up to 2.19 ulp; the portable versions are within 0.5 ulp --
but it IS a change: every model that evaluates one of these functions computes slightly different
outputs than before on every target, and a golden written under one policy is not the reference for
a build under the other. :func:`record` therefore goes into each golden's provenance and each
build's lowering stats, so a mismatch is visible rather than read as a codegen defect.

SCOPE. f32 erf/exp/tanh/log, the functions the frontends emit for GELU, softmax, sigmoid/SiLU,
tanh-GELU and log-softmax. ``rsqrt`` was already Merlin's own (``1/sqrtf``, two correctly rounded
operations). Every other transcendental libm symbol, and any LLVM math intrinsic the backend turns
into a libm call, stays on the target's C library; :func:`bind` lists each one it leaves in
``left_on_target_libm`` so a build that still depends on a C library's rounding says so.

The binding is a symbol rename on the final LLVM IR, done by a tokenizer for LLVM global
identifiers (not text matching): ``@erff`` becomes ``@merlin_erff`` in the declaration and at every
call. Nothing else in the module changes, and under ``libm`` the IR is byte-identical to before.
"""

from __future__ import annotations

import hashlib
import os

#: The optional-pass name (``merlin-compile --list-passes``) and its environment switch.
PASS_NAME = "portable-transcendentals"
SWITCH = "MERLIN_PORTABLE_TRANSCENDENTALS"

PORTABLE = "portable"
LIBM = "libm"

#: C library symbol -> the Merlin runtime symbol the portable policy binds it to.
BINDINGS: dict[str, str] = {
    "erff": "merlin_erff",
    "expf": "merlin_expf",
    "tanhf": "merlin_tanhf",
    "logf": "merlin_logf",
}

#: Inexactly specified libm functions ``convert-math-to-libm`` (or an LLVM math intrinsic) can reach,
#: in both precisions. A reference to one of these that the policy does not bind is reported.
_LIBM_TRANSCENDENTALS = (
    "acos acosh asin asinh atan atan2 atanh cbrt cos cosh erf erfc exp exp2 expm1 log log10 log1p log2 "
    "pow sin sinh tan tanh"
).split()
LIBM_SYMBOLS: frozenset[str] = frozenset(_LIBM_TRANSCENDENTALS + [name + "f" for name in _LIBM_TRANSCENDENTALS])

#: LLVM intrinsics the backend expands into those libm calls (the scalar or vector forms share a
#: prefix ``llvm.<name>.``).
_LIBM_INTRINSIC_STEMS = (
    "exp",
    "exp2",
    "exp10",
    "log",
    "log2",
    "log10",
    "pow",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "asin",
    "acos",
    "atan",
    "atan2",
)

#: The sources that define the portable functions, relative to the runtime directory.
_IMPLEMENTATION = ("abi/portable_math.h", "abi/portable_math_tables.h")


def enabled(environ=None) -> bool:
    """Whether the portable policy is selected: the optional-pass selection when it names the pass,
    else :data:`SWITCH` (on unless ``0``/``off``/``false``/``no``)."""
    from .optional_passes import active

    env = os.environ if environ is None else environ
    state = active(env).state(PASS_NAME)
    if state is not None:
        return state
    return str(env.get(SWITCH, "1")).strip().lower() not in ("0", "off", "false", "no")


def policy(environ=None) -> str:
    return PORTABLE if enabled(environ) else LIBM


def implementation_sha256() -> str:
    """Digest of the bytes that define the portable functions (their source and tables)."""
    from merlin.common.paths import runtime_dir

    h = hashlib.sha256()
    for rel in _IMPLEMENTATION:
        path = runtime_dir() / rel
        h.update(rel.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
    return h.hexdigest()


def record(environ=None) -> dict:
    """The policy as recorded beside a golden or in a build's stats."""
    selected = policy(environ)
    out: dict = {"name": PASS_NAME, "policy": selected}
    if selected == PORTABLE:
        out["functions"] = dict(BINDINGS)
        out["implementation"] = "merlin/runtime/abi/portable_math.h"
        out["implementation_sha256"] = implementation_sha256()
    return out


def cache_token(environ=None) -> str:
    """A string that changes whenever the selected transcendentals change, for keys of caches that
    hold compiled code (a kernel compiled under one policy is not the kernel of the other)."""
    rec = record(environ)
    return f"{rec['policy']}:{rec.get('implementation_sha256', '')}"


# ---- LLVM IR global-identifier tokenizer ------------------------------------------------------

_IDENT_EXTRA = frozenset("$._-")


def _is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch in _IDENT_EXTRA


def _scan_globals(text: str):
    """Yield ``(start, end, name)`` for every ``@name`` global reference in LLVM IR ``text``, skipping
    string literals (``c"..."``, ``!"..."``, quoted names are decoded) and ``;`` comments. ``start`` and
    ``end`` delimit the name after the ``@``, quotes included for a quoted name."""
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == ";":
            nl = text.find("\n", i)
            i = n if nl < 0 else nl + 1
            continue
        if ch == '"':
            close = text.find('"', i + 1)
            i = n if close < 0 else close + 1
            continue
        if ch == "@" and i + 1 < n:
            j = i + 1
            if text[j] == '"':
                close = text.find('"', j + 1)
                if close < 0:
                    return
                yield j, close + 1, text[j + 1 : close]
                i = close + 1
                continue
            k = j
            while k < n and _is_ident_char(text[k]):
                k += 1
            if k > j:
                yield j, k, text[j:k]
            i = k
            continue
        i += 1


def _intrinsic_libm_stem(name: str) -> str | None:
    if not name.startswith("llvm."):
        return None
    stem = name[len("llvm.") :].partition(".")[0]
    return stem if stem in _LIBM_INTRINSIC_STEMS else None


def bind(ll_text: str, environ=None) -> tuple[str, dict]:
    """Apply the selected policy to LLVM IR text; return ``(text, report)``.

    Under ``portable`` every reference to a :data:`BINDINGS` symbol is renamed to its Merlin
    runtime symbol. Under ``libm`` the text is returned unchanged. The report records the policy,
    how many references were rebound per symbol, and every transcendental (libm symbol or libm-backed
    LLVM intrinsic) the module still reaches on the target's C library."""
    rec = record(environ)
    portable = rec["policy"] == PORTABLE
    pieces: list[str] = []
    last = 0
    rebound: dict[str, int] = {}
    remaining: set[str] = set()
    for start, end, name in _scan_globals(ll_text):
        if portable and name in BINDINGS:
            pieces.append(ll_text[last:start])
            pieces.append(BINDINGS[name])
            last = end
            rebound[name] = rebound.get(name, 0) + 1
        elif name in LIBM_SYMBOLS or _intrinsic_libm_stem(name) is not None:
            remaining.add(name)
    if rebound:
        pieces.append(ll_text[last:])
        ll_text = "".join(pieces)
    report = {**rec, "rebound": dict(sorted(rebound.items())), "left_on_target_libm": sorted(remaining)}
    return ll_text, report


def reference_disagreement(provenance: dict | None, environ=None) -> str | None:
    """Why a golden whose provenance is ``provenance`` was not computed under the selected policy, or
    None when it was. A golden that predates the policy (no record) counts as ``libm``: that is what
    every earlier golden was computed with."""
    recorded = (provenance or {}).get("transcendental_math") or {"policy": LIBM}
    current = record(environ)
    if recorded.get("policy") != current["policy"]:
        return (
            f"the reference was computed with {recorded.get('policy')} transcendentals and this build uses "
            f"{current['policy']}; the two differ in the last ulp of erf/exp/tanh/log, which requantization "
            f"can amplify. Regenerate the reference under the build's policy (make_w8a8_golden.py --force)."
        )
    if current["policy"] == PORTABLE and recorded.get("implementation_sha256") != current["implementation_sha256"]:
        return "the reference was computed with a different revision of the portable transcendentals"
    return None
