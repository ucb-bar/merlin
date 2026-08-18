"""Resolve + verify the Triton frontend toolchain, and fail loudly on a version mismatch.

``merlin.triton`` depends on a *compiler-internal* Triton surface (``ASTSource`` / ``make_ir``),
which is not a stable public API and moves between minor releases. Best-effort tolerance of an
unexpected version is the wrong behavior here: a drifted Triton either raises deep inside the
frontend with an opaque traceback, or — worse — produces subtly different TTIR that changes the
compiled kernel with nothing to point at. So the version is exact-pinned and checked up front.

The pin lives in exactly two places that a test keeps in agreement: :data:`PINNED_TRITON` here and
the ``triton`` extra in ``pyproject.toml``. There is deliberately no third lock file restating it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# The exact triton release merlin.triton is developed and tested against. Bump here AND in the
# pyproject `triton` extra (test_triton_toolchain asserts they agree), and re-run the TTIR
# golden/hash tests: a new frontend can legitimately change TTIR spelling.
PINNED_TRITON = "3.7.1"


class TritonToolchainError(RuntimeError):
    """Triton is absent, or present at a version this frontend is not pinned to."""


@dataclass(frozen=True)
class Probe:
    """What is actually installed, and whether we are willing to use it."""

    installed: str | None            # triton version found, or None if absent
    pinned: str = PINNED_TRITON
    compatible: bool = False
    reason: str = ""
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"triton": self.installed, "pinned": self.pinned,
                "compatible": self.compatible, "reason": self.reason,
                "notes": list(self.notes)}


def _installed_version() -> str | None:
    """The installed triton version, or None.

    Read from the distribution metadata rather than ``triton.__version__`` so a probe never
    imports triton (importing it loads a large native library and, on some builds, touches the
    driver). Falls back to the module attribute only if the dist metadata is missing — which is
    itself a signal: a stripped/partial install can ship the Python tree with no dist-info and no
    ``triton/_C`` native library, and such a tree imports far enough to look real while being
    unable to build any IR.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("triton")
    except PackageNotFoundError:
        pass
    try:
        import triton
    except Exception:  # noqa: BLE001 — absent or unimportable are the same answer here
        return None
    return getattr(triton, "__version__", None)


def _native_library_present() -> bool:
    """Whether the install carries triton's compiled core (``triton._C.libtriton``).

    A Python-only tree cannot produce TTIR. This is checked separately from the version because a
    stripped install reports a plausible version and then fails much later, deep in ``make_ir``.
    """
    from importlib.util import find_spec

    try:
        return find_spec("triton._C.libtriton") is not None
    except (ImportError, ValueError, AttributeError):
        return False


def probe() -> Probe:
    """Report the toolchain state without raising — for CLIs, diagnostics and tests."""
    found = _installed_version()
    if found is None:
        return Probe(installed=None, compatible=False,
                     reason="triton is not installed (install the `triton` extra)")
    notes: list[str] = []
    if not _native_library_present():
        notes.append("triton._C.libtriton not importable — the install looks Python-only/stripped "
                     "and cannot build TTIR")
    if found != PINNED_TRITON:
        return Probe(installed=found, compatible=False, notes=notes,
                     reason=f"triton version mismatch: expected {PINNED_TRITON}, found {found}")
    if notes:
        return Probe(installed=found, compatible=False, notes=notes,
                     reason="triton is present at the pinned version but is not a usable install")
    return Probe(installed=found, compatible=True, reason="", notes=notes)


def require() -> Probe:
    """:func:`probe`, but raise :class:`TritonToolchainError` when unusable.

    Every entry point that will touch Triton internals calls this first, so the failure names the
    expected and found versions instead of surfacing as an AttributeError inside triton.
    """
    p = probe()
    if not p.compatible:
        detail = "\n  ".join(p.notes)
        raise TritonToolchainError(
            f"{p.reason}\n"
            f"  merlin.triton is pinned to triton=={PINNED_TRITON} because it uses triton's\n"
            f"  compiler-internal frontend (ASTSource/make_ir), which is not a stable API."
            + (f"\n  {detail}" if detail else ""))
    return p
