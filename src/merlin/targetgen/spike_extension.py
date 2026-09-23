"""WHICH Spike extension a target's L2 functional oracle loads — read from that target's own contract,
verified by digest, fail-closed.

THE FAILURE THIS EXISTS TO PREVENT. A Spike RoCC extension is compiled against a *specific* generated
``gemmini_params.h``. Two sibling elaborations of one generator differ in that header (mesh dimension,
accumulator geometry, whether the full-width accumulator read port exists at all), and a functional
model built from one header running a kernel compiled against the other does not fail loudly: it
returns wrong numbers. The repo has already paid for this once — a params.h skew produced an
all-zeros matmul that read as a compiler defect (memory ``libgemmini-driver-allzeros-header-skew``).

So the extension is not a property of "whichever chipyard the operator's ``$MERLIN_CHIPYARD`` names".
It is a property of THE TARGET, and a target that has its own model declares it in its own contract::

    runner:
      spike_extension:
        extension_name: <the extension class the .so registers>
        extlib: /abs/path/to/lib<name>.so
        sha256: <64 hex>

STRICTLY ADDITIVE, AND THAT IS LOAD-BEARING. A target that declares no ``runner.spike_extension`` is
resolved exactly as before: the caller's own default library directory and default extension name, with
no ``--extlib`` flag. Nothing about such a target's invocation changes, byte for byte
(``merlin/tests/gemmini/test_spike_extension.py`` asserts that against the live reference backend).

FAIL CLOSED, NEVER SUBSTITUTE. When a target DOES declare one, every way of not getting exactly those
bytes raises :class:`SpikeExtensionError`:

* the declaration is incomplete (no ``extlib``, no ``extension_name``, no ``sha256``);
* the file is absent or unreadable;
* the bytes on disk do not hash to the declared ``sha256``.

The one thing this must never do is fall back to another target's ``.so``, because that is the
mis-attribution above wearing a success's clothes. An operator who wants a different model changes the
contract (and its digest), which is reviewable; nothing here reads an env var that could silently
re-point the model a verdict was earned on.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "SpikeExtension",
    "SpikeExtensionError",
    "declared_extension",
    "resolve",
    "spike_invocation",
]


class SpikeExtensionError(RuntimeError):
    """A target DECLARED a Spike extension and the declared bytes could not be produced.

    Deliberately not a subclass of any target's backend error: the condition is about the ORACLE's
    identity, not about a program, and a caller that swallows backend errors must not swallow this.
    """


@dataclass(frozen=True)
class SpikeExtension:
    """The resolved L2 functional-model identity for one target.

    ``declared`` distinguishes the two cases a reader must never confuse: a target whose own contract
    names and digests its model (``True``), and a target that has no declaration and is therefore
    served by whatever the caller's toolchain default resolves to (``False``).
    """

    target: str
    extension_name: str
    #: The ``.so`` to pass as ``--extlib``, or ``None`` when the caller's default search path is used.
    extlib: Path | None
    #: The directory that must be on ``LD_LIBRARY_PATH`` for the extension to load.
    library_dir: Path
    #: The digest the contract DECLARES, and which the bytes on disk were verified against.
    sha256: str | None
    declared: bool

    def spike_flags(self) -> tuple[str, ...]:
        """The ``spike`` argv flags, in the order the extension API requires (``--extlib`` before
        ``--extension``; spike parses the library before it can look the extension name up)."""
        if self.extlib is None:
            return (f"--extension={self.extension_name}",)
        return (f"--extlib={self.extlib}", f"--extension={self.extension_name}")

    def describe(self) -> str:
        if not self.declared:
            return (
                f"{self.target}: toolchain-default spike extension {self.extension_name!r} from "
                f"{self.library_dir} (target declares none)"
            )
        return (
            f"{self.target}: declared spike extension {self.extension_name!r} at {self.extlib} (sha256 {self.sha256})"
        )


def _contract_runner(target: str) -> dict[str, Any]:
    """``runner`` from ``target``'s contract, or ``{}`` when the contract cannot be read.

    An UNREADABLE contract is not the same condition as an INVALID declaration and must not raise:
    a caller with no contract at all (a bare backend probe, an out-of-tree package that has not been
    generated) has always resolved to its own default, and turning that into an exception would break
    every such caller for a fact none of them declares. An unreadable contract yields "declares none",
    and the fail-closed half below then never runs.
    """
    try:
        from . import target_registry

        contract = target_registry.resolve(target).load_contract()
    except Exception:  # noqa: BLE001 — no contract / unresolvable target ⇒ "declares none", see above
        return {}
    if not isinstance(contract, dict):
        return {}
    runner = contract.get("runner")
    return runner if isinstance(runner, dict) else {}


def declared_extension(target: str) -> dict[str, Any] | None:
    """The raw ``runner.spike_extension`` block ``target`` declares, or ``None``.

    ``None`` means "this target names no model of its own" — the additive default path. A block that is
    PRESENT but malformed is returned as-is so :func:`resolve` can refuse it with a precise reason
    rather than silently reading it as an absence.
    """
    block = _contract_runner(target).get("spike_extension")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise SpikeExtensionError(
            f"{target}: runner.spike_extension must be a mapping with extension_name/extlib/sha256, "
            f"got {type(block).__name__}"
        )
    return block


#: Verified (path, size, mtime_ns) triples. The KEY carries the stat fields, so a rebuilt or replaced
#: ``.so`` is re-hashed rather than trusted: the cache saves the repeated hash inside one grade, and
#: cannot carry a verification across a change to the bytes. (Deliberately not keyed on the path alone.)
_verified_stat: dict[tuple[str, int, int], str] = {}


def _digest_verified(target: str, so: Path, declared_sha: str) -> str:
    try:
        st = so.stat()
    except OSError as exc:
        raise SpikeExtensionError(
            f"{target}: declared spike extension {so} cannot be stat'ed ({exc}). The L2 functional "
            f"model this target's verdicts are measured on is absent; refusing to run on another "
            f"target's extension"
        ) from exc
    key = (str(so), st.st_size, st.st_mtime_ns)
    cached = _verified_stat.get(key)
    if cached is not None:
        got = cached
    else:
        h = hashlib.sha256()
        try:
            with so.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
        except OSError as exc:
            raise SpikeExtensionError(f"{target}: declared spike extension {so} cannot be read ({exc})") from exc
        got = h.hexdigest()
        _verified_stat[key] = got
    if got != declared_sha:
        raise SpikeExtensionError(
            f"{target}: declared spike extension {so} has sha256 {got}, but the contract declares "
            f"{declared_sha}. These are DIFFERENT bytes, so this is a different functional model than "
            f"the one this target's contract was reviewed against — refusing to grade on it. Either the "
            f"file was rebuilt (update runner.spike_extension.sha256 with a reviewed rationale) or the "
            f"path now names somebody else's build"
        )
    return got


def resolve(target: str, *, default_library_dir: str | Path, default_extension_name: str) -> SpikeExtension:
    """The verified L2 extension identity for ``target``.

    ``default_library_dir`` / ``default_extension_name`` are the CALLER's status quo, used unchanged
    when the target declares nothing — the parameters exist so this module names no toolchain layout and
    no extension name of its own.

    Raises :class:`SpikeExtensionError` when the target declares an extension that cannot be produced
    exactly as declared.
    """
    block = declared_extension(target)
    if block is None:
        return SpikeExtension(
            target=target,
            extension_name=str(default_extension_name),
            extlib=None,
            library_dir=Path(default_library_dir),
            sha256=None,
            declared=False,
        )
    name = block.get("extension_name")
    extlib = block.get("extlib")
    sha = block.get("sha256")
    missing = [
        field
        for field, value in (("extension_name", name), ("extlib", extlib), ("sha256", sha))
        if not isinstance(value, str) or not value.strip()
    ]
    if missing:
        raise SpikeExtensionError(
            f"{target}: runner.spike_extension is declared but incomplete — missing {missing}. A model "
            f"a verdict cites must be named AND digested; an undigested path cannot be told apart from "
            f"a sibling elaboration's build sitting in the same directory"
        )
    so = Path(str(extlib).strip()).expanduser()
    if not so.is_absolute():
        raise SpikeExtensionError(
            f"{target}: runner.spike_extension.extlib must be an absolute path, got {extlib!r} — a "
            f"relative one resolves against whatever directory the grade happens to run from"
        )
    declared_sha = str(sha).strip().lower()
    if len(declared_sha) != 64 or any(c not in "0123456789abcdef" for c in declared_sha):
        raise SpikeExtensionError(f"{target}: runner.spike_extension.sha256 must be 64 hex characters, got {sha!r}")
    if not so.is_file():
        raise SpikeExtensionError(
            f"{target}: declared spike extension {so} is not a file. This target's L2 oracle is its own "
            f"model; refusing to fall back to the toolchain default at {default_library_dir}, which is "
            f"a DIFFERENT elaboration's extension and would put another device's verdict on this "
            f"target's submission"
        )
    _digest_verified(target, so, declared_sha)
    return SpikeExtension(
        target=target,
        extension_name=str(name).strip(),
        extlib=so,
        library_dir=so.parent,
        sha256=declared_sha,
        declared=True,
    )


def spike_invocation(
    target: str, *, default_library_dir: str | Path, default_extension_name: str
) -> tuple[tuple[str, ...], Path]:
    """``(spike argv flags, the directory to put on LD_LIBRARY_PATH)`` for ``target``.

    The one call a backend's ``run_elf`` needs. For a target that declares nothing this returns exactly
    ``((f"--extension={default_extension_name}",), Path(default_library_dir))`` — the status quo.
    """
    ext = resolve(target, default_library_dir=default_library_dir, default_extension_name=default_extension_name)
    return ext.spike_flags(), ext.library_dir


def library_path_with(library_dir: str | Path, env: "dict[str, str] | None" = None) -> str:
    """``library_dir`` prepended to ``LD_LIBRARY_PATH`` — the same string shape callers already build,
    factored out so the extension directory and the search path cannot drift apart."""
    existing = (env or os.environ).get("LD_LIBRARY_PATH", "")
    return f"{library_dir}:{existing}"
