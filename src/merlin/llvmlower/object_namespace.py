"""Select checked symbol rewriting or shared storage for an unchanged object.

The provider observes every symbol, including local and undefined references,
and implements the selected object rewrite. Merlin binds both observations to
the actual immutable input bytes and checks complete output symbol membership.
No ISA, ABI, executable semantics or runtime sharing follows from this receipt.
"""

import hashlib
import os
from collections import Counter
from pathlib import Path
from types import MappingProxyType


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("object namespace requires regular immutable input and output files")


def namespace_object(source, destination, renames, *, inspect_symbols, rewrite):
    """Avoid copying only when a complete observed symbol roster needs no change.

    The provider's inspector must include local, global and undefined names with
    multiplicity. The rewriter receives the same complete selected rename map.
    Changed/partial symbol observations or mutated source bytes refuse. An
    unchanged object receives a fresh hardlink and a distinct storage receipt,
    never a fictitious rewrite command. Existing destinations are untouched.
    """
    source, destination = Path(source), Path(destination)
    _regular(source)
    if destination.exists() or destination.is_symlink():
        raise ValueError("object namespace destination must be fresh")
    renames = dict(renames)
    if any(not isinstance(s, str) or not s for pair in renames.items() for s in pair):
        raise ValueError("object namespace mappings require nonempty symbol names")
    original_digest = _sha(source)
    names = tuple(inspect_symbols(source))
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("object namespace inspector returned an incomplete symbol roster")
    if _sha(source) != original_digest:
        raise ValueError("object namespace input changed during symbol observation")
    selected = {name: renames[name] for name in names if name in renames and renames[name] != name}
    targets = list(selected.values())
    if len(set(targets)) != len(targets) or set(targets) & (set(names) - selected.keys()):
        raise ValueError("object namespace mapping collides with an existing symbol")
    expected = Counter(renames.get(name, name) for name in names)
    if selected:
        rewrite(source, destination, MappingProxyType(renames))
        method = "provider_symbol_rewrite"
    else:
        os.link(source, destination)
        method = "shared_unchanged_object"
    _regular(destination)
    actual = tuple(inspect_symbols(destination))
    if Counter(actual) != expected or _sha(source) != original_digest:
        raise ValueError("object namespace output roster or immutable source changed")
    output_digest = _sha(destination)
    shared = os.path.samefile(source, destination)
    if not selected and (not shared or output_digest != original_digest):
        raise ValueError("unchanged object namespace did not preserve shared immutable bytes")
    return {"schema": "merlin.object_namespace_storage.v1", "method": method,
            "scope": "object_storage_and_observed_symbols_not_machine_equivalence_or_admission",
            "source": str(source.resolve()), "source_sha256": original_digest,
            "output": str(destination.resolve()), "output_sha256": output_digest,
            "bytes": destination.stat().st_size, "shared_inode": shared,
            "renames": selected, "symbol_count": len(names)}
