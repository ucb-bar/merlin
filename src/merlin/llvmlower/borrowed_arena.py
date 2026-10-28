"""Live transport of native-checked borrowed effects to the LLVM allocator.

Reports describe this binding; they cannot recreate it. A changed LLVM module
gets no additional allocator permissions. In particular, equal callee names in
another module do not establish completion or absence of escaping aliases.
"""

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from merlin.common.digest import sha256_file

from .borrowed_buffer_calls import CHECKPOINT, REPORT, ROSTER, BorrowedBufferCalls

_SEAL = object()


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _callback_identity(callback):
    function = getattr(callback, "__func__", callback)
    return (id(getattr(callback, "__self__", None)), function, getattr(function, "__code__", None))


def _ciface_annotation_changes(before, after, report):
    """Independent conformance, not replay of the annotation emitter.

    Every changed line must be a standalone function definition header with
    precisely one added ``alwaysinline`` token immediately before its body.
    All instruction/call/memory bodies, declarations, globals and signatures
    stay byte-identical. A report alone supplies no transport permission.
    """
    if (type(report) is not dict or set(report) != {'schema', 'max_flattened_arguments', 'inlined', 'count'}
            or report['schema'] != 'wide_ciface_inline_v1'
            or type(report['max_flattened_arguments']) is not int or report['max_flattened_arguments'] < 1
            or type(report['inlined']) is not list or type(report['count']) is not int):
        return None
    original, selected = before.splitlines(keepends=True), after.splitlines(keepends=True)
    if len(original) != len(selected):
        return None

    def definition_name(line):
        if not line.startswith('define ') or not line.rstrip('\r\n').endswith('{'):
            return None
        prefix, at, tail = line.partition('@')
        name, opening, signature = tail.partition('(')
        if (not at or not opening or not name or not (name[0].isalpha() or name[0] in '$._')
                or any(not (c.isascii() and (c.isalnum() or c in '$._-')) for c in name)
                or ')' not in signature or not prefix.strip()):
            return None
        return name

    definitions = {name for line in original if (name := definition_name(line)) is not None}
    changed = []
    for old, new in zip(original, selected, strict=True):
        if old == new:
            continue
        name = definition_name(old)
        if name is None or name.startswith('_mlir_ciface_') or '_mlir_ciface_' + name not in definitions:
            return None
        brace = old.rfind('{')
        prefix = old[:brace].rstrip()
        if 'alwaysinline' in prefix.split() or new != prefix + ' alwaysinline ' + old[brace:]:
            return None
        changed.append(name)
    rows = report['inlined']
    if len(changed) != report['count'] or len(rows) != len(changed):
        return None
    for name, row in zip(changed, rows, strict=True):
        if (type(row) is not dict or set(row) != {'implementation', 'wrapper', 'flattened_arguments'}
                or row['implementation'] != name or row['wrapper'] != '_mlir_ciface_' + name
                or type(row['flattened_arguments']) is not int
                or row['flattened_arguments'] <= report['max_flattened_arguments']):
            return None
    return tuple(changed)


@dataclass(frozen=True, init=False, eq=False, repr=False)
class BorrowedArenaLifetime:
    """One original selection, actual native observation and exact emitted LLVM.

    The allocator still proves allocation/free windows, escapes and generations.
    This object supplies only the original borrowed-call effects consumed by the
    native alias checker, conditional on the selected implementation remaining
    current. It is intentionally not a serializable capability.
    """

    def __init__(self, seal, selection, source, llvm_ir, pins, accepted, annotations=()):
        if seal is not _SEAL or type(selection) is not BorrowedBufferCalls:
            raise TypeError("borrowed arena lifetimes require actual native lowering")
        fields = dict(
            _selection=selection, _source=source, _llvm_sha256=_sha(llvm_ir),
            _pins=pins, _accepted=tuple(accepted),
            _original=json.dumps(selection.record(), sort_keys=True),
            _callback=_callback_identity(selection.validate_original),
            _observer=_callback_identity(selection.observe_native),
            _annotations=annotations,
        )
        for name, value in fields.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_state", tuple(fields.items()))

    def _check(self):
        if any(getattr(self, name) is not value for name, value in self._state):
            raise ValueError("original borrowed arena binding changed")
        selection = self._selection
        if (type(selection) is not BorrowedBufferCalls
                or _callback_identity(selection.validate_original) != self._callback
                or _callback_identity(selection.observe_native) != self._observer
                or json.dumps(selection.record(), sort_keys=True) != self._original):
            raise ValueError("original borrowed arena selection changed")
        selection.revalidate(self._source)
        if json.dumps(selection.record(), sort_keys=True) != self._original:
            raise ValueError("original borrowed arena selection changed during validation")
        for path, digest in self._pins:
            if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                raise ValueError("borrowed arena native observation changed")

    def checked(self, llvm_ir):
        """Validate once per allocator invocation; return symbols and a receipt.

        Unsupported post-lowering LLVM transformations retain the ordinary heap
        rules. A stale original implementation/checkpoint remains an error.
        """
        self._check()
        current = _sha(llvm_ir) == self._llvm_sha256
        symbols = self._accepted if current else ()
        return frozenset(symbols), {
            "schema": "merlin.borrowed_arena_lifetime.v1",
            "status": "checked" if current else "llvm_changed_conservative",
            "llvm_sha256": self._llvm_sha256,
            "accepted_symbols": list(symbols),
            "native_products": [{"path": str(path), "sha256": digest} for path, digest in self._pins],
            **({"abi_annotations": [dict(input_sha256=before, selected_sha256=after, functions=list(names),
                                         bodies_calls_memory="unchanged")
                                    for before, after, names in self._annotations]} if self._annotations else {}),
            "scope": "original conditional effects; exact native-checked lowering; final implementation independent",
        }

    def after_ciface_annotation(self, before, after, report):
        """Carry only the ordinary ABI annotation after independent conformance.

        Unsupported changes keep this exact original binding; the final allocator
        consequently observes a hash mismatch and receives no extra permissions.
        This is not a generic LLVM hash rebinding API.
        """
        if before == after or _sha(before) != self._llvm_sha256:
            return self
        self._check()
        names = _ciface_annotation_changes(before, after, report)
        if not names:
            return self
        from . import ciface_inline

        implementation = Path(ciface_inline.__file__).resolve(strict=True)
        return BorrowedArenaLifetime(
            _SEAL, self._selection, self._source, after,
            (*self._pins, (implementation, sha256_file(implementation))), self._accepted,
            (*self._annotations, (self._llvm_sha256, _sha(after), names)),
        )


def _from_verified_native(selection, source, work, llvm_ir, checked, runner):
    """Called by the lowering invocation after verify_native's actual token check."""
    selection.revalidate(source)
    work = Path(work)
    if json.loads((work / REPORT).read_text()) != checked or checked["status"] != "checked":
        raise ValueError("actual borrowed native observation changed before arena binding")
    paths = tuple((work / name).resolve(strict=True) for name in (ROSTER, CHECKPOINT, REPORT))
    paths = (*paths, Path(runner).resolve(strict=True))
    if checked["checkpoint_sha256"] != sha256_file(work / CHECKPOINT):
        raise ValueError("actual borrowed native checkpoint changed before arena binding")
    return BorrowedArenaLifetime(
        _SEAL, selection, source, llvm_ir,
        tuple((path, sha256_file(path)) for path in paths), checked["accepted_symbols"],
    )


class _LifetimeReceipt(dict):
    """JSON-compatible record with invocation-only, non-copyable custody.

    The live owner is not a mapping entry: JSON, receipt hashes and ordinary
    iteration see only the original record. Copies intentionally lose the grant.
    """

    def __init__(self, record=(), *, lifetime=None):
        super().__init__(record)
        self._lifetime = lifetime

    def __copy__(self):
        return dict(self)

    def __deepcopy__(self, memo):
        return deepcopy(dict(self), memo)

    def __reduce_ex__(self, protocol):
        return dict, (dict(self),)


def retain_borrowed_lifetime(record, lifetime):
    """Carry the live owner through an ordinary in-process build/audit join."""
    if lifetime is None:
        return record
    if type(lifetime) is not BorrowedArenaLifetime:
        raise TypeError("receipt lifetime requires the original live native binding")
    return _LifetimeReceipt(record, lifetime=lifetime)


def receipt_borrowed_lifetime(record):
    """Saved/copied dictionaries supply no grant; the allocator checks live owners."""
    return record._lifetime if type(record) is _LifetimeReceipt else None
