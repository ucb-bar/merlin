"""Byte-bound compiler-width premises for complete original source programs.

This context proves neither window extraction nor operation support. Existing
source-body recognizers still inspect each actual operation independently.
"""

from __future__ import annotations

import copy
from hashlib import sha256
from pathlib import Path
from weakref import WeakKeyDictionary

from merlin.llvmlower.target_data_layout import selected_index_bits

_ISSUED = WeakKeyDictionary()


class WholeProgramSourceContext:
    """A closed roster of unchanged full programs and a selected compiler observation."""

    def __init__(self, captures, *, source_sha256, selected_index_observation):
        if (type(captures) is not dict or not captures or len(captures) > 512
                or set(captures) != set(source_sha256)
                or any(type(name) is not str or not name for name in captures)):
            raise ValueError("whole-program context requires an exact bounded source roster")
        selected = copy.deepcopy(selected_index_observation)
        if selected_index_bits(selected) is None:
            raise ValueError("whole-program context requires selected compiler index observation")
        rows = {}
        for name, location in captures.items():
            path = Path(location)
            if (not path.is_absolute() or path.resolve() != path or path.is_symlink()
                    or not path.is_file()):
                raise ValueError("whole-program context requires direct canonical source files")
            raw = path.read_bytes()
            if sha256(raw).hexdigest() != source_sha256[name]:
                raise ValueError("whole-program source differs from selected bytes")
            rows[name] = (path, raw)
        _ISSUED[self] = (rows, selected)

    def require(self, captures):
        issued = _ISSUED.get(self)
        if issued is None:
            raise ValueError("whole-program source context was not issued")
        rows, _ = issued
        if (set(captures) != set(rows)
                or any(Path(captures[name]) != rows[name][0] for name in rows)):
            raise ValueError("whole-program context names another source roster")
        for path, raw in rows.values():
            if path.is_symlink() or path.resolve() != path or path.read_bytes() != raw:
                raise ValueError("whole-program context source bytes changed")

    def for_program(self, program):
        issued = _ISSUED.get(self)
        if issued is None:
            raise ValueError("whole-program source context was not issued")
        rows, selected = issued
        self.require({name: path for name, (path, _) in rows.items()})
        if Path(program) not in {path for path, _ in rows.values()}:
            raise ValueError("whole-program context names another program")
        return {"selected_index_observation": copy.deepcopy(selected)}

    def control_admission_context(self, program, *, target):
        self.for_program(program)
        return None

    def index_admission_context(self, program):
        self.for_program(program)
        return None
