"""Program-local, immutable-input preparation for ordinary host admission.

Only declaration validation and selector lookup are retained. Every supplied
source body and every admission decision is still checked by the original
consumer; this object is neither a source proof nor a persistent verdict cache.
"""

import copy
import json
import weakref
from collections.abc import Mapping


def _bytes(selected):
    return json.dumps(selected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


_ISSUED = {}


class PreparedHostCapabilities(Mapping):
    """Own a private copy and an order-preserving exact-selector index."""

    def __init__(self, selected):
        from merlin.targetgen.host_capabilities import validate_host_capabilities

        if not isinstance(selected, dict):
            raise ValueError("selected host capabilities must be a profile mapping")
        input_bytes = _bytes(selected)
        selected_copy = copy.deepcopy(selected)
        indices = {}
        for name, selection in sorted(selected_copy.items()):
            if not isinstance(selection, dict):
                raise ValueError("selected host capability profile must be a mapping")
            document = selection.get("capability_spec")
            if document is None:
                continue
            validate_host_capabilities(
                document, package_sha256=selection.get("package_sha256"), dtype_strategy=selection.get("dtype_strategy")
            )
            exact, fallback = {}, []
            for position, declaration in enumerate(document["operations"]):
                selectors = declaration.get("ops") or []
                if not selectors:
                    fallback.append(position)
                for operation in selectors:
                    exact.setdefault(operation, set()).add(position)
            indices[name] = exact, tuple(fallback)
        identity = id(self)
        _ISSUED[identity] = (
            weakref.ref(self, lambda _: _ISSUED.pop(identity, None)),
            input_bytes,
            selected_copy,
            indices,
        )
        self.require_current(selected)

    def _binding(self):
        original = _ISSUED.get(id(self))
        if original is None or original[0]() is not self:
            raise ValueError("host capability profile has no original issued preparation")
        return original

    def __len__(self):
        return len(self._binding()[2])

    def __iter__(self):
        return iter(self._binding()[2])

    def __getitem__(self, name):
        # Other consumers cannot mutate the prepared document through Mapping.
        return copy.deepcopy(self._binding()[2][name])

    def require_current(self, selected):
        """Reopen the caller's selected bytes before accepting the program."""
        _, input_bytes, selected_copy, _ = self._binding()
        if _bytes(selected) != input_bytes or selected != selected_copy:
            raise ValueError("selected host capability profile changed during source admission")

    def _items(self):
        return sorted(self._binding()[2].items())

    def _declarations(self, name, row):
        _, _, selected, indices = self._binding()
        exact, fallback = indices[name]
        positions = set(fallback)
        for identity in (row.get("frontend_op"), row["mlir_operation"]):
            positions.update(exact.get(identity, ()))
        document = selected[name]["capability_spec"]
        return tuple(document["operations"][position] for position in sorted(positions))
