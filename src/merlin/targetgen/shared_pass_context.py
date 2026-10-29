"""Original caller's structural pass-interface requirement during compilation.

This scope changes no authoring grants or semantic qualification. Fresh process
callers must explicitly transport their original selection; contexts alone do
not cross process boundaries.
"""

from __future__ import annotations

import contextvars
import json
from contextlib import contextmanager

_REQUIREMENT = contextvars.ContextVar("original_shared_pass_requirement", default=None)


def _encode(requirement):
    from .shared_pass_contract import validate_shared_pass_requirement

    return json.dumps(validate_shared_pass_requirement(requirement), sort_keys=True, allow_nan=False)


def active_shared_pass_requirement():
    """Return detached structural policy, never an authority or mutable owner."""
    selected = _REQUIREMENT.get()
    return None if selected is None else json.loads(selected)


@contextmanager
def scoped_shared_pass_requirement(requirement):
    """Select before authoring/compilation; nested callers cannot weaken it."""
    encoded = None if requirement is None else _encode(requirement)
    outer = _REQUIREMENT.get()
    if outer is not None and encoded != outer:
        raise ValueError("cannot replace the original shared pass requirement")
    token = _REQUIREMENT.set(encoded)
    try:
        yield
    finally:
        _REQUIREMENT.reset(token)


def selected_shared_pass_requirement(explicit=None):
    """Resolve an explicit or scoped requirement without silently weakening it."""
    outer = _REQUIREMENT.get()
    encoded = outer if explicit is None else _encode(explicit)
    if outer is not None and encoded != outer:
        raise ValueError("explicit shared pass requirement differs from original scope")
    return None if encoded is None else json.loads(encoded)
