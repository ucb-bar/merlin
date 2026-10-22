"""Frozen public scalability inputs for developmental rolling Spike checks.

This adds no measurement authority and issues no full-suite qualification.
The existing public_scalability final gate remains the only suite issuer.
"""
from __future__ import annotations

from . import public_scalability


class Selection:
    def __init__(self, context):
        self.context = context
        self.selected = public_scalability.selected(context)
        self.cases, self.identity = {}, None
        if self.selected is not None:
            root, record = self.selected
            self.cases = {name: root / 'cases' / name for name in record['cases']}
            self.identity = {'root': str(root), 'record': record}

    def verify(self):
        if public_scalability.selected(self.context) != self.selected:
            raise ValueError('rolling public scalability selection changed')
