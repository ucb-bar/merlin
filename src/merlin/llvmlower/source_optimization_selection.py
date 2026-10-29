"""Original caller limits for source choices consumed by ordinary lowering.

The request is proposal data. This selection must be supplied by the original
compiler owner, with permissions and a ceiling selected before candidate work.
It is not a qualification issuer, software-spec parser or restored admission.
Frozen authoring environments need a separately reviewed transport and release;
constructing this object never grants a candidate access to these modules.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .source_optimization_request import GATHER_PASS, PACKING_PASS, SourceOptimizationRequest


@dataclass(frozen=True)
class SourceOptimizationSelection:
    request: SourceOptimizationRequest
    permitted_passes: tuple[str, ...]
    maximum_added_operations: int

    def bind(self, source):
        if type(self.request) is not SourceOptimizationRequest:
            raise ValueError("source selection requires the exact bounded request type")
        return self.request.bind(
            source,
            permitted_passes=self.permitted_passes,
            maximum_added_operations=self.maximum_added_operations,
        )

    def record(self):
        return {
            "request": self.request.record(),
            "original_permitted_passes": list(self.permitted_passes),
            "original_maximum_added_operations": self.maximum_added_operations,
        }


class SourceCompilationOptimization:
    """Bind once at the actual lower_model input, then revalidate on return.

    Choices select whole passes, not source-region witnesses or stale analyses.
    Earlier graph/row rewrites may derive new IR; each chosen pass still checks
    that current IR independently. No environment variable is changed.
    """

    def __init__(self, source, selection):
        if type(selection) is not SourceOptimizationSelection:
            raise ValueError("source choices require the original typed selection")
        self._source, self._selection = source, selection
        self._states, self.maximum_added_operations = selection.bind(source)
        self._record = selection.record()
        self._identity = self._digest(self._record)
        self._library = self._library_sources()

    @staticmethod
    def _digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

    @staticmethod
    def _library_sources():
        names = (
            "source_optimization_selection.py",
            "source_optimization_request.py",
            "optional_passes.py",
        )
        return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}

    def revalidate(self):
        states, maximum = self._selection.bind(self._source)
        if (
            self._digest(self._selection.record()) != self._identity
            or states != self._states
            or maximum != self.maximum_added_operations
            or self._library_sources() != self._library
        ):
            raise ValueError("original source selection or request library changed during compilation")

    def options(self, name):
        self.revalidate()
        if name not in (GATHER_PASS, PACKING_PASS):
            raise ValueError("source selection cannot control an unrelated pass")
        return {"enabled": self._states.state(name), "max_added_operations": self.maximum_added_operations}

    def report(self, *, gather, packing):
        self.revalidate()
        return {
            "schema": "merlin.source_optimization_selection.v1",
            "binding_stage": "lower_model_input_before_source_optimizations",
            "selection": self._selection.record(),
            "selection_sha256": self._identity,
            "library_sources_sha256": dict(self._library),
            "consumers": {
                GATHER_PASS: {
                    "source_file_sha256": gather["source_file_sha256"],
                    "output_file_sha256": gather["output_file_sha256"],
                    "rewrites": gather["rewrites"],
                },
                PACKING_PASS: {
                    "source_file_sha256": packing["source_file_sha256"],
                    "output_file_sha256": packing["output_file_sha256"],
                    "rewrites": packing["rewrites"],
                },
            },
            "machine_conformance": "not_established",
            "qualification": "no new import, numerical, execution or Phase 2 admission",
        }
