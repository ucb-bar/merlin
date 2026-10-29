"""Explicit capability selections over the frozen fresh10 environment.

These selections are independent of historical EL numbers. Source membership is
verified separately; a capability record alone never grants an implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

BASE_REVISION = "f9d7fb5c3d6fa2e5168e982953dd0b1d9b480076"
NAMES = (
    "fresh10-tooling-v1/infra",
    "fresh10-tooling-v1/scheduling",
    "fresh10-tooling-v1/no-reference-kernels",
)
#: Seeded continuations of one frozen fresh10 compiler for the CCA coverage study. They keep the
#: fresh10 author tools, services and prompt composition, stage the current reviewed compiler
#: sources and library instead of the baseline projection, and read one pinned raw reference set.
COVERAGE_NAMES = ("cca-coverage-v1/raw",)
#: The raw reference grant every coverage condition shares: the three original whole-model
#: executables, by exact bytes. Derived analyses (CCA, disassembly, maps) are never part of it.
COVERAGE_RAW_REFERENCE_SHA256 = (
    "97decefb6ff3852d01ce69d30f60d97f7f13bb6ce86815081899da66b87765fd",
    "e598857dd6001d177868d34dac598a333a4c393dc9202734e6587456b121196b",
    "f7a70f3a9ad2791956a808ae5aa212043614307e8ed5d5ab22328af41534de65",
)
TOOLS = ("merlin_infra", "xdsl_kit", "isa_tools", "rtl_generators", "rtl_facts")
SERVICES = frozenset(("reference_analysis", "automatic_kernel_learning", "scheduling_semantics",
                      "transform_library", "catalog_advice", "raw_kernel_references",
                      "performance_diagnostics", "compiler_integration"))


@dataclass(frozen=True)
class Capabilities:
    name: str
    scheduling_semantics: bool
    transform_library: bool
    raw_kernel_references: bool
    #: A coverage continuation: seeded, current sources, pinned raw references only.
    coverage: bool = False

    def allows(self, service):
        if service not in SERVICES:
            raise ValueError("unknown fresh10 author capability: " + str(service))
        if service in {"reference_analysis", "automatic_kernel_learning"}:
            return False
        if service in {"performance_diagnostics", "compiler_integration"}:
            return True
        if service == "catalog_advice":
            return self.transform_library
        return getattr(self, service)

    def record(self):
        return {name: self.allows(name) for name in sorted(SERVICES)}


def resolve(name):
    if name == NAMES[0]:
        return Capabilities(name, False, False, True)
    if name == NAMES[1]:
        return Capabilities(name, True, False, True)
    if name == NAMES[2]:
        # User scope: remove reference material, retain general compiler passes.
        return Capabilities(name, True, True, False)
    if name in COVERAGE_NAMES:
        # The no-reference-kernels tooling plus the study's raw reference grant; CCA stays off.
        return Capabilities(name, True, True, True, coverage=True)
    raise ValueError("unknown fresh10 tooling profile: " + str(name))


def require_preparable(name):
    return resolve(name)


def broker_allowed(name, action):
    capabilities = resolve(name)
    if action in {"reference-analysis", "kernel-learning"}:
        return False
    if action in {"common-transform", "common-check"}:
        return capabilities.transform_library
    return True  # Ordinary numerical selfcheck and measurement admission stay common.
