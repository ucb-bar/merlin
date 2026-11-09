"""Private final lifecycle observations without physical execution authority.

Numerical readback and frozen callback controls establish their exact observation
scope. A callback's reported cycles and witness hashes are not independent
source-to-hardware, loading, reset, clock or timer qualification.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProtectedFinalObservation:
    member: str
    reference_reported_cycles: int
    candidate_reported_cycles: int
    reference_observation_sha256: str
    candidate_observation_sha256: str
    observation_identity_sha256: str
    accuracy_passed: bool
    element_count: int
    outputs: tuple
    verifier_scope: str

    @property
    def physical_status(self):
        return "UNKNOWN"
