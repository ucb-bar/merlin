"""Harness errors at admission boundaries of the pinned native CPU prover."""


class NativeCPUCertificateUnavailable(RuntimeError):
    """A lane decision requires unavailable certified native CPU artifacts."""

    code = "native_cpu_certificate_unavailable"


def require_certificate_available(proof):
    """Reject an unavailable admission proof, retaining its validation reason."""
    if isinstance(proof, dict) and proof.get("code") == NativeCPUCertificateUnavailable.code:
        raise NativeCPUCertificateUnavailable(proof.get("reason", NativeCPUCertificateUnavailable.code))
    return proof
