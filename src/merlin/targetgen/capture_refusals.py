"""Typed capture refusals shared by worker and parent; unexpected failures stay closed."""

import json


class M2MUnavailable(RuntimeError):
    """The selected capture implementation could not produce a qualified artifact."""


class InputStorageUnavailable(M2MUnavailable):
    """The portable input payload does not record storage read by the CPU call."""

    code = "input_storage_unavailable"

    def __init__(self, reason, *, details):
        super().__init__(reason)
        self.details = details


class NumericCaptureRefused(M2MUnavailable):
    """A frontend-proved refusal permits preserving the original host graph."""

    def __init__(self, reason, *, code, details=None):
        super().__init__(reason)
        self.code = code
        self.details = details or {}


class NoEligibleCaptureOperations(NumericCaptureRefused):
    def __init__(self, reason):
        super().__init__(reason, code="no_eligible_recipe_operations")


def write_refusal(directory, *, code, reason, **details):
    (directory / "refusal.json").write_text(
        json.dumps(
            {
                "schema": "merlin.capture.refusal.v1",
                "code": code,
                "reason": reason,
                **details,
            }
        )
        + "\n"
    )


def write_exception_refusal(directory, refusal, *, fallback_code):
    """Preserve a typed worker refusal's structured details in its receipt."""
    write_refusal(
        directory,
        code=getattr(refusal, "code", fallback_code),
        reason=str(refusal),
        **getattr(refusal, "details", {}),
    )


def read_refusal(path):
    refusal = json.loads(path.read_text())
    if (
        refusal.get("schema") != "merlin.capture.refusal.v1"
        or refusal.get("code")
        not in {
            "no_eligible_recipe_operations",
            "nonfinite_calibration",
            "complex_calibration",
            "declared_affine_domain_unproved",
            "unquantized_special_readout_unproved",
            "integerization_golden_disagreement",
        }
        or not isinstance(refusal.get("reason"), str)
    ):
        return None
    if refusal["code"] == "no_eligible_recipe_operations":
        return NoEligibleCaptureOperations(refusal["reason"])
    return NumericCaptureRefused(refusal["reason"], code=refusal["code"], details=refusal)


def refuse_integerization(directory, receipt):
    if receipt.get("golden_agreement", {}).get("status") == "passed":
        return False
    write_refusal(
        directory,
        code="integerization_golden_disagreement",
        reason="integerization receipt has no passed original-vs-rewrite golden agreement",
        integerization_receipt=receipt,
    )
    return True
