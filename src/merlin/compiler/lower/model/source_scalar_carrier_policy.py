"""Explicit numerical permission for a source-bound scalar representation.

The budget concerns the binary32 scalar carrier before its unchanged finishing
operations. It is not a bound on the final integer code or an entire model.
Those outputs keep their independent original validation gate. No policy is
selected by default, and permission does not establish profitability.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ScalarCarrierErrorBudget:
    atol: float
    rtol: float

    def validate(self):
        if any(type(value) is not float or not math.isfinite(value) or value < 0 for value in (self.atol, self.rtol)):
            raise ValueError("explicit finite nonnegative scalar carrier tolerances required")
        if self.atol == self.rtol == 0:
            raise ValueError("an approximate scalar carrier requires a nonzero explicit budget")


@dataclass(frozen=True)
class ApproximateScalarCarrierPolicy:
    budget: ScalarCarrierErrorBudget
    leading_bits: int
    proof_leading_bits: int
    max_table_bytes: int
    max_proof_table_bytes: int
    approximate_output_permitted: bool
    independent_original_output_validation_required: bool
    original_source_fallback_retained: bool
    target_incoming_rne_predicate_required: bool

    def validate(self):
        if type(self.budget) is not ScalarCarrierErrorBudget:
            raise ValueError("typed scalar carrier budget required")
        self.budget.validate()
        if type(self.leading_bits) is not int or type(self.proof_leading_bits) is not int:
            raise ValueError("explicit integer carrier and proof partition widths required")
        if not 9 <= self.leading_bits <= self.proof_leading_bits <= 23:
            raise ValueError("binary32 carrier/proof widths must satisfy 9 <= carrier <= proof <= 23")
        for amount, required in (
            (self.max_table_bytes, (1 << self.leading_bits) * 12),
            (self.max_proof_table_bytes, (1 << self.proof_leading_bits) * 8),
        ):
            if type(amount) is not int or amount < required:
                raise ValueError("scalar carrier exceeds its explicit immutable storage budget")
        if any(
            value is not True
            for value in (
                self.approximate_output_permitted,
                self.independent_original_output_validation_required,
                self.original_source_fallback_retained,
                self.target_incoming_rne_predicate_required,
            )
        ):
            raise ValueError("complete explicit scalar approximation permission required")

    @property
    def canonical_sha256(self):
        self.validate()
        data = (
            self.budget.atol,
            self.budget.rtol,
            self.leading_bits,
            self.proof_leading_bits,
            self.max_table_bytes,
            self.max_proof_table_bytes,
            self.approximate_output_permitted,
            self.independent_original_output_validation_required,
            self.original_source_fallback_retained,
            self.target_incoming_rne_predicate_required,
        )
        return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()
