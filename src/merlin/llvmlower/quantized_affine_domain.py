"""Exact pair observations on caller-bound, source-proven signed-byte domains.

The complete type table remains recorded; excluded pairs grant no equivalence.
Typed producer/consumer and effect binding is independently required. No captured
operand range or workload identity participates in this certificate.
"""

from __future__ import annotations

import hashlib

import numpy as np

from . import quantized_affine_pair as pair


def intervals(domains):
    if not isinstance(domains, list) or len(domains) != 2:
        raise ValueError("two explicit signed-byte operand intervals required")
    for bounds in domains:
        if not isinstance(bounds, list) or len(bounds) != 2 or any(type(value) is not int for value in bounds):
            raise ValueError("two integer endpoints required for each operand interval")
        if not -128 <= bounds[0] <= bounds[1] <= 127:
            raise ValueError("nonempty intervals within signed-byte type required")
    return domains


def derive(source, predictor, domains):
    domains = intervals(domains)
    full = pair.derive(**source, **predictor)
    expected = pair.source_table(**full["source"])
    predicted = pair.predictor_table(**full["predictor"], relu=full["source"]["relu"])
    a = np.arange(-128, 128)[:, None]
    b = np.arange(-128, 128)[None, :]
    mask = (a >= domains[0][0]) & (a <= domains[0][1]) & (b >= domains[1][0]) & (b <= domains[1][1])
    mismatch = expected != predicted
    if np.any(mismatch & mask):
        raise ValueError("predictor differs from original source within admitted operand domains")
    return dict(
        schema="quantized_affine_domain_certificate_v1",
        source=full["source"],
        predictor=full["predictor"],
        operand_intervals=[list(bounds) for bounds in domains],
        admitted_pairs=int(mask.sum()),
        complete_type_pairs=65536,
        source_table_sha256=full["source_table_sha256"],
        predictor_table_sha256=full["predictor_table_sha256"],
        admitted_source_sha256=hashlib.sha256(expected[mask].tobytes()).hexdigest(),
        admitted_prediction_sha256=hashlib.sha256(predicted[mask].tobytes()).hexdigest(),
        excluded_pairs=int((~mask).sum()),
        excluded_mismatches=int((mismatch & ~mask).sum()),
        exact_for_all_admitted_pairs=True,
        exact_for_complete_type_domain=not bool(mismatch.any()),
        source_arithmetic=full["source_arithmetic"],
        predictor_arithmetic=full["predictor_arithmetic"],
        obligations=[
            "Caller binds each interval to actual typed source producers through verified element-preserving uses",
            "Provider independently qualifies predictor arithmetic, layouts, effects and immutable original "
            "observation",
            "Excluded pairs grant no source equivalence or runtime sampled-range permission",
        ],
        performance="UNKNOWN",
    )


def validate(certificate):
    if not isinstance(certificate, dict):
        raise ValueError("domain certificate dictionary required")
    if certificate != derive(certificate["source"], certificate["predictor"], certificate["operand_intervals"]):
        raise ValueError("domain certificate changed after exhaustive admitted-pair proof")
    return certificate


def source_result_interval(source):
    """Bound an exact producer of the complete ordered scalar source observation.

    Caller must prove the actual typed operation implements this scalar graph.
    This does not recognize declarations or trust a result attribute by itself.
    """
    canonical = pair.derive(**source, p=1, q=1, scale=1.0)["source"]
    table = pair.source_table(**canonical)
    return dict(
        schema="complete_scalar_source_result_interval_v1",
        source=canonical,
        minimum=int(table.min()),
        maximum=int(table.max()),
        pairs=65536,
        source_table_sha256=hashlib.sha256(table.tobytes()).hexdigest(),
        caller_binding_required=True,
    )
