"""Design-pressure analysis: workload region -> cutpoints -> pressure metrics -> emitted reports.

``pressure_vector.compute_rpv`` turns a region into a Region Pressure Vector, ``synthesize`` maps it to
recommended interface features under the policy set, and ``emit`` writes the schema-validated
design_pressure, interface_candidate and candidate_contracts documents.
"""
