# AGENT.md — src/merlin/development

This host-private research domain owns real experiment declarations and their validation.
definition.py preserves the existing YAML/schema/catalog/run-directory implementation.
Its lazy adapter lookup is a temporary high-level validation seam to the existing
merlin_experiments registry. Keep shared compiler, lowering and backend code independent
of this domain. Engine execution, grading, source grants and resume policy stay with
their current owners.

Preserve merlin_experiments.spec module/object/pickle identities through its exact alias.
Source records must observe the real definition and package initializer, literal legacy
compatibility file and the existing schema resource. A move never reseals old receipts
or grants candidate visibility. Logical component ownership and author withholding
must cover the canonical spelling as well as the retained name.
