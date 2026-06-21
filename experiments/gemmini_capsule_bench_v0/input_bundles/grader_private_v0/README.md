# grader_private_v0

Grader-private resources, used ONLY in the post-agent grading phase (full host access, never mounted
into an agent workspace): the hidden capsules + goldens, the `capsule_grade`/`capsule_runner` grader,
the VCS/FireSim adapters, and the `merlin.runtime` oracle side. Hidden capsules are graded only after
the submission is frozen + hashed. Authoritative spec: `input_bundle_manifest.yaml`.
