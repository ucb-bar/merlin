# TVM cost-model re-audit

Local bounded checks and source patches, not a FireSim result. See summary.json
for source identities, exact test scopes and limits. The first smoke failure
(missing XGBoost) is preserved; the isolated CPU package and corrected runs pass.
The threshold test uses three warmup samples and two subsequent model-enabled
trials to test the same transition used by the requested 50+5 configuration.
Its constant diagnostic costs cannot establish learned ranking accuracy.

The new kernel-case grouping and wall-budget precedence await clarification.
Use current FIRESIM_HANDOFF.md and the separate kernel benchmark handoff;
frozen older releases retain their original configuration and evidence.
