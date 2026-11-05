# Explicit portable host suppliers

`host-capabilities-portable-v3.yaml` retains the attention-v2 source admissions
and replaces five f32 mathematical supplier declarations with an explicit
portable-runtime object contract. The older host files remain historical inputs.
The shared profile permits `portable_exp_trig` and `portable_neural_f32` for
sin/cos/exp; pow/tanh require the latter. Selected model data chooses the policy.
Binary64 sin/cos/pow keep their original selected-archive requirements.

Each portable declaration records the exact recipe source digest, strict compile
flags, wrapper link flags and original C symbol. Native qualification checks the
selected reference policy, compiles that source through the ordinary saved-model
builder, and observes the actual link's `__wrap_*` defining object. Reopening
requires the selected compiler, source, returned object compilation and object
bytes, wrapper flags and real linker diagnostics. An archive cannot substitute
for the portable object, and a portable object cannot satisfy an archive claim.
Missing or historical untraced links require a new ordinary build.

This contract proves source admission and defining supplier only. Source-call
routing, accelerator coverage, instruction admission, whole-chain comparisons
and all existing numerical thresholds remain separate mandatory gates. The new
rows grant no new numerical accuracy claim or CPU contraction permission.
