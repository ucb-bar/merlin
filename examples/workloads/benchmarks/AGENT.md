# Authored benchmark inputs

This directory holds small authored programs and source descriptions for explicit
benchmark and calibration work. It does not hold generated model captures, arrays,
measurement tables or experiment results.

Generated captures and derived corpora belong under ignored `out/`. Select those
artifacts explicitly for a run; never repopulate this source directory as a hidden
fallback. Record the selected compiler, target facts and input identities with
measurements. Authored diagnostic programs alone establish no hardware support,
numerical correctness or cost-model coefficient.

Keep reusable compiler mechanisms in `src/merlin/`. Compiler-generation engines
live in `src/merlin/development/`; experiment selection and records live in
`src/merlin/experiments/`. Do not add a second fixture or harness forest here.
