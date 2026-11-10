# Permanent behavioral suite

The permanent suite contains 150 collected cases: 70 for compiler/runtime
behavior, 50 for verification, and 30 for contracts and workflow integrity.
Parameterization counts toward the ceiling. A small real program and a distinct
negative witness take priority over private helper tests and flag matrices.

The files retain subsystem ownership under `infra/`, `ir/`, `runtime/` and
`targetgen/`. `fixtures/corefixtures/` supplies compiler inputs and independent
execution fixtures; `fixtures/verification_witnesses/` supplies reference inputs
and verification fixtures. Assertions defining the cases live directly in the
collectors. Existing `data/` remains available to real compiler, mining and
reproduction consumers.

```sh
python build_tools/scripts/check_test_budget.py --output out/build/python/ci/test-budget.json
python -m pytest tests -q -ra --junitxml=out/build/python/ci/core.xml
```

Use the selected project environment. Missing native compilers, capture workers,
provers or companion backends remain visible as missing prerequisites; skipped
execution is not qualification. Generated conformance programs, device execution,
model quality and performance use the production verifier/evaluation paths and
their own selected evidence. They are not a renamed archive of retired pytest
matrices.

Historical framework tests and paper instruments remain recoverable from their
recorded Git revisions or frozen inputs. See the
[historical replay guide](../../docs/guides/09-verification.md).
Do not reinterpret a smaller current suite as the original research instrument.
