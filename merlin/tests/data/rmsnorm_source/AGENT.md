# RMSNorm source fixtures

These four 52-line source inputs pin RMSNorm recognition through the linalg
bridge, for one and 128 rows and both gamma operand orders. They are actual
model2MLIR captures of the standalone PyTorch expression in
`merlin/tests/fixtures/capture_rmsnorm_source.py`, not handwritten MLIR or model
numerical qualification. Inputs use deterministic nonzero random values.

Regenerate into a fresh directory with the model2MLIR capture revision in
`merlin/contract/software_pins.yaml` on PYTHONPATH, torch 2.11.0+cpu, and xdsl
0.68.0:

```sh
python merlin/tests/fixtures/capture_rmsnorm_source.py /tmp/rmsnorm-source
```

Compare every file with this directory before replacing the fixtures and
`capture.json`. The receipt authenticates the generator and all four sources.
Missing or changed files fail the tests. Generated target outputs belong outside
this fixture directory.
