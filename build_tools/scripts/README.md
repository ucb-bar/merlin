# build_tools/scripts

Maintenance and validation scripts.

- `check_structure.py` — verify the repository scaffold is intact:

  ```bash
  python build_tools/scripts/check_structure.py
  ```

- `run_spec_validation.py` — run a pinned Core ATen selection and seal its spec-validation record.
  Supply both host documents explicitly: `--host-capabilities` is the descriptor lane
  checked by `corpus derive --spec-validation` and verification;
  `--component-host-capabilities` is the suite execution and per-case expectation lane.
  The record seals both paths and SHA-256 digests. Verification also reopens the
  component document and refuses changed bytes. For Gemmini, from the worktree root:

  ```bash
  PYTHONPATH=src:packages/merlin-experiments/src:packages/merlin-timing/src:merlin/python \
  PYTHONDONTWRITEBYTECODE=1 /scratch2/lorenhung/merlin/.venv/bin/python \
    build_tools/scripts/run_spec_validation.py \
    --selection "$SELECTION" --corpus-193 "$CORPUS_193" --bounded "$BOUNDED" \
    --output "$OUTPUT" --software-spec examples/gemmini/target/software-spec.yaml \
    --host-capabilities examples/gemmini/platform/host-capabilities-portable-v3.yaml \
    --component-host-capabilities examples/gemmini/platform/host-capabilities-core-aten.yaml \
    -- --target gemmini --facts "$FACTS" --repo "$REPO" \
    --m2m-dir "$M2M_DIR" --m2m-python "$M2M_PYTHON" --device-package "$DEVICE_PACKAGE"
  ```

  Set the shell variables to the selected suite, output, facts, and runner tool paths.
  Put both host flags before `--`, once each. Arguments after `--` go to the suite
  runner. `--record-only` judges existing shards with the same two declarations;
  `--source`, `--case-id`, and `--limit` produce incomplete records that cannot
  satisfy a required derivation gate.
