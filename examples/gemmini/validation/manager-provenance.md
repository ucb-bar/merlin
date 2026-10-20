# Install full-model ELF provenance on the manager

This integration records which complete-chain ELF was downloaded and which file
the simulator actually consumed. It runs after either compiler freeze. It does
not change V26, quantization, numerical tolerances, or accelerator placement.
Only full-model validation uses the FPGA; the tests below are local and unpaid.

The current manager wrapper is outside this repository. The supplied
[SSM document](FsimRunElf.provenance.yaml) is a proposed next version, not a
deployment receipt or a replacement for that wrapper's FireSim lifecycle.
Administrator installation is required before activating it. Preserve the old
document version and historical results unchanged.

## Code and deployment identity

Workhorse staging is `merlin_experiments.post_freeze.elf_upload`. It uploads the
selected ELF and exact prepared request bytes under unique keys, requires
non-null S3 versions, downloads those exact versions, and retains raw responses.
It produces an **unsent** `SendCommand` plan. It starts no manager or FPGA.
The selected AWS CLI must support create-only `PutObject` (`IfNoneMatch`);
staging checks this locally and refuses older clients before any API call.
The installed workhorse CLI 1.22.34 needs upgrading for this path. Do not remove
the create-only guard to accommodate it.

```sh
python -m merlin_experiments.post_freeze.elf_upload /absolute/request.json \
  --model MODEL --output /absolute/new-staging-directory --upload
```

This command performs S3 uploads, so use it only after the manager/document
pins and CLI upgrade are ready. The old two-argument `fpga-run-elf` helper does
not carry this new contract; its administrator-owned update must consume the
prepared plan, retain the raw command response, enforce its elapsed deadline,
and perform the existing lifecycle cleanup. A staging success is not deployment.

The manager/runtime helpers are these three standard-library-only files in
`packages/merlin-experiments/src/merlin_experiments/post_freeze/`:

- `elf_custody.py`: versioned download; before/after observations on the runtime
  host holding the final boot operand. The same pinned observer runs both times.
- `chain_observation.py`: ordered UART step/array counts and target counters;
  complete failed comparisons remain failures. This does not replace the
  existing selected-reference checker.
- `manager_result.py`: binds the request, original command, document source,
  download, actual consumed-file observations, output files, and cleanup outcome.

They can be deployed under a fixed Python namespace directory without installing
the compiler. Bind all installed helper SHA256 values in the administrator-owned
manager contract, and have the fixed wrapper check those values before use.
The pinned contract has schema
`merlin.post_freeze.firesim.manager_contract.v1`, an explicit
`boot_argument_prefix`, and `helpers` mapping exactly `elf_custody`,
`chain_observation`, and `manager_result` to `{sha256, bytes}` identities.
Preserve its existing hardware/build/runtime fields as additional fields. The
prepared request embeds that verified contract definition so the manager can
check it without resolving workhorse-local paths. Both observers and the
publisher refuse helper bytes or boot operand selectors outside that contract.
Use Python 3.10 or newer. The SSM document uses
[environment interpolation](https://docs.aws.amazon.com/systems-manager/latest/userguide/documents-syntax-data-elements-parameters.html);
it intentionally fails if the agent does not provide those variables. It passes
literal arguments to a fixed wrapper and checks that wrapper's selected SHA256.
No request supplies a host command, script path, or execution environment.

The selected deployment adds three fields: `ssm_document_version` (a concrete
numeric version), `ssm_document_sha256` (the exact returned document `Content`
UTF-8 bytes), and `manager_entrypoint_sha256`. Stage with those exact fields;
never use `$LATEST` or `$DEFAULT`. The
[SendCommand API](https://docs.aws.amazon.com/cli/latest/reference/ssm/send-command.html)
supports selecting the document version and SHA256 explicitly.

## Changes to the fixed wrapper

Keep its existing environment setup and normal FireSim launch/kill/terminate
sequence. Apply these insertions to the real wrapper source before activation:

1. Validate all arguments and select the fixed bucket, region and deployment
   configuration. Hold the existing single-FPGA lock through cleanup. Enforce
   `job-wall-seconds` as an elapsed deadline including setup and cleanup, capped
   at six hours. Retain actual exit, cancellation, timeout and teardown outcomes.
2. GET `requestKey` with **exactly** `requestVersion`, retain the raw response as
   `request-get-object.json`, and check the actual request file against
   `requestSha256` before using it. Do not reopen workhorse-local paths on the
   manager. Request metadata chooses assets, not host commands.
3. Download the ELF with `elf_custody download` using the exact key/version and
   expected SHA256/size. Retain `download.json` as `elf-download.json` and its
   byte-exact `get-object.json` as `elf-get-object.json` in the result directory.
4. Save the actual simulator argv as a JSON string array when the normal manager
   constructs it. After final runtime-host staging, before starting that
   simulator, run `elf_custody before` **on that runtime host**, with its actual
   working directory, boot argument prefix (this deployment uses `+prog0=`),
   and generated driver script. Collect the resulting `elf-before.json` through
   the manager's existing authenticated runtime-host transport. Hashing a manager
   staging copy is insufficient. Keep the boot file and script read-only and
   exclusively owned for the duration of the run.
5. Run `elf_custody after` on the same runtime host before teardown deletes its
   files. It requires unchanged bytes, inode/stat identity, driver script,
   observer and runtime boot identity. Collect `elf-after.json` alongside the
   byte-exact UART, hardware summary and generated `sim-run.sh`. These observations
   do not independently establish filesystem isolation or prevent a malicious
   writer from restoring bytes between observations.
6. Preserve the raw original `SendCommand` response as `ssm-command.json`, and
   exact-version `GetDocument` response as `ssm-document.json`. The workhorse
   must publish its full command response under
   `firesim-commands/<submissionId>.json` after submission; use a bounded wait.
   Bind its CommandId to the actual SSM orchestration context on the manager;
   never infer it from a workload name. GET responses alone do not identify the
   S3 key: check the key/version parameters against the fixed selected request.
7. After normal execution and cleanup, write `execution.json` with schema
   `merlin.post_freeze.firesim.manager_execution.v1`, actual `ssm_command_id`,
   integer `returncode`, and `cleanup: {complete: <actual bool>}`. Retain detailed
   kill/termination observations as additional fields. Missing teardown proof
   remains false. Publish the manager manifest with `manager_result`, then upload
   the whole unique result prefix. A malformed/truncated transport issues no
   success manifest; retain the failure and raw evidence.

The document's root-owned installation and fixed wrapper must prevent callers
from choosing arbitrary manager source. The existing workhorse role cannot
install the wrapper or update the document. Extend only the required scoped
S3 permissions for versioned requests and command-response evidence; preserve
the single locked SSM entry point. Do not enable general shells or direct F2 APIs.

## Helper calls at the insertion points

All paths below are administrator-selected local paths. `SIMULATOR_ARGV` contains
the concrete driver argv, not a shell string or a reconstructed planned command.

```sh
python3 -m merlin_experiments.post_freeze.elf_custody download \
  --bucket "$BUCKET" --key "$ELF_KEY" --version "$ELF_VERSION" \
  --sha256 "$ELF_SHA256" --size "$ELF_BYTES" --region "$REGION" \
  --directory "$NEW_DOWNLOAD_DIRECTORY"

# On the actual runtime host, immediately before its driver execution:
python3 -m merlin_experiments.post_freeze.elf_custody before \
  --download-record "$DOWNLOAD_RECORD" --request "$REQUEST" \
  --request-sha256 "$REQUEST_SHA256" --model "$MODEL" \
  --argv-json "$SIMULATOR_ARGV" --cwd "$SIMULATOR_CWD" \
  --boot-argument-prefix '+prog0=' --driver-script "$SIM_RUN_SCRIPT" \
  --out "$BEFORE_RECORD"

# On that same runtime host, after exit and before deleting runtime files:
python3 -m merlin_experiments.post_freeze.elf_custody after \
  --before-record "$BEFORE_RECORD" --out "$AFTER_RECORD"

# On the manager, after authenticated result collection and lifecycle cleanup:
python3 -m merlin_experiments.post_freeze.manager_result \
  --request "$REQUEST" --request-sha256 "$REQUEST_SHA256" --model "$MODEL" \
  --result-dir "$RESULT_DIRECTORY" --command-id "$ACTUAL_SSM_COMMAND_ID" \
  --entrypoint /home/ubuntu/fsim-run-elf.sh
```

The publisher currently handles explicitly selected `all_element_comparison_v1`
complete-chain images. Counts come from actual canonical UART records, not the
request's planned counts. The importer reopens the retained custody and output
records, and still runs the existing compact selected-reference decoder. Neither
publisher nor importer grants final model/placement qualification.

## Local verification and activation

```sh
PYTHONPATH=packages/merlin-experiments/src:src python -m pytest -q \
  packages/merlin-experiments/tests/test_chain_observation.py \
  packages/merlin-experiments/tests/test_manager_elf_provenance.py \
  packages/merlin-experiments/tests/test_post_freeze_firesim.py \
  packages/merlin-experiments/tests/test_post_freeze_elf_upload.py
```

Before activation, exercise the adapted real wrapper with fake AWS/driver tools:
wrong versions/bytes, final-path substitution, changed post-run files, missing
UART records, driver failure and deadline expiry must retain failures and perform
cleanup. Install a new document version only after those wrapper-specific tests
pass. Run the next deliberately selected **full-model validation** through the
approved workhorse helper, then import its exact new prefix. No paid smoke or
capsule job is needed to develop this integration. Historical manifests cannot
be manufactured from matching local hashes or UART outputs.
