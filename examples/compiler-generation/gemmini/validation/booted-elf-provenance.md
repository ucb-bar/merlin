# Lightweight boot-file provenance

For nonadversarial reproducibility, select this mode explicitly in a direct
complete-chain FireSim selection:

```yaml
deployment:
  mode: direct_elf_ssm_on_demand
  provenance_mode: booted_elf_hash_v1
```

Keep the other deployment and asset fields from
[firesim_workhorse.yaml](firesim_workhorse.yaml), including the pinned manager
contract file. Preparation records the installed `fpga-run-elf ELF NAME` helper
as `launch_argv`; it does not select `elf_upload` staging or require installation
of the strict version-pinned manager helpers. Document/version fields, if
retained as deployment metadata, do not become verified custody in this mode.
Omitting `provenance_mode` preserves the existing strict/legacy selection rules.
This mode accepts only `all_element_comparison_v1` complete-chain images.

The administrator retains `booted-elf.json` beside the actual complete `uartlog`,
`HW_CFG_SUMMARY`, and `sim-run.sh`. The record describes the actual boot file on
the runtime host, before or after its run, rather than a staging copy on the
manager. The minimally required record is:

```json
{
  "schema": "merlin.post_freeze.firesim.booted_elf_hash.v1",
  "ssm_command_id": "actual-command-id",
  "bucket": "selected-artifact-bucket",
  "elf_key": "firesim-elfs/actual-upload.elf",
  "agfi": "selected-agfi",
  "observed_utc": "2026-10-20T12:00:00Z",
  "runtime_host": "actual-runtime-hostname",
  "working_directory": "/actual/runtime/sim_slot_0",
  "simulator_argv": ["./actual-simulator", "+prog0=actual-boot.elf"],
  "observation_stage": "boot_file_after_run",
  "consumed_elf": {
    "path": "/actual/runtime/sim_slot_0/actual-boot.elf",
    "sha256": "<SHA256 of the actual runtime boot file>",
    "bytes": 12345
  },
  "uartlog": {
    "sha256": "<SHA256 of the complete collected UART>",
    "bytes": 23456
  }
}
```

`boot_file_before_run` is also accepted. Preserve the complete actual simulator
argv, including its other arguments. Exactly one `+prog0=` must resolve against
the reported working directory to `consumed_elf.path`, and that exact command
must occur in the retained `sim-run.sh`. The collector parses nested shell
`-c` strings as data and executes nothing. An rsync destination or an uploaded
file hash alone cannot establish the boot-file observation.

Use the ordinary `firesim collect` command with `--ssm-invocation`,
`--ssm-command`, and `--elf-upload`. Preserve the original successful SSM
invocation and send-command response, plus the existing upload record containing
the actual versioned downloaded ELF and S3 GetObject response. The bucket/key,
command ID, selected boot hash/size, UART hash/size, AGFI, and complete compact
step/array/reference roster must agree. No `result_manifest.json` or synthetic
manager-result document is required for this explicitly selected mode.

The result records `evidence_class: lightweight_booted_elf_hash`. It is a
runtime-host report cross-checked against the retained artifacts, without a
before/after custody guarantee or adversarial isolation claim.
`scientific_qualification` and `model_validation_complete` remain false.
Complete guest all-element comparison is reported separately; native source,
placement, instruction, hardware correspondence and final model qualification
retain their existing gates. Raw cycles do not establish a clock conversion or
full-system memory-timing equivalence. The strict manager-custody path remains
available through its existing selection and is unchanged.
