# CI Workflows

- `pr-fast.yml`
  - PR-time fast checks (lint + patch gate).
- `nightly-upstream-drift.yml`
  - Scheduled/manual drift checks against pinned upstream commits.
- `riscv-cross-build.yml`
  - Manual cross-build workflow for selected RISC-V profiles.
- `release-tracker.yml`
  - Scheduled/manual check for new upstream IREE stable releases and optional
    automatic tracking issue creation.
