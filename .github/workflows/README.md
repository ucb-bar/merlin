# CI Workflows

- `pr-fast.yml`
  - PR-time fast checks (lint + CLI docs drift + release tracking config validation).
- `nightly-upstream-drift.yml`
  - Scheduled/manual drift checks against pinned upstream commits.
- `release-tracker.yml`
  - Scheduled/manual check for new upstream IREE stable releases and optional
    automatic tracking issue creation.
- `docs-pages.yml`
  - PR docs validation and `main` deployment to GitHub Pages with MkDocs.

Non-CI operational flows (board SSH runs, ad hoc cross deploy) remain as
manual scripts under `benchmark/target/` and are intentionally not part of
GitHub Actions.
