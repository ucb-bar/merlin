# Cross-machine handoff artifacts

## GSIM Radiance branch

`gsim_submission_ASPLOS2026_20260908.bundle` preserves the commits on
`submission/ASPLOS2026-gsim` that could not be pushed to the upstream GSIM
repository because the current account does not have write access.

- Tip: `319a099d27d7f91d0407f39e56f4e1750adf500d`
- Required upstream base: `50b371c60910ebe7aaabb248f8c2b8a56c3a96e6`
- SHA-256: `5573820a94383137465e17f814001e94de02b22319123239bd048a30f6dfa8f3`

After cloning GSIM and fetching its `master` branch, restore the branch with:

```sh
git fetch /path/to/gsim_submission_ASPLOS2026_20260908.bundle \
  submission/ASPLOS2026-gsim:submission/ASPLOS2026-gsim
git switch submission/ASPLOS2026-gsim
```
