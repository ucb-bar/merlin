# Patch Stack

This directory is the single home for Merlin's out-of-tree patch stack.

## Contents

- `manifest.env`
  - Canonical pinned base commits and repository paths used by automation.
- `series.iree`, `series.llvm`
  - Ordered patch series definitions.
- `iree/`, `llvm/`
  - Patch files applied on top of pinned upstream commits.
- `tools/`
  - Patch-stack automation scripts (`apply`, `verify`, `refresh`, `drift`).

## Usage

Use the stable entrypoint:

```bash
python3 tools/merlin.py patches apply
python3 tools/merlin.py patches verify
python3 tools/merlin.py patches drift
python3 tools/merlin.py patches refresh
```
