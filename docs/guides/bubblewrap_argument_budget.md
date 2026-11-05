---
title: Prepare an explicit bubblewrap argument budget
kind: guide
status: current
owner: experiments
last_verified: 2026-10-12
related: [verified_compiler_tooling, targetgen]
code_refs: [build_tools/scripts/build_bwrap_argument_budget.py, packages/merlin-experiments/src/merlin/targetgen/sandbox/bwrap.py]
---

# Prepare an explicit bubblewrap argument budget

Large frozen tool inventories can require more than 9,000 bubblewrap arguments.
The public [bubblewrap 0.9.0 source](https://github.com/containers/bubblewrap/tree/v0.9.0)
enforces a global `MAX_ARGS = 9000`, including arguments read from `--args` file
descriptors. Descriptor transport avoids Linux argument-byte limits but does not
raise this separate parser bound. Preserve the complete ordered mount policy.

## Build locally from pinned public source

Obtain the [0.9.0 release archive](https://github.com/containers/bubblewrap/releases/tag/v0.9.0)
and independently select its SHA-256 before running the recipe. The retrieved
release archive used for the local control had SHA-256
`c6347eaced49ac0141996f46bba3b089e5e6ea4408bc1c43bab9f2d05dd094e1`.
That digest identifies bytes; it is not an upstream signature verification.

Run from the checkout with canonical, explicit tool paths and a fresh task-local
output directory. For example, after selecting the corresponding variables:

```sh
python build_tools/scripts/build_bwrap_argument_budget.py \
  --archive "$BWRAP_ARCHIVE" --archive-sha256 "$BWRAP_ARCHIVE_SHA256" \
  --max-arguments 65536 --output "$TMPDIR/bubblewrap-65536" \
  --meson "$MESON" --ninja "$NINJA" --cc "$CC" \
  --pkg-config "$PKG_CONFIG" --build-python "$BUILD_PYTHON" --jobs 2
```

The supported explicit bound is 9,001–131,072, with at most four build workers.
The recipe verifies archive bytes before extraction, rejects escaping/ambiguous
members, preserves the release's internal license link, and changes only the
unique static `MAX_ARGS` declaration. Source inventories assert that no other
file changed. Meson retains its ordinary optional security-feature detection;
the recipe does not disable a sandbox feature or run an install command.

The output includes `build/bwrap`, `max-arguments.patch`, original/patched source
inventories, build logs and `build-record.json`. The record pins the archive,
patch, recipe, selected build tools, declared dependency inputs, build options
and resulting binary. It hashes the effective build environment without storing
its values. It does not qualify the complete compiler/system library closure,
sandbox policy, or experiment deployment.

If development headers are unavailable, extract a pinned distribution development
archive into the task directory without installing it. Provide a task-local
pkg-config file naming those headers and the selected library. Record the archive
with `--dependency-archive PATH SHA256`, the actual used header/library/pkg-config
files with repeated `--dependency-file PATH`, and its directory with
`--pkg-config-path PATH`. Rewriting that local pkg-config file is dependency
staging, independent of the single bubblewrap source patch. Declared dependency
files are reopened after the build; undeclared/transitive dependencies remain
outside the record's completeness claim.

## Select and qualify on the worker

This recipe never installs or automatically selects its binary. The operator must
explicitly select `build/bwrap` through the existing worker `PATH`/tool selection,
record its SHA-256 and rerun the ordinary sandbox/deployment qualification. Do not
substitute it during an already frozen experiment without requalifying the
original tool/environment selection.

Host namespace admission can depend on the executable path. For example, a host
AppArmor profile may admit user namespaces only for `/usr/bin/bwrap`; a new local
path then fails even if its argument parser succeeds. Qualify the selected
binary under the same original policy. A host with an existing admitted `bwrap`
profile can explicitly use `aa-exec -p bwrap -- SELECTED_BWRAP ...` for a CPU
control, recording that selector and profile as part of the environment. The
recipe neither changes system profiles nor inserts that prefix automatically.
Changing namespace flags or dropping masks is not a remedy.

A useful small control submits more than 9,000 arguments with the exact same
namespace flags and ordered masks to the stock and rebuilt binaries. The stock
parser should refuse its bound; the rebuilt binary must execute, keep each
private file masked, preserve a public sentinel and leave the original host
files unchanged. This control establishes finite parser/isolation behavior on
that worker, rather than a full experiment or hardware qualification.
