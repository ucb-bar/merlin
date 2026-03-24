# Contributing

This project is maintained by a small team. Keep changes small, automated, and
repeatable.

## Development Principles

1. Prefer out-of-tree changes in `compiler/`, `samples/`, `benchmarks/`, and
   scripts in this repository.
2. Treat `third_party/iree_bar` and nested LLVM in `third_party/iree_bar/third_party/llvm-project` as pinned dependencies.
3. Any unavoidable in-tree fork edits must be represented in patch files under
   `patches/`.
4. Keep commands scriptable; avoid one-off manual steps without docs.

## Commit Scope

1. Keep PRs focused (one subsystem per PR when possible).
2. Include docs updates for workflow/process changes.
3. If behavior changes, include at least one runnable command in PR description.

## CI Expectations

PRs should pass:

1. Script/python lint gates.
2. Patch-stack verification and drift checks.
3. Any workflow-specific checks touched by your change.

## Releasing Binaries

Linux release artifacts are built locally with Docker; macOS is built by CI.

1. Tag the commit: `git tag v<VERSION>`
2. Build Linux artifacts: `./build_tools/docker/build_release.sh v<VERSION>`
3. Push the tag: `git push origin v<VERSION>` (triggers CI for macOS)
4. Upload Linux tarballs: `gh release upload v<VERSION> dist/*.tar.gz`
5. Edit the draft release on GitHub, add notes, and publish.

The Docker builder (`build_tools/docker/`) uses a reproducible container with
the conda environment baked in. It produces three tarballs in `dist/`:
`merlin-host-linux-x86_64.tar.gz`, `merlin-runtime-spacemit.tar.gz`, and
`merlin-runtime-saturnopu.tar.gz`.

See the "Creating a release" section in README.md for the full walkthrough.
