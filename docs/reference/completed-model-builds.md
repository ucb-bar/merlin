# Reusing completed model compilation

Private full-model qualification can select `completed_build_reuse` in its
operator-owned private specification. The default remains an ordinary fresh
build. This selection reuses compiler products after a later checker refusal;
it does not reuse that checker result or import a prebuilt qualification.

The closed selection has these fields:

```yaml
completed_build_reuse:
  schema: merlin.phase1.completed_build_reuse.v1
  cache_root: /absolute/selected/out/build/completed-models
  producer_run: /absolute/original/native/run
  source_snapshot: {path: /absolute/producer/snapshot.<digest>.json, sha256: <digest>}
  runtime_archive:
    workspace: /absolute/runtime-archive/workspace
    repo: /absolute/runtime-archive/repository
    manifest: {path: /absolute/runtime-archive/bundle.json, sha256: <digest>}
  sdk_roots:
    - /absolute/complete/compiler/distribution
    - /absolute/complete/interpreter/distribution
    # Include all remaining default loader, library, Python and search inputs.
```

Create the framework source release through the existing
`merlin_experiments.source_snapshot.create` API. Include the complete core,
experiments sources (both its `merlin_experiments` and `merlin` namespaces)
C runtime, contract and schema sources, use `runtime_links=False`, and launch its
producer through `frozen_python.python_command`. The producer's actual guarded
seal, builder module origins and runtime path must match that release. The
historical common-transform profile has its own original source owner and
archive; its repository name cannot stand in for the framework producer.

Create the independent runtime archive using the ordinary
`merlin.targetgen.sandbox.bwrap.materialize_bundle_inputs(workspace, bundle,
repo=repository)` API, with a JSON bundle manifest declaring the selected SDK
inputs. Pin that manifest and retain its V4 `bundle_inputs/snapshot.json`.
The archive may be the original profile archive if it already contains the
complete required inputs; otherwise select a new archive without changing the
original profile. No historical build can retrospectively issue production.

A `bin` directory alone is insufficient. The selection must cover each complete
installed tool prefix, the actual interpreter and venv prefixes, Python import
paths, executable/search paths, standard loader directories and `ld.so.cache`.
The helper also checks the selected Clang/GCC default program/library searches,
GCC sysroot, actual loader-cache entries, and transitive `ldd` dependencies
including RPATH/RUNPATH resolutions outside those default directories. These metadata queries run against
exact tools reopened in the runtime archive. SDK aliases retain their lexical
link, target and complete member bytes; cycles and special files refuse.
Unknown compiler hooks, implicit search paths and unsupported options refuse.
For a system interpreter this conservative policy can require a large system
prefix; contained selected distributions reduce that preparation cost.

The first successful ordinary compiler return records its actual Python
Invocation, completed compilation recipe and consumed products before private
post-build checks run. A hit requires the same explicit producer, input,
candidate, host package, target facts, board, DTS, math policy, options, tools
and complete effective environment and Python path identity (values are
hashed, not published). It retains the original artifact paths
and producer records, creates a fresh consumer receipt, and runs the currently
selected linked-ELF policy. The ordinary private source, host computation,
request/routing, supplier and numerical checks still run afterward. Additional
checker diagnostics may coexist with the retained products; a changed consumed
product refuses reuse. Original product paths must remain retained: deletion
by a later cleanup makes the entry ineligible; recipes are never rewritten to
resurrect deleted artifacts.

Every operation rescans declared membership and reopens bytes. Existing Linux
read leases may reuse payload hashes while the leases remain valid; no stat-only
hash cache is added. Unsupported lease cases retain ordinary hashing. Runtime
and product changes refuse, and the final boundary reopens the selection and
products before publication.

The raw environment binding deliberately permits conservative misses: changing
run paths, import paths or framework transport settings may require a new build.
This mechanism does not infer that arbitrary source edits are checker-only.
Historical runs without the issued production record are ineligible. A hit is
compilation reuse evidence, not a new compilation timing sample, simulator run,
model numerical result or full qualification certificate.
