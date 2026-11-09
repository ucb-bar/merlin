---
title: Freeze evaluator source before issuing execution owners
kind: guide
status: current
owner: experiments
last_verified: 2026-11-09
related: [phase2_preflight, phase2_public_diagnostic_launch, shared_pass_phase2]
code_refs:
  - src/merlin/experiments/source_snapshot.py
  - src/merlin/experiments/frozen_python.py
---

# Freeze evaluator source before issuing execution owners

Use the existing `merlin.experiments.source_snapshot.create` and
`merlin.experiments.frozen_python.python_command` APIs to separate host evaluator
source from subsequent compiler edits. A source snapshot supplies byte custody;
it does not issue compiler, numerical, hardware, timing, or formal authority.

Assemble ordinary source copies from explicitly selected clean commits in a
temporary task directory. Retain the selected Merlin layout (`src`, explicitly
selected separate distribution sources, `pyproject.toml`, `build_tools/package_resources.json`, and the
required `merlin/{runtime,schemas,contract}` resources). Place selected OOT support
under `oot/<repository>/`, preserving any sibling checker directories used by
the original implementation. Exclude `.git`, environments, runtime binaries,
corpora, generated results, contract examples, and linked capsule collections.
Use committed regular files; copying a support directory alone can lose checker
paths derived from its repository layout.

Seal explicit source roots with `runtime_links=False`. Declare the core and
optional source directories as `python_roots`, in the intended precedence, and
the selected OOT bare-module directory as a `legacy_roots` entry. These are
host-private import roots. They do not change the compiler's separately selected
public SDK or sandbox grants.

Launch an original host entrypoint with:

```python
argv = frozen_python.python_command(
    release,
    [python, str(release / "metadata_controls" / "bootstrap.py"), input_path, input_sha256],
    verifier_source=Path(source_snapshot.__file__),
    instrumentation_from_snapshot=True,
)
```

The bootstrap must reopen its explicit input pin and restore the operator's
selected provider, facts, execution contract, output directory and interpreter
selection after frozen startup. Startup deliberately clears ambient target
selection. Reuse `require_current_native_contract` where an explicit native
contract pin applies. Runtime and hardware inputs remain external, with their
original checks and absolute identities.

Issue a **new** original pass factory from canonical frozen modules. Its record
must include the original canonical issuer and required helper closure; restore
that record through `restore_original_factory`. Do not rewrite callback paths or
source pins in historical records. A successful factory round trip without
`create()` is a source-transport check, not admission.

Construct a fresh original compiler profile against the frozen source release
before target execution. An old profile may correctly reject the new repository
ownership. Likewise, trace and native receipts retain the source paths actually
used by their producers. Their source checks must independently permit reuse;
byte-identical relocation alone supplies no permission.

Several original grader children launch Python directly. Preserve their source
pins, explicit frozen import paths, environment equality, CPU transfer, cleanup,
and deadlines. A frozen parent does not establish guarded descendant imports.
Do not wrap these commands blindly: inherited frozen startup clears target
selectors and can delay the child's original parent-death guard. Any future
transport migration needs its own exact selection and lifecycle controls.

The current shared namespaces are `src/merlin`, `src/merlin_analysis` and
`src/merlin/experiments`. Declare their actual selected source root once; add
separately distributed optional sources only when the snapshot explicitly
contains them. Retained historical snapshots keep their original root layout,
module identities and verifier rather than being rewritten to the current tree.
