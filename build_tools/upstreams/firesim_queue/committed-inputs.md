# Opt-in FireSim queue file handoff

This patch is for the explicitly reviewed external `firesim_queue.py` whose
SHA-256 is `3319b5e3aaaf9d6a7ebf70486130848a0d9ba85b0a7276cb2b9d7f3d1181e566`.
Its repository revision is unavailable. Do not apply the patch to a different
source, install it, or restart a shared daemon without operator review.

The patch SHA-256 is
`7c28a1d73af7cb3944d4b8bbb60afeeaabf86f3fc977cc7f3667b888461a3166`.
The resulting queue source SHA-256 is
`af721197a5f6a5ffd0a2a46da2555c52026c2f2f97eafc84c5dcd976406d256a`.

## Installation and explicit selection

The operator installs the reviewed patched script and an adjacent standalone
copy of `src/merlin/common/pinned_files.py`, SHA-256
`081a7014eeff604cda30570fff61329f09484f63a35ba33fa2f1662772bd644e`.
The queue rechecks those helper bytes on every optional use, executes only that
fixed adjacent source, and retains one module instance for exact dataclass
identity. It does not import Merlin or load a submitter-selected module path.
The host Python must support Python 3.10 syntax and provide PyYAML; the concrete
native control workers used Python 3.10 and PyYAML 6.0.1.

All three new options are required together with `--stage-from`,
`--hw-config` and `--hwdb-config-artifact`:

- `--committed-file ABSOLUTE_CANONICAL_PATH SHA256`, repeated once for the
  staged ELF and once for each of the two local archive files.
- `--hwdb-local-uri FIELD EXACT_ORIGINAL_URI SLOT_ARCHIVE_BASENAME`, repeated
  for `bitstream_tar` and `driver_tar`. Each URI must match the exact scalar in
  the original one-entry HWDB and select its committed canonical local file.
- `--consumed-slot RELATIVE_SLOT_DIRECTORY RENAMED_ELF_BASENAME`, repeated for
  every explicitly selected consuming slot.

The declaration rejects missing, duplicate or foreign members, ambiguous YAML
keys/aliases, nonlocal archive URIs, path escapes and archive links/special
files. Different archives cannot overwrite each other's regular members or
declared slot ELF/archive names. Requiring both local archives is a deliberately
restricted opt-in contract; other queue configurations retain the legacy path.

Under the existing FPGA flock, the daemon reopens the committed files and takes
exclusive copies in its protected private snapshot root. It retains the
original HWDB and its hash, rewrites only the selected archive URI fields to
private copies, and separately hashes the effective HWDB. The generated runtime
YAML is rendered under the same lock directly into that private directory,
then privately snapshotted. The opt-in renderer never writes the shared job's
`config_runtime.yaml`; a precreated shared YAML symlink remains untouched.
FireSim receives those private YAML paths.
The selected ELF is staged from its private copy.

Before any FireSim phase, the private inputs, generated configuration and
staged ELF must match. After infrasetup, before runworkload, and after trailing
kill, each declared slot ELF, archive and complete regular-file archive member
is reopened and hash-checked. Successful checks and bounded refusal details are
written in the protected snapshot root. A consumed-byte refusal prevents
runworkload, marks the job failed, and retains mandatory trailing kill once a
FireSim phase began. Diagnostic publication cannot suppress teardown.
The lock remains held through the final checks. Legacy dispatch never loads
the new helper when these options are absent.

## Public manager correspondence

The naming observations came from exact public FireSim commit
`b084672c2f8cf32e55d78f73a001074c23f8a2b8`:

| Source | SHA-256 | Observed relation |
| --- | --- | --- |
| `deploy/runtools/runtime_config.py` | `ac825d8cb358201ca5bb2048b0b2c36e4bc545dfce1d32e41e4607e65d44bc32` | `get_driver_tar_filename()` returns `driver-bundle.tar.gz`; `get_bitstream_tar_filename()` returns `firesim.tar.gz`. `URIContainer` keys its cache by SHA-256 of the exact URI and skips an existing cache entry. |
| `deploy/runtools/run_farm_deploy_managers.py` | `f4b7f30919dad11dc6a902440bc4fce5dc56f86725090edb507ae840342d2acd` | `get_remote_sim_dir_for_slot()` combines the selected simulation directory with `sim_slot_<index>`. Infrastructure copies files through `rsyncdir`, then extracts the selected archives in the slot. |
| `deploy/runtools/firesim_topology_elements.py` | `04c950f2d6f78ac6749327904a5786f895d9177ec7554b9bb5b5cab002286dee` | `get_bootbin_name()` prefixes the bootbinary basename with the actual job name and `-`. |

These source relations explain the explicit options; they are not built into
the queue as platform or slot guesses. The physical control owner must derive
the actual job name, slot indices, host and generated manager roster from its
selected runtime configuration. A caller-supplied slot list alone does not prove
that every consumer was included. Remote slots are unsupported by this local
check. The selected runtime template, workload declaration, environment,
platform loader and live hardware identity remain separate evidence owners.

Unique private job URIs avoid adopting the previous mutable-source URI cache
identity. Slot archive/member checks still detect wrong cached or extracted
bytes. This is not a proof of the bytes programmed during infrasetup or of
continuous file integrity between checks. Actual loaded hardware, driver
consumption, operating-system isolation, successful physical cleanup, complete
outputs, effects and matched timers remain independently required. The patch
grants no compiler, runtime, numerical, physical or performance authority.

## Isolated replay

Copy the exact selected original script to a fresh directory under `out/`,
verify its source hash, then apply `committed-inputs.patch` with
`patch --batch --fuzz=0 -p1`. Verify the resulting source hash and copy the exact
standalone helper adjacent to it. This preparation must not touch the live
installation. This is a context-free unified patch; checking the exact base
and result hashes is mandatory, even when the patch tool reports success.

Run the explicit source-selected controls from the repository root:

```sh
MERLIN_TEST_QUEUE_SOURCE=/path/to/reviewed/original/firesim_queue.py \
  python -m pytest -q build_tools/upstreams/firesim_queue/test_committed_inputs.py
```

The test fixture verifies the original source hash, applies the actual patch,
and creates only owned temporary queues and fake FireSim executables. Real
subprocesses prove that the lifecycle flock is held, stale original-URI cache
entries are not adopted, wrong ELF/archive/member/configuration bytes refuse
runworkload, late changes fail after teardown, and changed submission inputs
launch no phase. The legacy control runs without any installed helper. Missing
external source selection is explicitly skipped; such a run is not the native
qualification requested above. The neutral external original HWDB lifecycle
test source is separately selected by SHA-256
`97abd06301b9f4c7fb8e9485e1155e10aa5a5d78991b772687a6b4fc7e10964e`;
its four original controls must also be run against the patched script before
installation:

```sh
MERLIN_TEST_QUEUE_SOURCE=/path/to/reviewed/original/firesim_queue.py \
MERLIN_TEST_QUEUE_ORIGINAL_TESTS=/path/to/reviewed/test_hwdb_snapshot.py \
  python -m pytest -q build_tools/upstreams/firesim_queue
```

The original test code and assertions remain unmodified; the wrapper only sets
its `QUEUE_MODULE` to the actually patched source. The concrete combined gate
has 33 new controls and four original controls, with both external selections
provided and zero skips. The new tests include an actual ordinary legacy
lifecycle without an installed helper, separately from the original suite.
