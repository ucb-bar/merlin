# New narrow pure-Lean development context

This coordinator-reviewed context is independent of the held MLIR-full oracle
and bound-qualification-v1. It permits development compilation/audits of new
captured-literal graph proof sources. It does not repair an old failure, approve
an actual IR boundary, issue a graph receipt or complete qualification.

## Concrete evidence now available

- `../pure-runtime-provenance-v1.json` links unchanged original F SDK and W
  fixed-context manifests. Eleven executable/runtime refs match original pins;
  no replacement expected hashes were introduced.
- `../pure-loader-diagnostic-v1/OBSERVATION.json`: one direct pinned loader
  listing with cache disabled, empty built-in hwcaps mask and replaced C locale
  environment resolves ten library paths to those original refs. System preload
  was absent before/after; all eleven pins stayed unchanged. No Lean query ran.
- `../minimal-pure-probe-v1/`: the Init-only dependency command returned zero
  and reported the original `Init.olean`. A known diagnostic trace-parser error
  on empty main-fini/namespace suffix was preserved; this command was not rerun.
- `../minimal-pure-probe-v2/OBSERVATION.json`: the trace was repaired OFFLINE;
  the FIRST proof invocation checked a literal Nat equality with no axioms.
  All3294 unique original SDK/runtime paths matched before/after; original
  metadata covers653 Init modules. Raw loader traces show only known runtime
  libraries and the expected main, with no dynamic loading. The old failed
  frontend profile was never invoked.

The diagnostic invokes the original canonical loader directly, bypassing the
unpinned `/lib64` selector. The library path is SDK/lib, SDK/lib/lean, /usr/lib.
The environment is replaced, not inherited; LANG/LC_ALL=C, LEAN_SYSROOT and
LEAN_PATH are explicit. LD_DEBUG=libs,files is an exact diagnostic setting,
not an arbitrary inherited variable. See retained commands and environment.

This binds actual observed files under explicit OS/kernel/vDSO, loader/module
reporting and host-transport trust. It is NOT a formal proof that Linux can
never search a default directory or make a future late load. Do not convert
those stronger old API flags to true. They remain false and their old inspector
remains unavailable. Formal OS-loader verification is outside the graph goal.

## Authorized next substantial batch

1. Create NEW owned stdlib-only development-control leaves/private outputs for
   captured-literal proofs. Do not import package initializers that load optional
   xdsl/z3/MLIR libraries into the control process. Invoke via absolute Python
   `-I -S -B` paths. No installed-toolchain write, new package/tool download or
   new expected runtime hash. Preserve all old contexts, seals and helper APIs.
2. Use the exact original runtime union and closed launch recipe from the
   successful probe. Validate original parent manifests and every selected SDK/
   runtime binding before/after each build/query. Check system preload absence.
   Preserve full loader diagnostics; reject any unknown library, namespace,
   dynamic load, missing/changed binding or unexpected runtime behavior. Handle
   the observed empty main-fini/[0] syntax explicitly, not by ignoring unknown
   lines. Unknown native signals/mismatches stop this NEW context.
3. Freeze current owned source copies and discover actual source/object imports
   in this new context before compiling the corresponding body. Every standard
   import must be within the ORIGINAL F Init inventory; no affected source bridge,
   Std.Time.Format.Basic, unpinned standard module or ambient project fallback.
   Every project source/object must be explicitly bound in the new profile.
   Byte-copy qualified old objects with their exact refs or build private source
   copies; never rebuild or edit an old capsule. An unqualified source dependency
   remains honestly labeled; compiling a lemma never promotes its old wrapper.
4. Compile/audit the captured-literal candidates in dependency order. Complete
   actual parent/prefix/caching, all-site SSA/return and full-graph correspondence
   rather than producing more unchecked source drafts. Preserve first ordinary
   type/elaboration failures and bounded source repairs under new source identities;
   do not retry an unchanged timed-out/crashed or binding-failed measurement.
5. Add kernel-checked mathematical examples/countermodels and std3 exact-root
   audits. Preserve distinctions between syntactic matching, prefix success,
   initialized tensor values, source graph execution and ABI/native contracts.
   No source-success/whole-program equality premise may replace the local proof.
6. Deliver one coherent checked pure-source batch, context manifest, raw commands/
   outputs/observed imports/runtime traces, exact source patch and paper row. A
   later graph qualification needs actual capture/parser binding, valid negatives,
   frozen original/replay evidence and independent review. Those are not supplied
   by this development authorization.

## Holds that remain

Do NOT invoke the old MLIR-full oracle, rerun bound-qualification-v1, refresh
expected hashes, explain the stale reference by a later matching read or relabel
the historical SDK/source-bridge failures. Exact old failing file/cause remains
UNKNOWN. The old46-case v1 and failed stronger wrapper stay separate.

No numerical work, protected/shared/production/spec/lock changes, new agents or
foreign edits. Only propext/Classical.choice/Quot.sound; no sorry/admit/new axiom,
native_decide/bv_decide/reflection/unsafe/extern/substituted implementations.
Coordinator alone stages/commits/pushes. Continue this batch and the original
captured-literal graph assignment after compaction; a NEW user stop wins.

Reference for reviewed diagnostic behavior: [GNU loader invocation](https://sourceware.org/glibc/manual/latest/html_node/Dynamic-Linker-Invocation.html)
and [Linux ld.so options](https://man7.org/linux/man-pages/man8/ld.so.8.html).
These document the controls; they do not prove the observer or operating system.
