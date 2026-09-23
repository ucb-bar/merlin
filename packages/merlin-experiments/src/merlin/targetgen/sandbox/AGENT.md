# AGENT.md — merlin/python/merlin/targetgen/sandbox

## Purpose

Shared, descriptor+manifest-driven agentic bwrap sandbox — a new target gets a correct, continuously-guarded isolation sandbox from its `target_experiment.yaml` (+ capability manifest) with ZERO copied scripts. Routing is by compute-unit KIND / sim FAMILY, never a target name.

## Modules

New bundle freezes use V4 captured support ownership. Preserve historical V2/V3
inspection, but require a newly frozen run for verified execution. Ownership
metadata is bound by the existing host environment marker, never by the mutable
marker alone; final masks and Phase 2 public projections use captured private
views even after original aliases or selected providers change.

- `answer_surfaces.py` — The DERIVED answer-surface mask set (goldens/model weights/hidden/prior/oracle/grader/memory) and transcript-audit tokens, consuming layout-independent identities from `merlin.common.access`.
- `toolchain.py` — The legit tools bound back over the deny-by-default masks: universal + the descriptor's `sim_via` family, cross-checked by `kind` via `merlin.targetgen.families`.
  Importing this owner must not resolve a checkout or tool installation. `ToolchainPaths`
  supplies immutable explicit universal paths; omitted runtime configuration retains the
  legacy checkout resolver. `universal_probes()` derives probes from the selected paths,
  not import-time constants. Explicit universal paths do not qualify simulator-family
  configuration, sandbox operability or native tool execution.
- `bwrap.py` — Deny-by-default argv assembly + the hermetic mount-table replay that PROVES no answer surface is reachable (coverage guard), without launching bwrap.
- `__init__.py` — `build_sandbox(descriptor, ws, bundle)` → a `Sandbox` facade (argv / env / wrap / coverage_gap); `resolve_kind` for family routing.
- `build_dependencies.py` — One-call host-owned pure-build source grants, preserving the existing answer masks and exact worker/request/tool pins. Never an agent mount API.
- `preflight.py` — Whether the sandbox can actually be BUILT on this host, as a named condition (ok / absent / inoperable / unknown) rather than a `which` hit.
- `cleanroom.py` — Prevention by construction when there is no kernel isolation: a workspace materialised so the answer key is ABSENT, verified independently of how it was built.
- `read_audit.py` — Detection by audit: what a run's agent event log says was read, as CLEAN / CONTAMINATED / UNKNOWN.
- `executable_dependencies.py` — One exact host-adapter-pinned raw engine leaf, with argv/ELF/receipt revalidation and unchanged answer masks; no implicit backend or data grants.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

## Invariants / notes

- DERIVE, never per-target-branch. The mask set comes from the descriptor's corpus/`answer_surfaces` + the declared `ORACLE_MODULES`/`GRADER_MODULES` registry; the toolchain routes on `sim_via` (declarative `SIM_TOOLCHAINS` table) + `kind`. There is no `if target ==` anywhere.
- Selected support-provider roots are withheld as whole packages, including sibling
  conformance, build-helper, tool and data files. Metadata discovery must not import
  provider code or suppress invalid provider selection. Candidate compiler and host
  schedule roles are not support. Only the existing explicit `contracts/` exemption
  survives; the descriptor's historical backend location remains withheld too.
- The coverage guard is the drift/cheat gate: `bwrap.coverage_gap(argv, surfaces)` replays the ordered mount table and returns any answer surface still reachable — it MUST be empty. The historical cheat gap (a hard-coded slug that left the `~/.claude` memory dir unmasked) is exactly a non-empty gap; the memory dir is derived from the current repo path so it can never go stale.
- `apply_answer_masks` only masks a surface a bind would otherwise RE-EXPOSE (so it never `/dev/null`-overlays a path whose parent is an empty tmpfs — that mount would fail). Surfaces already hidden by deny-by-default need no overlay.
- `apply_final_answer_masks` is the shared final policy for complete runtime bind lists: descriptor
  answers plus verified host-input mappings, including live/frozen aliases and private snapshot
  metadata. Invoke it after grant reapplication; never append new runtime binds afterward. It
  preserves the existing snapshot verification authority and does not rehash full corpus payloads
  per shell command. Argv visibility tests are not operating-system isolation qualification.
- The coverage guard cannot see a rule that was DROPPED, so `answer_surfaces.dropped_declarations(te)` is its required companion for required paths and plugin discovery. `answer_surfaces()` filters to paths that exist because masking an absent path is a no-op mount that fails. A caller making a fairness claim checks BOTH (`Sandbox.containment_record()` returns them together); the registry-driven eviction sweep returns its failure instead of swallowing it. Logical module identities in `merlin.common.access` are different: historical OOT and optional research implementations may be absent, and `unresolved_modules` records them for migration audits while `module_locations` masks every installed physical copy.
- A target's codegen packages are withheld by DIRECTORY (`out/artifacts/targets/<target>/`), not by the descriptor's list of names — an enumeration is stale the next time a mining/autotune/targetgen run mints a package, not the next time somebody edits the descriptor (measured on gemmini: 4 of 40 named, hand-authored `hand_v0` uncovered). A way in exists only where the descriptor's `answer_surfaces.prior_backend_exemptions` declares one, and an exemption naming a package that does not exist is itself a dropped declaration. Only THIS target's subdir is withheld: another target's packages are legitimately granted as cross-target baselines.
- Continuously guarded by `merlin/tests/infra/test_sandbox_isolation.py` (hermetic policy assertions for every roster target + a guarded live bwrap probe).
- Before moving source/resources, register the new namespace/location in `merlin.common.access` and
  compare `unresolved_modules(repo_root())` before/after: no implementation may disappear from the
  physical mask because of its move. Historical OOT identities can remain unresolved, but retain their
  audit identity. Both old and new copies are masked during compatibility; extension namespaces do not
  inherit the public input-dialect import exception. `test_access_registry.py` exercises relocated
  masks with an unmasked negative control, and keys persisting after an optional scorer is absent.

## Containment when bwrap cannot run

`bwrap` is the strong mechanism and is preferred whenever `preflight.probe_sandbox().usable` says it can be built. It cannot be built on a host with `kernel.apparmor_restrict_unprivileged_userns=1` and a non-setuid binary: namespace creation succeeds and the uid-map write is refused, so every invocation dies at `bwrap: setting up uid map: Permission denied`. Two mechanisms cover that case, and neither is sufficient alone — say so rather than overclaiming:

- **`preflight`** — presence is not operability. `shutil.which("bwrap")` succeeds on a host where the sandbox cannot run, so a refusal guarded by a presence check does not fire, the run proceeds behind a prefix that always exits nonzero, and the harness reports the agent as having produced nothing. The probe RUNS a trivial command inside a minimal sandbox and returns one of four named conditions; `usable` is true for `ok` alone, so `unknown` never reads as a pass. The phase-2 performance stage and its launcher both gate on it (`perf_agent_stage._require_working_sandbox`, `run_agentic_perf_experiment.preflight`), so a dead host refuses at launch and `--dry-run` answers NO-GO instead of GO; the probe record travels in the sealed candidate record, and `verify_sandbox_containment_evidence` refuses one that does not evidence a sandbox that could be built.
- **`cleanroom`** — materialises only what an arm may see, so the answer key is not present to be read rather than present-and-denied. Guarantees a property of the TREE: no answer surface is reachable by walking down, and nothing in it walks back up into the checkout (no escaping symlink, no `.git` pointer, no room inside the checkout, no writable shared inode). **Verification is independent of construction** — it re-derives the surfaces and judges the bytes actually present, so a broken builder cannot certify its own output; `test_mutation_disabling_the_deny_rule_is_caught_by_verification` holds that property.
- **`read_audit`** — a clean room does not stop a process opening an absolute path elsewhere, and without kernel isolation or a separate uid nothing does. So the run is audited afterwards from its event log. **UNKNOWN never reads as clean**: a missing, truncated, unparseable or schema-unfamiliar log yields UNKNOWN, `is_clean` is true for `CLEAN` alone, and `require_clean()` raises otherwise.

### Precedence: deny wins, and only a SURFACE may declare a way in

The mount policy resolves a grant against a mask by longest destination, so a deeper grant beats a broader deny there. The clean room deliberately does NOT use that rule. Under it the decision about what escapes a deny belongs to whoever writes the bundle, and an answer key is re-admitted the first time somebody grants a subdirectory for a good reason, with nothing recording that it happened. Instead:

- a withheld surface wins over any grant, at any depth;
- unless the SURFACE itself declares a grantable sub-path (`AnswerSurface.grantable`, from `GRANTABLE_SUBPATHS_BY_ORIGIN`) — today only a target package's `contracts/`, because the package is denied as a blanket precisely so an arm can still read the facts it is required to derive from;
- and reaching inside an exemption takes an EXPLICIT grant at or below it, so a broad grant that merely contains it does not pick it up.

Widening containment is therefore a reviewed edit to `answer_surfaces.py`, not a manifest line. The clean room is consequently stricter than the mount policy for every surface without a declared exemption; that asymmetry is deliberate and in the safe direction.

### The audit's three outcomes

Hits are advisory (the protection worked, or nothing was read), violations (withheld content demonstrably reached the agent), or INDETERMINATE — a withheld path was named and the outcome is not recoverable, typically because the captured output belongs to a whole compound pipeline. Indeterminate hits drive UNKNOWN rather than being guessed either way. Token matching is by shape, not bare substring: a bare module stem must BE a path component, because substring-matching `decode` against shell text flagged every `jq` filter mentioning a `decoded` field (30 of 34 accusations on the archived transcripts, all false).
