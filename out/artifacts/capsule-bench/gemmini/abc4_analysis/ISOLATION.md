# Isolation & fresh-start guarantees — what's shared, what's denied, how it's enforced

Answering: *do all arms start a new experiment from scratch, and are we sure no arm sees prior
runs / prior implementations / the oracle?*

## Fresh-start: YES, with precise semantics (verified on abc4)
- **Each arm's round-0 submission is EMPTY**, assembled fresh from its bundle's `allowed` list. No prior
  implementation is ever copied into the workspace.
- **The bundle `allowed` list contains NO prior backend / submission / oracle** — verified, it is exactly:
  `bench_contract/` (the problem + schemas), the public ISA headers (`gemmini.h`, `gemmini_params.h`),
  `task/`, the LLVM/MLIR toolchain, the `hwbringup_gemmini_v0` RTL info-set (RTL + README + 4 example
  kernels), and `agent_selfcheck.py`. That's the *shared* surface — identical for all three arms.
- **Across experiments** (abc1 vs abc4): different run-ids → different empty workspaces → from scratch.
- **`--resume`** reuses **only THIS run-id's own** workspace+submission (cross-rate-limit-window
  continuation of the *same arm's own* authoring); a non-resume launch refuses to touch an existing run
  dir. It never pulls from another arm or another experiment.
  - *Nuance:* because abc4 was resumed across quota windows, the arms are the same agent's authoring
    continued across sleeps — **not** a single uninterrupted session. That is intended (robustness), and
    it does not import any other run's work. A pure single-session protocol (no `--resume`) is available
    if a cleaner "one continuous from-scratch session" is wanted.

## What is DENIED (the answer surfaces) and the two-layer enforcement
| Denied surface | Layer 1: bundle `denied` (workspace mask) | Layer 2: `chmod 000` on disk (verified) |
|---|---|---|
| prior backends `generated_targets/gemmini/{agent_spec_v0,agent_spec_v1,merlin_native_v0,hand_smoke_oot}` | ✅ | ✅ `d---------` |
| prior run submissions `runs/*/*/submission`, `_qa_work`, `qa_history` | ✅ (`runs/` denied) | ✅ (9 locked) |
| other on-disk gemmini-opt backends `/path/to/agent_bench/**` | n/a (out of repo) | ✅ locked this session |
| the oracle `merlin/runtime/{reference,simulator}`, `reference_outputs` | ✅ (`merlin/` denied) | — (same-uid; covered by audit + integrity scan) |
| capsule `golden.yaml`, hidden capsules, grader-private | ✅ | golden withheld at workspace build; hidden denied |
| experimenter `~/.claude` memory | n/a | ✅ locked |

## The known limitation + why we're still safe
`sandbox=none` (bwrap crashes the Bun-based `claude` binary here), so an agent *can* read any
**unlocked** absolute path. Mitigations, in order of strength:
1. **`chmod 000` on every known answer surface** (the real lock — an unreadable file can't be read).
2. **Post-run cheat-scan** (`transcript_tooling_audit.py`) — hard-**disqualifies** any run whose transcript
   shows a prior-backend path (`generated_targets`/`agent_spec`/`agent_bench`/`preflight`/`_qa_work`/…),
   a `golden.yaml` read, or an oracle import. Now hardened to cover the on-disk backends too.
3. **Integrity scan** on the shipped package (no `import merlin`/oracle in the submission).

**abc4 status (all verified this session):** locks intact (`d---------`); cheat-scan **PASS** on all 3
arms; the C++ transcript references **none** of 13 leak vectors; the submission is a real generic compiler
(626-line shape-driven `Backend.cpp`, 0 hardcoded capsule names). The C++ result is legitimate.

## Why C++ went from "bad first run" to fast — not suspicious
The "bad" early run (abc1) had **no toolkit**: no self-check tool, no RTL/repo, no example kernels — the
agent authored blind against a redacted spike-only verdict (4 rounds, partial). abc4 gives the **realistic
dev environment** (fast spike self-check + RTL/ISA + 4 worked example kernels + "stop when correct"), and
**all three arms** jumped to 25/25 — C++ included. The improvement is fully explained by the toolkit
change (blind → real dev loop), applies to every arm, and is independent of any leak (verified above).
C++ is efficient simply because it targets mature MLIR directly with little ceremony.

## Pre-launch isolation checklist (run before every new experiment)
1. `chmod 000` all `generated_targets/gemmini/*` backends, all `runs/*/*/submission`, `agent_bench`,
   `~/.claude/projects/*/memory`. Verify each is `d---------`.
2. `--dry-run` the launcher; confirm each arm's bundle `allowed` has no submission/backend, `denied` lists
   the answer surfaces.
3. After the run, `transcript_tooling_audit.py` on every arm → require **cheat-scan PASS**; treat any
   DISQUALIFIED as invalid and re-run.
4. Restore perms (`chmod -R u+rwX`) only after the experiment + audits complete.
