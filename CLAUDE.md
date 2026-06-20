# Working-tree & branch rules for this checkout

## ⚠️ This is a SHARED working directory — one HEAD for ALL Claude Code sessions
Every Claude Code session running in `/scratch/agustin/projects/oscar-merlin` shares the **same** `.git`
and the **same** checked-out HEAD. There is no per-session branch isolation. Therefore:

- **DO NOT switch branches** (`git checkout <branch>` / `git switch <branch>`) to a *different* branch.
  Switching flips HEAD for **every** concurrent session — it will silently move another session onto a
  different branch mid-task. This has already caused work to land on an unexpected branch once.
- **Commit to whatever branch is currently checked out.** Do not create new long-lived branches for
  parallel work in this same checkout.
- The two active branch names — `feature/kernel-policy-mining` and `feature/rtl-derived-checks` — are kept
  pointing at the **same** commit (one linear history; one is not "ahead" of the other in content). Keep
  committing on the current branch; if you must, fast-forward the *other* name to match afterward so they
  never drift. Never `git reset`/force-push either in a way that drops commits.
- If you genuinely need branch isolation, use a **git worktree** (separate directory) instead of switching
  HEAD in this shared tree.

Rationale: this repo is worked on by multiple concurrent agents. The only safe invariant in a shared
working tree is "everyone is always on the current HEAD; nobody flips it."
