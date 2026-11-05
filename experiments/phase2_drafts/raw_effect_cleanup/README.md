# Disabled raw effect cleanup draft

Paused at the user’s request before any test or execution. These are exact source
bytes, not an enabled pass, reviewed release, import grant or qualification. No
new test, compiler, Spike or gSIM job ran for this draft.

`manifest.json` records each draft hash and its intended source path. The proposed
shared-library roster change is retained as `library_sources.patch.disabled` and
is **not applied**. Drafts use `.py.draft` suffixes so imports and test discovery do
not activate them. The original active library roster is unchanged.

The proposal is a bounded, standalone generic effect induction: retain original
operation indices, remove only identical persistent state writes or barriers
proved redundant in actual retained state, and reject unproved omissions. The
corresponding target decoder, composition hook and unrun tests live in the OOT
repository at `experiments/phase2_drafts/raw_effect_cleanup`.

Before any restoration, review both repositories together, fix any issues, run
the intended tests, freeze the complete candidate/import closure, and obtain the
original independent source/effect/numerical/execution checks. No baseline or
existing SDK/release is changed by retaining these snapshots.

Activation requires independent execution and calibration evidence for the
selected sources; this draft does not supply that qualification.
