# Frontend preservation

`archive_frontend.py` completed successfully on2026-10-23 at approximately03:48UTC.
[Manifest](frontend-trees-v1/manifest.json) records three archives, their SHA256 digests,
sizes, original Git HEADs and member-list digests. Each archive passed `zstd -t` and a full
tar listing. This checks archive readability, not a fresh proof/test qualification or an
exhaustive restored-file comparison.

The stopped canonical, byte-compatible and unfinished numerical-context trees remain at
their original `/tmp` paths. Their compressed backups total5,020,093,538bytes under `/home`.
No original tree was moved, deleted, rebuilt or resumed. No publication hold was released.

These are backup archives, not relocated Git worktrees: `.git` pointers and absolute paths
may require deliberate reconstruction before use elsewhere. Tar preserved symlinks without
following them; external linked targets are not included. Existing receipt paths still use
the original trees. Logs: [archive-v1.log](archive-v1.log).
