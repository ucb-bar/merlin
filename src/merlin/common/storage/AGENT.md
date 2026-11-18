# src/merlin/common/storage

## Purpose

Storage of generated output: the hard-linked content store, immutable input snapshots, read
leases, what may be deleted, tree operations, and the `merlin-storage` command.

- `content_store.py` — the content-addressed store snapshots link into.
- `bundle_snapshots.py` — immutable input snapshots, guarded file digests and the lexical path
  index.
- `read_leases.py` — sha256 reuse protected by kernel read leases.
- `retention.py` — leases, pins, declared retention and citations: what may be deleted, and the
  roots the storage tools scan.
- `tree_operations.py` — move, link and open-file census.
- `usage_report.py` — the `merlin-storage report` model.
- `cli.py` — the `merlin-storage` entry point (folds, dedup, retention plan, prune, organize).

## Invariants

- CAS-backed snapshots share file modes. Before deleting a read-only bundle, use its snapshot
  cleanup owner; bare TemporaryDirectory cleanup can chmod shared files after an unlink failure
  and invalidate other sealed consumers.
- Content-store copy fallbacks keep public files read-only (0444 or executable 0555), including
  cross-filesystem and disabled-store copies. Explicit permission-preserving copies bypass CAS
  and keep only the source's read/execute bits; they must never widen owner-only access or
  overwrite a shared inode. Destination symlinks refuse; callers own destination paths,
  existing regular files and directory modes.
- Guarded file digests reuse exact selected file reads only inside a host-issued request owner
  and continuous kernel watches. Every file and parent are watched before the first full hash;
  any event, overflow, mount change or unavailable watch restores complete reads. Invalidated
  guards are never rearmed. Explicit joined-worker scopes recheck custody before publishing and
  close with the owner; forked/closed owners cannot reuse. No timestamps, global digest cache,
  source grants or semantic proof authority are introduced. Outside the scope RTL source checks
  retain their ordinary full reads.
- `bundle_snapshots` owns generic original-input custody: V4 archived declarations, canonical
  frozen source mappings, payload digest/counts and separately bound marker bytes. It has no
  experiment dependency and does not decide candidate visibility. The experiment freezer
  delegates payload hashing; its independent mount/answer checks remain mandatory. A verified
  input is not an execution or semantic receipt, and host-only inputs do not become public
  grants.
- Inside one `verified_snapshot_scope`, Linux watches of every file and directory may reuse the
  names/types/sizes roster. Watches include individual inodes so an external hard-link resize
  invalidates the shortcut. Mount-table changes, events (including overflow), unavailable
  watches or closed custody restore complete scanning. Payload hashing and selected read-lease
  checks remain independent at both operation boundaries; this shortcut never authenticates
  payload bytes. An original executor may select its issued `SharedEntry` with
  `shared_entry_scope`; implicit selection stays on the issuing process/thread and each nested
  operation receives a distinct verification token. Every final boundary independently
  reads/hashes payloads or validates their live read leases. A kernel-watched roster may avoid
  the repeated directory walk at that boundary; any event during the read repeats the complete
  scan and payload verification. Independent consumer reads clear both entry and roster reuse.
- Explicit bounded `MERLIN_SNAPSHOT_LEASE_MIN_BYTES` and `MERLIN_SNAPSHOT_LEASE_MAX_FILES` select
  cache admission, never replace a live kernel read lease. Cached path checks pin an O_PATH
  lookup and compare its kernel-resolved ordinary name and inode; replaced, indirect or deleted
  names refuse. Cache admission reserves at least one quarter of the descriptor limit (and at
  least 64 descriptors) for ordinary concurrent work, then rechecks current headroom every 32
  admissions. Existing leases retain custody; new admissions decline to full hashing when the
  available budget shrinks.
- `read_leases.reuse` batches hits for a closed pin set that one process reverifies repeatedly.
  It returns only members admitted earlier by `digest`, after rechecking every member's O_PATH
  name, identity and lease, then every lease again with one custodian check. A failed member
  declines so the caller's ordinary per-file hit reaches the existing refusal; `required=True`
  refuses directly. It never admits, hashes or trusts timestamps. An explicit `minimum_bytes`
  admission threshold changes only which files may be leased, never capacity or custody.
- `tree_operations` refreshes cached lsof listings after a verified move copy and at each
  peer/store duplicate replacement boundary. Census may reuse its original listing; a
  destructive operation may not. Live fuser checks and injected checkers retain their holders
  interface. A refused late check preserves the original name and removes only the operation's
  temporary staged link.
- Nothing is removed on an mtime-based liveness proxy: retention orders units by the timestamp
  token in their own names, and a unit without one is never dropped.
