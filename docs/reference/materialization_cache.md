# Application compilation reuse

Application compilation executes fresh compiler tools and region compilation by
default. Both reuse controls are explicit opt-ins on `native-compile`:

- `--materialization-cache` reuses exact deterministic tool output within that
  invocation when the target declares the capability.
- `--region-materialization-cache` compiles checked canonical regions once and
  binds each original request through the target's publication checks. It is
  independent of tool output reuse.
- `--no-materialization-cache` overrides both opt-ins and runs fresh.

Append both opt-in flags to the normal `native-compile --application` command
to combine them. The supported Python API uses `reuse_materializations=True`
and `reuse_region_materializations=True`, respectively; both default to False.
Targets advertise support through `supports_materialization_cache = True`.
That capability declaration never enables reuse by itself.

Automatic publication verification and standalone `native-verify --application`
always rerun tools fresh in an independent disabled cache session. They never
join or populate the compilation cache. The legacy verification
`--no-materialization-cache` option remains accepted. The Python verification
`reuse_materializations` argument is retained for compatibility and cannot
enable verification reuse.

Status `materialization_cache` counters include the whole invocation, including
fresh automatic verification. Its `compilation` and `verification` maps separate
the two phases; `verification.hits` is always zero. Counters and timing stay
outside authenticated publication bytes. Reuse preserves each original source
replay and per-request publication, including source identity, ABI and bounds.

`--deterministic-receipts` separately opts into target deterministic publication
receipts. Its Python equivalent is `deterministic_receipts=True`. A target must
provide `publication_options(deterministic_receipts=True)` to accept this mode.
Both fresh and reused compilations need this option for literal publication
comparison. Without it, targets retain their original elapsed-time and sizing
records, including operational cache paths. Independently executed default
compilations therefore have baseline timing/path variation. The option does
not enable either cache or weaken independent verification. Deterministic
receipts declare their mode explicitly; ordinary receipts keep their original
verification requirements.
