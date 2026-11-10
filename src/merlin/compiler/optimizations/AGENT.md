# AGENT.md - src/merlin/compiler/optimizations

These are existing selectable optimization strategies. The registry owns one
FeatureRegistry and executes its eager declarations in their original order.
Implementation modules define actual schedule/pass builders without registering
on import. Their on-demand declarations import the registry only when called;
this prevents import cycles without proxying globals or wrapping callbacks.

Keep the empty selection unchanged, retain exact schedule text, prerequisites,
feature spellings, ordered hooks and explicit numerical/effect conditions.
Generic upstream vector schedules are reusable compiler mechanisms. Existing
register-group facts/cflags and other backend helpers remain explicit delegates
to their legacy owners during migration; no target kernels or encodings belong
here. Native/source receipts use implementation_paths without initializing the
catalog. Frozen SDK inventories stay immutable.
