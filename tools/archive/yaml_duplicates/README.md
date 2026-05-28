# `yaml_duplicates/` — yaml siblings collapsed by the 2026-05-25 cleanup

## firesim_shuttle_gemmini_os.yaml

Archived from `models/firesim_shuttle_gemmini_os.yaml` on 2026-05-25.

The file's own header says: *"Identical to firesim_shuttle_gemmini.yaml
except this YAML OMITS `--iree-gemmini-use-loop-ws=true`"*. In practice
**neither** the regular nor the `_os` variant had `--iree-gemmini-use-loop-ws=true`
active — the regular file only contained a comment about not adding it. So the
two yamls were functionally identical.

`grep` across `tools/`, `tests/`, `benchmarks/`, `samples/` (excluding tmp/,
build/, third_party/) found **zero external references** to
`firesim_shuttle_gemmini_os` other than the file's own self-reference, so
archiving it does not break any caller.

If the OS-emission debug story re-surfaces and you need an explicit flag
toggle, add a `Gemmini_OS_debug` target variant under `firesim_shuttle_gemmini.yaml`'s
`targets:` section rather than recreating a sister yaml.

Preserved here (not deleted) per the repo-wide preservation rule.
