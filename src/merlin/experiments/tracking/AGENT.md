# Experiment tracking

`__init__.py` exposes dashboard generation, local serving and the plain run
watcher. `readers/` reads stored records; `pages/` renders self-contained HTML;
`terminal/` owns terminal rendering and read-only host sampling. `console/`
serves the separate interactive console. Its panel sources are loaded by name;
do not remove them based on static imports alone.

Page composition and local serving share `pages/views.py`. Plain summary
formatting lives in `terminal/watch.py` alongside the animated terminal command.
Entry functions must import the actual owners, not retired `live`, `text`,
`phase0` or other former module names. Historical record identifiers are data
contracts and remain readable.

## Invariants

- Read records; never measure targets, grade, build, launch agents or write into
  run directories or measurement stores. Dashboard pages are the output.
- Missing, unreadable and foreign-schema records remain unavailable, never zero
  or a guessed success. Distinguish process liveness from scientific evidence.
- Use recorded outcome fields and their owners' constants to classify results.
  Do not reconstruct qualification from diagnostic text.
- Keep pages self-contained: inline CSS, SVG and JavaScript, without network
  dependencies. Private evaluation material is a count or "present" unless the
  caller explicitly selects operator-private mode; every such page names its mode.
- Live views open inputs read-only and refuse output paths inside any input
  directory. Tail JSONL through the last complete line; reuse sections only when
  their input signatures and relevant clock buckets permit it.
- Label file modification times and reconstructed timelines as such. Host CPU
  and thread samples are resource observations, not simulation evidence or proof
  of exact residency.
