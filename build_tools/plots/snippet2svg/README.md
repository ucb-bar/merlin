# snippet2svg

`snippet2svg` turns source excerpts into tightly cropped, native SVG for the
Merlin paper. MLIR uses LLVM's real TextMate grammar. YAML, Python, JSON/JSONC,
shell, C++, assembly, and LLVM IR use Shiki's bundled grammars. Unknown file
extensions render as plain text unless `--lang` is given.

The tool only reads its input. It does not format source, run compiler passes,
or change VS Code settings. It emits no editor frame, title bar, shadow, HTML,
raster image, external stylesheet, or embedded font.

## Install

From the repository root:

```sh
npm ci --prefix build_tools/plots/snippet2svg
```

Node 20 or newer is required. Rendering is offline after this command.

## Render one snippet

```sh
build_tools/plots/snippet2svg/snippet2svg path/to/example.mlir \
  -o out/artifacts/paper-figures/snippets/example.svg
```

The `mlir2svg` wrapper is an equivalent convenience name for MLIR files:

```sh
build_tools/plots/snippet2svg/mlir2svg path/to/example.mlir \
  -o out/artifacts/paper-figures/snippets/example.svg
```

The language is inferred from the suffix. Use `--lang` for a suffixless excerpt:

```sh
build_tools/plots/snippet2svg/snippet2svg excerpt --lang yaml -o excerpt.svg
```

Supported language names are `mlir`, `yaml`, `python`, `json`, `jsonc`,
`shellscript`, `cpp`, `asm`, `llvm`, and `text`. Common aliases such as `py`,
`yml`, `bash`, `sh`, and `ll` also work.

Render every supported source file below one directory with one initialized
highlighter:

```sh
node build_tools/plots/snippet2svg/render-directory.mjs path/to/snippets \
  -o out/artifacts/paper-figures/snippets/my-figure \
  --transparent
```

This writes one `.svg` per input plus a deterministic `manifest.json` containing
source/output hashes, languages, and SVG dimensions. Output names retain the
input suffix (for example, `capsule.yaml.svg`) to avoid collisions.
Omit `--transparent` when a white background is desired.

## Select source lines

The range is inclusive. The full file is tokenized before the range is selected,
so a cropped MLIR excerpt retains grammar state from earlier lines.

```sh
build_tools/plots/snippet2svg/mlir2svg path/to/example.mlir \
  --lines 6:18 \
  -o out/artifacts/paper-figures/snippets/example-lines-6-18.svg
```

Add original source line numbers with:

```sh
build_tools/plots/snippet2svg/mlir2svg path/to/example.mlir \
  --lines 6:18 \
  --line-numbers \
  --font-size 18 \
  -o out/artifacts/paper-figures/snippets/example-numbered.svg
```

The displayed numbers remain 6 through 18. An invalid or out-of-bounds range is
an error.

For a short paper excerpt composed from several parts of the same real file,
use ordered, non-overlapping segments:

```sh
build_tools/plots/snippet2svg/snippet2svg command_buffer.json \
  --segments 24:30,52:69,78:84 \
  -o out/artifacts/paper-figures/snippets/command-buffer-short.svg
```

The renderer tokenizes the complete file first and inserts a visible `⋮` at
every omitted region. It never joins non-contiguous source silently.

## Typography and layout

```sh
build_tools/plots/snippet2svg/snippet2svg input.py \
  --font-family '"IBM Plex Mono", monospace' \
  --font-size 18 \
  --line-height 1.4 \
  --tab-width 4 \
  -o output.svg
```

Tabs are expanded only in the SVG. Spaces, blank lines, indentation, and source
line breaks are otherwise preserved. `--transparent` omits the default white
background. `--padding-x` and `--padding-y` adjust the tight crop.

The SVG names fonts but does not include font files. The default fallback list
is `DejaVu Sans Mono`, `Liberation Mono`, `Consolas`, and `monospace`. These are
non-ligature monospaced faces, so the SVG preserves literal operators such as
`->` and keeps the renderer's measured token positions aligned.

Ordinary code uses medium weight (`500`) so it remains legible after a snippet
is reduced inside a composed paper figure. Accented operations and custom types
use `700`; the hierarchy therefore remains visible without making the entire
snippet bold. Both weights are centralized in `theme.mjs`.

## Paper theme and dialect emphasis

All palette values, typography defaults, and the default accent dialect list are
in [`theme.mjs`](theme.mjs). Change a value there and regenerate the SVGs.

The LLVM grammar provides the base token scopes. A small MLIR-aware pass then
emphasizes, without changing the source:

- SSA values in dark indigo;
- operations in configured dialects in navy;
- custom types in configured dialects in sage; and
- builtin MLIR types in slate.

`merlin_iface` is accented by default. Add another dialect for one invocation:

```sh
build_tools/plots/snippet2svg/mlir2svg input.mlir \
  --accent-dialect gemmini \
  -o output.svg
```

The override checks MLIR grammar scopes and operation position. It does not
mistake `merlin_iface.target = "gemmini"` for an operation.

To inspect how Shiki and the semantic pass classified a source file, add
`--dump-tokens`. The command still writes the SVG and prints selected token
contents, TextMate scopes, colors, and semantic categories as JSON.

## PowerPoint workflow

Use **Insert -> Pictures -> This Device**, then select the generated `.svg`.
Keep the SVG as one generated object. Add arrows, braces, labels, circles,
highlight boxes, and the rest of the figure composition in PowerPoint. Do not
recolor code tokens there. If source changes, rerun `snippet2svg` and replace the
SVG object.

Generated examples and paper assets belong under
`out/artifacts/paper-figures/snippets/`. Do not commit `node_modules` or copy it
into Overleaf.

The current generated batch contains:

- `capsule-derivation/individual/`: standalone capsule-derivation snippets;
- `capsule-derivation/current-pipeline/`: PT2, profile synthesis, SMT, EqSat,
  phase-policy, and capsule snippets;
- `capsule-derivation/connected-gemv/`: one connected application-to-capsule
  example;
- `phase1/b3-toolchain/`: a real B3 capsule, generated target IR, command
  buffer, LLVM-dialect lowering, decoded trace, L2/L3 result, RTL facts, CCA
  route, compiler repair, and generated ISA helper; and
- `phase1/tooling/`: current tier, coverage, certification-budget,
  RTL-to-check, and RTL-to-ISA implementation excerpts.

The SVG files under `out/` are ignored. Batch-rendered directories also contain
a deterministic `manifest.json` with source and SVG hashes.

## Tests

```sh
npm test --prefix build_tools/plots/snippet2svg
```

The tests cover source integrity, full-file tokenization before cropping,
original line numbers, semantic MLIR colors, all Phase 1/capsule snippet
languages, XML escaping, deterministic bytes, source mtime/hash preservation,
layout bounds, and the absence of browser-only or raster SVG features.

## Grammar provenance

See [`UPSTREAM.md`](UPSTREAM.md). Rendering uses the vendored `grammar.json` and
never downloads a grammar at runtime. The renderer checks the grammar digest at
startup and stops if the file no longer matches the reviewed LLVM revision.
`npm run update:grammar` is a separate, hash-checked maintenance command.

## Known grammar limitations

LLVM's grammar is syntax highlighting, not an MLIR parser. It does not validate
operations, attributes, regions, types, or symbol references. Some punctuation
has no specific TextMate scope, and generic operation syntax in quotes is
classified as a string. The renderer leaves unmatched syntax in charcoal and
never hides or marks it as invalid. Compiler and capsule checks remain the
authorities for correctness.
