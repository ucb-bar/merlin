# WP-2 handoff: IR ingress (exporter, decoder, printer, round trip)

Status: **nothing implemented or committed yet** — work stopped during the survey/design step. This
note records what was established so the next owner starts from it.

## Done

- Interface change request approved and landed (75455b83f): `Module.attrs`/`Module.props` and the
  schema's optional `module.attrs`/`module.props`. Needed because 721/734 corpus files carry
  `module attributes {...}` (`prov.*`, `merlin_iface.*`).
- Environment verified: `lake build` passes with `out/build/lean/packages` symlinked to
  `/home/jeremy/Projects/merlin/out/build/lean/packages`; `.venv` (py3.14) installs cleanly.

## Facts established

- MLIR: `/home/jeremy/Projects/llvm-project/build`, static libs, Release+assertions, RTTI/EH off,
  `LLVM_REVISION` in `include/llvm/Support/VCSRevision.h` = HEAD `a47bddccec30…` — use it for
  `producer.mlir_revision`. `mlir-opt --version` prints no revision.
- The repo has no `.env`; `merlin.verify.tools.find_mlir_tool` looks in `third_party/llvm-{build,install}`
  (both absent). Have CMake bake `${LLVM_TOOLS_BINARY_DIR}/mlir-opt` into the exporter and report it,
  so the round trip uses the mlir-opt of the same MLIR build.
- Generic printer (`AsmPrinter.cpp` `printGenericOp`): props = `op->getPropertiesAsAttribute()`
  (null when no props on registered ops; any attr, possibly an empty dict, on unregistered ops);
  attrs = `getDiscardableAttrs()` when properties storage exists, else `getAttrs()`.
- Corpus: 734 `.mlir` files, 26.8 MB (capsules 720, tests/data 14, examples 6); the largest is
  `merlin/contract/capsules/model/SY_model_smolvla/capsule.interface.mlir` (10.9 MB). No
  `dense_resource` and no `dialect_resources` occur.

## Design decisions to keep

1. Top level: unwrap the top-level module body into `module.ops`, emit its discardable attrs/props
   into `module.attrs`/`props` (always present, `{}` when empty), and keep nested `builtin.module`
   as an ordinary op. The printer always wraps in `"builtin.module"() <{props}> ({^bb0: ...}) {attrs}`.
2. The printer always names values `%v<id>` and labels every block `^bb<i>`. It quotes every attr-dict
   key and symbol (`@"..."`), prints ints as `<dec> : <ty>`, floats as hex bits `0x.. : <ty>`, and
   affine expressions fully parenthesised. Results are `-> (t, ...)`. Operand types come from an
   id→type table (an unknown id is an error, never a guess).
3. i1 dense: the exporter normalises to one byte per element. MLIR packs i1 into bits, so the printer
   must print i1 as a nested literal (`dense<[[true,false],...]>`), not as hex. Check how this LLVM
   parses i1 hex before relying on it.
4. Opaque: unranked/dynamic-only types, float dense arrays, string dense, strided layouts,
   `dense_resource` without a blob, non-UTF-8 strings, and unknown attrs are emitted as `opaque`
   with their `Attribute::print` text. An unregistered op whose properties are not a dict is an
   exporter error.
5. Decoder: avoid `partial def` by using fuel = input byte length (structural on `Nat`; depth can
   never exceed it). Check id uniqueness and that every operand is defined. Decode hex by
   index-recursion on `toUTF8`.
6. Round-trip blind spot: MLIR accepts inherent attrs in the discardable dict, so a props/attrs swap
   still canonicalises identically. Register all dialects and extensions in the exporter, exactly as
   mlir-opt does, so its split matches.
7. Round trip result statuses: `identical` / `different` (with diff excerpt) / `failed` (export or
   parse error) / `unavailable` (tool missing). Only `identical` is success.

## Remaining (all of WP-2)

C++ exporter (`merlin/lean/exporter/`), `Core/Json.lean`, `Core/Print.lean`, Main.lean
subcommands, `src/merlin/verify/lean/{export,xdsl_export}.py`, `merlin/tests/ir/test_lean_export.py`,
and the corpus round-trip report.

## Next steps

1. `cmake -G Ninja -S merlin/lean/exporter -B out/build/lean-exporter
   -DMLIR_DIR=/home/jeremy/Projects/llvm-project/build/lib/cmake/mlir
   -DLLVM_DIR=/home/jeremy/Projects/llvm-project/build/lib/cmake/llvm` (link `MLIR_DIALECT_LIBS`
   and `MLIR_EXTENSION_LIBS`; use `registerAllDialects` and `registerAllExtensions`; stream with
   `llvm::json::OStream`; hash with `llvm::SHA256`).
2. Write Print.lean, then Json.lean, with `#guard` goldens in `MerlinVerify/Tests/Ingress*.lean`.
3. Write `export.py` and run the 734-file corpus. Start with the largest file to measure the cost of
   `Lean.Json` and `List UInt8`.
