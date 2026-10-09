# Opt-in public Verilog engine replay

This recipe rebuilds a selected public Icarus Verilog compiler and its `vvp`
executor in a caller-owned build directory. It can execute original Verilog
initialization, including parameterized system functions that another selected
frontend refuses. A successful build or small RTL control does not qualify a
target platform, a compiler candidate, an execution service or physical timing.

## Exact public source selection

- Repository: [steveicarus/iverilog](https://github.com/steveicarus/iverilog).
- Public release: `v12_0`, version `12.0 (stable)`.
- Annotated tag object: `d61ab44c8a62d10dc80c3ae219387de04b7217c7`.
- Commit referenced by that tag: `4fd5291632232fbe1ba49b2c26bb6b2bf1c6c9cf`.
- Source archive:
  `https://codeload.github.com/steveicarus/iverilog/tar.gz/4fd5291632232fbe1ba49b2c26bb6b2bf1c6c9cf`.
- Observed archive size: `2997829` bytes.
- Archive SHA256:
  `690f02e6fd0e0c05e4768bcb65a2c7f1b64d20289344d1ad3ccfb69bae8bd55b`.

Resolve the annotated tag to its commit; the tag object is not a source commit.
Select the exact archive bytes and refuse another revision or checksum. These
pins identify this replay, rather than a moving release name. The public
[build instructions](https://github.com/steveicarus/iverilog/blob/4fd5291632232fbe1ba49b2c26bb6b2bf1c6c9cf/README.md#buildinginstalling-icarus-verilog-from-source)
describe the native prerequisites and source build.

## Owned build and actual invocation records

1. Create a fresh directory under `merlin.common.paths.build_dir()`. Save the
   replay controller as a real file in that directory before invoking it.
   A callable defined on standard input has no reusable source-file pin.
2. Fetch the public tag object and exact commit archive with an explicitly
   selected downloader. Disable its implicit user configuration and use an
   explicitly selected process environment. Retain the actual download command,
   source products and their checksums. Do not reuse an existing engine binary.
3. Extract only regular source files and directories into a fresh immutable
   snapshot. Reject absolute paths, parent traversal, unexpected archive roots,
   links and special files. Record the complete relative file roster and hashes.
   Copy that snapshot into a separate mutable build directory. Configuration and
   generated parser files must not overwrite the original source snapshot.
4. Select and pin the actual shell, C/C++ compilers, make, autoconf, bison, flex
   and gperf used by the build. Keep their paths and hashes with the source and
   controller inventory. Select the effective environment explicitly, including
   `PATH`, locale and `CC`/`CXX`; retain its recorder identity at every process
   boundary. Record further discovered tools, headers, libraries and generated
   dependencies separately. Top-level tool pins do not prove a transitive
   toolchain closure.
5. In the mutable source directory, run the public build steps with actual
   recorded commands and bounded deadlines:

   ```text
   <selected-shell> autoconf.sh
   <selected-shell> configure --prefix=<fresh-owned-install-directory>
   <selected-make> -j<explicit-bounded-parallelism>
   <selected-make> install
   ```

   The prefix must remain inside the caller's owned build directory. Do not
   install system-wide, alter a shared toolchain or modify public source.
   Preserve configuration, parser generation and compiler warnings or failures.
6. Recheck the immutable source roster, every installed product and actual
   successful invocation record. Invoke the newly installed `iverilog -V` and
   `vvp -V` with recorded commands. Retain the reported version alongside the
   exact binary hashes; version text alone is insufficient.

Use `merlin.common.invocation_record.run` for each actual command, with its
explicit `env`, `cwd`, input, output and controller dependencies.
Use `observe_call` only for actual source-owned materialization or checking
functions. Keep failed or interrupted records separate from successful records;
`invocation_record.verify` deliberately refuses a failed command. Reopen their
unchanged tool, input, output and console pins for an audit without relabeling
them as successful executions. Save all build products and receipts under the
owned `out/` directory; this recipe vendors no upstream source or binary.

## Original RTL execution controls

Select the original RTL, complete instance parameter bindings, top module,
stimulus, program arguments and original full output checks independently of
this tool recipe. Those target bindings remain with their OOT owner. Compile
the unchanged source using the selected installed `iverilog`, an explicit
language mode and top module. Execute its actual generated product with the
selected installed `vvp` and the exact recorded argument vector.

Pin the complete installed engine inventory and generated product at both
boundaries. Retain the selected native loader's actual resolved dependencies and
compiler-declared VPI modules where needed; a startup dependency list does not
prove the absence of later dynamic loads or unrestricted filesystem access.
Engine products, selected public support and dependency pins still need the
ordinary independent execution and containment owners.

Preserve every declared output row, width, unknown bit, original process outcome
and completion marker. Include no-argument and valid-override controls, omitted
or nonmatching arguments, declared-width boundaries and malformed arguments.
The tested public engine can return unknown bits for malformed numeric input;
do not substitute a value or treat that observation as a numerical pass.
Reject unknown bits before binding them into a numeric native state.

Include a bounded timeout or requested-stop negative under the original RTL
conditions. Keep its actual stop request distinct from ordinary completion,
process exit and complete numerical agreement. Preserve partial output and a
failed or incomplete original numerical denominator when a stop truncates the
stream. A valid control that completes all original outputs must pass their
original numerical gate separately. Retain unsupported-frontend and isolated
macro-selection diagnostics; a working native engine does not retrospectively
qualify the refused or altered route.

## Scope

Actual independent replay built the exact public source and executed unchanged
parameterized RTL, including complete defined-output controls and a malformed
numeric-input counterexample. Separate native execution retained complete
original numerical outputs and an original failure-stop control. These are
diagnostic observations only; their source and result files remain outside this
recipe.

No tool discovery, download, build, patch or execution occurs on import. Nothing
here issues an author input, tool grant, compiler seed, backend default, runtime
role or performance estimate. Physical cycles, clock/reset/loading contracts,
whole-platform equivalence, trusted observer integrity and target runtime
qualification remain with their independently evaluated owners.
