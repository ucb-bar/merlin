---
name: inspect-ir
description: Dump and read your compiler's IR at every stage for one case, check it against the interface and target-dialect contracts, and bisect a miscompile to the first stage that goes wrong. Open it when a case fails numerically or structurally and you do not yet know which stage is at fault, when a variant changes output and you want the first differing stage, or when you need the interface, linalg, dense, schedule or target IR printed. Covers how the grader invokes the opt tool named in submission/manifest.yaml, a script that runs every declared command and keeps each stage, the library printers and parsers, and the contract check.
---

# Inspect IR stage by stage

`SKILL` below is this skill's directory. Run commands from the workspace root. Stage detail,
library entry points and contract fields are in `reference/stages.md`.

## 1. Know how your tool is run

`submission/manifest.yaml` names the tool (`entrypoints.tool`, or `build.tool_output`) and each
command as `commands.NAME.argv`. The grader:
- substitutes only `{tool}`, `{input_mlir}` and `{output_json}`, as absolute paths;
- runs every command from the package root;
- hands every stage the same interface file.

It runs `parse`, then `lower_interface_to_target` (its stdout is the target IR), then the
command-buffer and artifact commands. Your tool has only the flags you gave it. Add your own
print-after or verify-each flags if you want them; the grader never passes them.

## 2. Dump every stage for one case

```
python SKILL/scripts/dump_stages.py run CASE                          # CASE: name, case dir or .mlir
python SKILL/scripts/dump_stages.py run CASE --submission variants/NAME --out $TMPDIR/ir-stages/CASE.v
python SKILL/scripts/dump_stages.py run CASE --append=--your-print-flag
```
The script writes to `$TMPDIR/ir-stages/CASE` by default:
- `input.interface.mlir`, `lowered.target.mlir`, `command_buffer.json` and `lowered.llvm.mlir`;
- each stage's `STAGE.stderr`;
- `stages.json`, with argv, exit code, time and digests.

It stops at the first failing stage, where the grader would stop.

## 3. Read each stage

- **Interface input.** The grammar is in `interface_grammar.md` and the contract in `interface_dialect_contract.yaml`. Print the library's reading of the input and compare its `commands` with your `command_buffer.json`. A difference is a front-end bug.
  ```
  python -c "import json,sys; from merlin.targetgen.contract.interface_emit import parse_interface_mlir; print(json.dumps(parse_interface_mlir(open(sys.argv[1]).read()), indent=1))" $TMPDIR/ir-stages/CASE/input.interface.mlir
  ```
- **Linalg-on-tensors sources:** `merlin.targetgen.contract.linalg_iface.parse_linalg_mlir` returns ops with `kind`, `extents`, `reduction_dims` and `prov`.
- **Your dense, plan and schedule IR:** print them from your code with `dense_schedule_text`, `Kernel.text()` or `compact_text` (see the reference).
- **Target IR:** recorded as evidence, not semantically gated. It must be non-empty, parseable MLIR.
- **Lowered artifact:** this is what gets compiled. See the `inspect-kernel` skill.

## 4. Check the contracts

```
python SKILL/scripts/dump_stages.py check $TMPDIR/ir-stages/CASE
```
The check covers four things:
- interface attributes and op names against the contract;
- whether the target IR is non-empty and parses (xDSL, your own ops unregistered);
- whether the command buffer is JSON with a `reason` on any `declined` entry;
- whether the lowered module defines a function.

A linalg source that carries `params.global_program_plan` also has a plan preflight:
`python -m merlin.targetgen.oot_starterkit.plan validate --source IN.mlir --command-buffer CB.json --lowered-mlir L.mlir`.

## 5. Bisect a miscompile

1. **Compare a good dump with a bad one.** The good one can come from a passing variant or from an older copy of your tree under `variants/`. Run `python SKILL/scripts/dump_stages.py compare GOOD BAD`; it names the first stage that differs.
2. **If `command_buffer.json` differs first,** the fault is in planning or lowering. Compare your buffer with `parse_interface_mlir` of the input, and run `isa_tools.py debug` on both buffers (see the `read-command-buffer` skill).
3. **If only `lowered.llvm.mlir` differs,** the fault is in emission. Run `loop_map.py` from `inspect-kernel` and `python isa_tools.py disasm FILE`.
4. **Inside your compiler, keep halving.** Print after the middle pass in both runs and keep the half that differs. Verify after each pass; the first one that fails verification is the culprit.

Before importing a library module, check "Denied inputs" in `ALLOWED_MERLIN_TOOLS.md`. Some
modules are present in the workspace but are not granted to this arm.
