"""Validate and generate target-independent glue for multi-program inference sessions.

A paper session may need several fixed-shape compiled entry points: causal-LM prefill followed by
recurrent decode, or SmolVLA prefix encode followed by recurrent flow matching and action decode.
This module keeps that orchestration declarative.  It validates every cross-program tensor edge
against the captured MLIR ABIs and emits C glue; it does not choose schedules or inspect model names.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common.mlir_query import forward_signature
from ..common.yaml import load_yaml
from . import c_runtime
from .model_runner import parse_forward_signature


def _identifier(value: object, where: str) -> str:
    text = str(value or "")
    if not text.isidentifier() or not text.isascii():
        raise ValueError(f"{where} must be an ASCII identifier, got {text!r}")
    return text


def _mapping(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    return value


@dataclass(frozen=True)
class Program:
    name: str
    bundle: Path
    steps: int
    inputs: tuple[tuple[list[int], str], ...]
    outputs: tuple[tuple[list[int], str], ...]


@dataclass(frozen=True)
class Binding:
    name: str
    source_program: str
    source_output: int
    target_program: str
    target_input: int


@dataclass(frozen=True)
class MultiProgramSession:
    root: Path
    raw: dict[str, Any]
    programs: tuple[Program, ...]
    bindings: tuple[Binding, ...]
    quality_program: str
    quality_output: int

    @property
    def program_names(self) -> tuple[str, ...]:
        return tuple(program.name for program in self.programs)


def _safe_bundle(root: Path, value: object, where: str) -> Path:
    relative = Path(str(value or ""))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"{where} must be a relative path contained by the session bundle")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{where} escapes the session bundle") from exc
    return resolved


def _child_state_names(program: Program) -> set[str]:
    path = program.bundle / "session_contract.yaml"
    if not path.is_file():
        if program.steps != 1:
            raise ValueError(
                f"program {program.name}: {program.steps} steps require a child session contract")
        return set()
    child = _mapping(load_yaml(path), f"program {program.name} child session")
    if int(child.get("version", 0)) != 1:
        raise ValueError(f"program {program.name}: child session contract must be version 1")
    if list(child.get("stages", ()) or ()) != [program.name]:
        raise ValueError(
            f"program {program.name}: child session stages must be exactly [{program.name!r}]")
    if int(child.get("steps", 0) or 0) != program.steps:
        raise ValueError(f"program {program.name}: child step count differs from root schedule")
    return {str(item.get("name")) for item in child.get("states", ()) or ()
            if isinstance(item, dict)}


def load(root: str | Path) -> MultiProgramSession:
    """Load a version-2 session and fail before codegen on any path, stage, or ABI ambiguity."""
    root = Path(root).resolve()
    raw = _mapping(load_yaml(root / "session_contract.yaml"), "multi-program session")
    if int(raw.get("version", 0)) != 2:
        raise ValueError("multi-program session contract must have version 2")
    stages = tuple(str(value) for value in raw.get("stages", ()) or ())
    program_rows = raw.get("programs", ()) or ()
    if not isinstance(program_rows, list) or not program_rows:
        raise ValueError("multi-program session requires a non-empty programs list")
    programs: list[Program] = []
    for index, item in enumerate(program_rows):
        item = _mapping(item, f"programs[{index}]")
        name = _identifier(item.get("name"), f"programs[{index}].name")
        bundle = _safe_bundle(root, item.get("bundle"), f"programs[{index}].bundle")
        steps = int(item.get("steps", 0))
        if steps < 1:
            raise ValueError(f"program {name}: steps must be positive")
        required = ("model.mlir", "inputs.npz", "weights.safetensors",
                    "weights.safetensors.manifest.json")
        missing = [str(bundle / filename) for filename in required
                   if not (bundle / filename).is_file()]
        if missing:
            raise ValueError(f"program {name}: capture artifacts are absent: {missing}")
        inputs = tuple(parse_forward_signature(bundle / "model.mlir"))
        _, output_values = forward_signature(bundle / "model.mlir")
        outputs = tuple(output_values)
        if not outputs:
            raise ValueError(f"program {name}: compiled entry point has no outputs")
        programs.append(Program(name, bundle, steps, inputs, outputs))
    names = tuple(program.name for program in programs)
    if len(set(names)) != len(names):
        raise ValueError("multi-program session has duplicate program names")
    if names != stages:
        raise ValueError(f"program order {list(names)} must exactly equal stages {list(stages)}")
    by_name = {program.name: program for program in programs}

    schedule = raw.get("stage_schedule", ()) or ()
    if not isinstance(schedule, list) or len(schedule) != len(programs):
        raise ValueError("stage_schedule must contain exactly one row per program")
    for program, row in zip(programs, schedule, strict=True):
        row = _mapping(row, f"stage_schedule[{program.name}]")
        if row.get("name") != program.name or int(row.get("steps", 0)) != program.steps:
            raise ValueError(f"stage_schedule row differs from program {program.name}")
        if row.get("timed") is not True or not str(row.get("execution", "")).startswith("compiled"):
            raise ValueError(f"program {program.name}: every primary stage must be timed compiled code")

    bindings: list[Binding] = []
    targets: set[tuple[str, int]] = set()
    order = {name: index for index, name in enumerate(names)}
    for index, item in enumerate(raw.get("bindings", ()) or ()):
        item = _mapping(item, f"bindings[{index}]")
        source = _mapping(item.get("from"), f"bindings[{index}].from")
        target = _mapping(item.get("to"), f"bindings[{index}].to")
        source_name = str(source.get("program", ""))
        target_name = str(target.get("program", ""))
        if source_name not in by_name or target_name not in by_name:
            raise ValueError(f"bindings[{index}] references an unknown program")
        if order[source_name] >= order[target_name]:
            raise ValueError(f"bindings[{index}] must flow forward in stage order")
        source_output, target_input = int(source.get("output_index", -1)), int(
            target.get("input_arg", -1))
        source_program, target_program = by_name[source_name], by_name[target_name]
        if source_output < 0 or source_output >= len(source_program.outputs):
            raise ValueError(f"bindings[{index}] source output is outside the compiled ABI")
        if target_input < 0 or target_input >= len(target_program.inputs):
            raise ValueError(f"bindings[{index}] target input is outside the compiled ABI")
        target_key = (target_name, target_input)
        if target_key in targets:
            raise ValueError(f"bindings[{index}] duplicates target {target_key}")
        targets.add(target_key)
        if source_program.outputs[source_output] != target_program.inputs[target_input]:
            raise ValueError(
                f"bindings[{index}] ABI mismatch: {source_program.outputs[source_output]} -> "
                f"{target_program.inputs[target_input]}")
        bindings.append(Binding(str(item.get("name", f"binding{index}")), source_name,
                                source_output, target_name, target_input))

    # Cross-program bindings also carry immutable context (e.g. prefix padding masks) and final
    # values into a post-processing stage. Only child recurrent contracts define carried state;
    # naming every edge a state would make the semantic state set depend on harmless wiring.
    routed_states: set[str] = set()
    for program in programs:
        routed_states.update(_child_state_names(program))
    declared_states = {str(item.get("name")) for item in raw.get("states", ()) or ()
                       if isinstance(item, dict)}
    if declared_states != routed_states:
        raise ValueError(
            f"root carried states differ from executable routes: declared={sorted(declared_states)} "
            f"routed={sorted(routed_states)}")
    quality = _mapping(raw.get("quality"), "quality")
    quality_program = str(quality.get("program", ""))
    if quality_program not in by_name:
        raise ValueError("quality.program must name a compiled program")
    quality_output = 0
    quality_child_path = by_name[quality_program].bundle / "session_contract.yaml"
    if quality_child_path.is_file():
        quality_child = _mapping(load_yaml(quality_child_path), "quality child session")
        quality_output = int(_mapping(quality_child.get("quality"), "quality child output").get(
            "output_index", 0))
    if quality_output < 0 or quality_output >= len(by_name[quality_program].outputs):
        raise ValueError("quality output is outside the compiled program ABI")
    return MultiProgramSession(
        root, raw, tuple(programs), tuple(bindings), quality_program, quality_output)


# MLIR bare-identifier continuation characters, per the langref (`[a-zA-Z0-9_$.]`).  A word boundary
# is the wrong test here: it treats `$` and `.` as separators, so `@forward.bwd` would be renamed as
# though it were the `@forward` symbol.
_SYMBOL_TAIL = "$."


def rename_forward(mlir_text: str, entrypoint: str) -> str:
    """Rename exactly the public `forward` symbol before lowering a stage into a shared binary."""
    entrypoint = _identifier(entrypoint, "entrypoint")
    head, *rest = mlir_text.split("@forward")
    out, count = [head], 0
    for tail in rest:
        following = tail[:1]
        if following.isalnum() or following == "_" or following in _SYMBOL_TAIL:
            out.append("@forward")  # a longer symbol that merely starts with `forward`
        else:
            out.append(f"@{entrypoint}")
            count += 1
        out.append(tail)
    if count < 1:
        raise ValueError("stage MLIR has no @forward symbol to rename")
    return "".join(out)


def _adapter_source(index: int, program: Program, invoke_name: str) -> str:
    prefix = f"merlin_stage_{index}"
    return f'''/* Generated multi-program stage adapter. */
#include <stddef.h>
#include "merlin_model.h"
#include "model_gen.h"
#include "model_io.h"
extern void {invoke_name}(void **);
static merlin_descriptor_t DESCS[MERLIN_N_ARGS];
static size_t arg_bytes(const merlin_arg_t *arg) {{
  size_t n = (size_t)arg->elem_size;
  for (int i = 0; i < arg->rank; i++) n *= (size_t)arg->dims[i];
  return n;
}}
void {prefix}_reset(void) {{ merlin_reset_session(); }}
int {prefix}_run(const void *weights, long step, int validate) {{
  merlin_prepare_step(step);
  merlin_run_multi_with(MERLIN_ARGS, MERLIN_N_ARGS, weights, MERLIN_INPUT_PTR,
                        MERLIN_OUTPUT_PTR, DESCS, {invoke_name});
#if MERLIN_N_STATE_PAIRS > 0
  if (merlin_commit_state(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_INPUT_PTR, MERLIN_OUTPUT_PTR,
                          MERLIN_N_STATE_PAIRS, MERLIN_STATE_INPUT_ARGS,
                          MERLIN_STATE_OUTPUT_INDICES) != 0) return -1;
#endif
  if (validate) merlin_validate_step(step);
  return 0;
}}
void *{prefix}_input(int arg) {{
  return arg >= 0 && arg < MERLIN_N_ARGS && MERLIN_ARGS[arg].kind == MERLIN_INPUT
      ? MERLIN_INPUT_PTR[arg] : 0;
}}
size_t {prefix}_input_bytes(int arg) {{
  return arg >= 0 && arg < MERLIN_N_ARGS && MERLIN_ARGS[arg].kind == MERLIN_INPUT
      ? arg_bytes(&MERLIN_ARGS[arg]) : 0;
}}
static int output_arg(int wanted) {{
  int seen = 0;
  for (int arg = 0; arg < MERLIN_N_ARGS; arg++)
    if (MERLIN_ARGS[arg].kind == MERLIN_OUTPUT && seen++ == wanted) return arg;
  return -1;
}}
void *{prefix}_output(int output) {{
  return output >= 0 && output < MERLIN_N_OUTPUTS ? MERLIN_OUTPUT_PTR[output] : 0;
}}
size_t {prefix}_output_bytes(int output) {{
  int arg = output_arg(output); return arg >= 0 ? arg_bytes(&MERLIN_ARGS[arg]) : 0;
}}
long {prefix}_quality_steps(void) {{ return merlin_quality_steps(); }}
long {prefix}_quality_min_cos_ppm(void) {{ return merlin_quality_min_cos_ppm(); }}
long {prefix}_quality_max_rel_ppm(void) {{ return merlin_quality_max_rel_ppm(); }}
long {prefix}_quality_top1(void) {{ return merlin_quality_top1(); }}
long {prefix}_correctness_steps(void) {{ return merlin_correctness_steps(); }}
long {prefix}_correctness_min_cos_ppm(void) {{ return merlin_correctness_min_cos_ppm(); }}
long {prefix}_correctness_max_rel_ppm(void) {{ return merlin_correctness_max_rel_ppm(); }}
long {prefix}_correctness_top1(void) {{ return merlin_correctness_top1(); }}
size_t {prefix}_output_elems(int output) {{
  int arg = output_arg(output); if (arg < 0) return 0;
  size_t n = 1; for (int i = 0; i < MERLIN_ARGS[arg].rank; i++) n *= MERLIN_ARGS[arg].dims[i];
  return n;
}}
long {prefix}_output_lastdim(int output) {{
  int arg = output_arg(output); return arg >= 0 && MERLIN_ARGS[arg].rank > 0
      ? MERLIN_ARGS[arg].dims[MERLIN_ARGS[arg].rank - 1] : 1;
}}
'''


def _session_sources(session: MultiProgramSession) -> tuple[str, str]:
    declarations: list[str] = []
    for index, _program in enumerate(session.programs):
        prefix = f"merlin_stage_{index}"
        declarations += [
            f"void {prefix}_reset(void);",
            f"int {prefix}_run(const void *, long, int);",
            f"void *{prefix}_input(int);", f"size_t {prefix}_input_bytes(int);",
            f"void *{prefix}_output(int);", f"size_t {prefix}_output_bytes(int);",
            f"long {prefix}_quality_steps(void);",
            f"long {prefix}_quality_min_cos_ppm(void);",
            f"long {prefix}_quality_max_rel_ppm(void);",
            f"long {prefix}_quality_top1(void);",
            f"long {prefix}_correctness_steps(void);",
            f"long {prefix}_correctness_min_cos_ppm(void);",
            f"long {prefix}_correctness_max_rel_ppm(void);",
            f"long {prefix}_correctness_top1(void);",
            f"size_t {prefix}_output_elems(int);", f"long {prefix}_output_lastdim(int);",
        ]
    names = ",".join(json.dumps(program.name) for program in session.programs)
    steps = ",".join(str(program.steps) for program in session.programs)
    reset_lines = "\n".join(f"  merlin_stage_{i}_reset();"
                              for i in range(len(session.programs)))
    cases: list[str] = []
    for target_index, target in enumerate(session.programs):
        copies: list[str] = []
        for binding in session.bindings:
            if binding.target_program != target.name:
                continue
            source_index = session.program_names.index(binding.source_program)
            copies += [
                f"      size_t source_bytes = merlin_stage_{source_index}_output_bytes("
                f"{binding.source_output});",
                f"      size_t target_bytes = merlin_stage_{target_index}_input_bytes("
                f"{binding.target_input});",
                "      if (!source_bytes || source_bytes != target_bytes) return -1;",
                f"      memcpy(merlin_stage_{target_index}_input({binding.target_input}), "
                f"merlin_stage_{source_index}_output({binding.source_output}), source_bytes);",
            ]
        body = "\n".join(copies) if copies else "      (void)0;"
        cases.append(f"    case {target_index}: {{\n{body}\n      return 0;\n    }}")
    run_cases = "\n".join(
        f"    case {index}: return merlin_stage_{index}_run(weights, step, validate);"
        for index in range(len(session.programs)))
    quality_index = session.program_names.index(session.quality_program)
    source = f'''/* Generated multi-program session scheduler. */
#include <stddef.h>
#include <string.h>
#include "merlin_session.h"
{chr(10).join(declarations)}
static const char *NAMES[MERLIN_SESSION_N_PROGRAMS] = {{{names}}};
static const long STEPS[MERLIN_SESSION_N_PROGRAMS] = {{{steps}}};
const char *merlin_session_program_name(int program) {{
  return program >= 0 && program < MERLIN_SESSION_N_PROGRAMS ? NAMES[program] : 0;
}}
long merlin_session_program_steps(int program) {{
  return program >= 0 && program < MERLIN_SESSION_N_PROGRAMS ? STEPS[program] : 0;
}}
void merlin_session_reset(void) {{
{reset_lines}
}}
int merlin_session_prepare_program(int program) {{
  switch (program) {{
{chr(10).join(cases)}
    default: return -1;
  }}
}}
int merlin_session_run_step(int program, const void *weights, long step, int validate) {{
  switch (program) {{
{run_cases}
    default: return -1;
  }}
}}
long merlin_session_quality_steps(void) {{ return merlin_stage_{quality_index}_quality_steps(); }}
long merlin_session_quality_min_cos_ppm(void) {{
  return merlin_stage_{quality_index}_quality_min_cos_ppm();
}}
long merlin_session_quality_max_rel_ppm(void) {{
  return merlin_stage_{quality_index}_quality_max_rel_ppm();
}}
long merlin_session_quality_top1(void) {{ return merlin_stage_{quality_index}_quality_top1(); }}
long merlin_session_correctness_steps(void) {{
  return merlin_stage_{quality_index}_correctness_steps();
}}
long merlin_session_correctness_min_cos_ppm(void) {{
  return merlin_stage_{quality_index}_correctness_min_cos_ppm();
}}
long merlin_session_correctness_max_rel_ppm(void) {{
  return merlin_stage_{quality_index}_correctness_max_rel_ppm();
}}
long merlin_session_correctness_top1(void) {{
  return merlin_stage_{quality_index}_correctness_top1();
}}
void *merlin_session_quality_output(void) {{
  return merlin_stage_{quality_index}_output({session.quality_output});
}}
size_t merlin_session_quality_output_elems(void) {{
  return merlin_stage_{quality_index}_output_elems({session.quality_output});
}}
long merlin_session_quality_output_lastdim(void) {{
  return merlin_stage_{quality_index}_output_lastdim({session.quality_output});
}}
'''
    header = f'''/* Generated multi-program session API. */
#ifndef MERLIN_SESSION_H
#define MERLIN_SESSION_H
#define MERLIN_SESSION_N_PROGRAMS {len(session.programs)}
const char *merlin_session_program_name(int program);
long merlin_session_program_steps(int program);
void merlin_session_reset(void);
int merlin_session_prepare_program(int program);
int merlin_session_run_step(int program, const void *weights, long step, int validate);
long merlin_session_quality_steps(void);
long merlin_session_quality_min_cos_ppm(void);
long merlin_session_quality_max_rel_ppm(void);
long merlin_session_quality_top1(void);
long merlin_session_correctness_steps(void);
long merlin_session_correctness_min_cos_ppm(void);
long merlin_session_correctness_max_rel_ppm(void);
long merlin_session_correctness_top1(void);
void *merlin_session_quality_output(void);
size_t merlin_session_quality_output_elems(void);
long merlin_session_quality_output_lastdim(void);
#endif
'''
    return header, source


def generate(root: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Generate stage-local runtime artifacts plus a cross-stage scheduler."""
    session = load(root)
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, program in enumerate(session.programs):
        entrypoint = f"merlin_stage_{index}_{program.name}"
        invoke = "merlin_invoke" if index == 0 else f"merlin_stage_{index}_invoke"
        stage_out = out / f"stage_{index}_{program.name}"
        info = c_runtime.generate(program.bundle, stage_out, program.bundle / "inputs.npz",
                                  ciface_name=entrypoint, invoke_name=invoke)
        (stage_out / "stage_adapter.c").write_text(
            _adapter_source(index, program, invoke), encoding="utf-8")
        renamed = rename_forward(
            (program.bundle / "model.mlir").read_text(encoding="utf-8"), entrypoint)
        (stage_out / "model.renamed.mlir").write_text(renamed, encoding="utf-8")
        weights = stage_out / "weights.bin"
        with weights.open("rb") as stream:
            weights_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        records.append({
            "index": index, "name": program.name, "steps": program.steps,
            "bundle": str(program.bundle), "entrypoint": entrypoint, "invoke": invoke,
            "generated": str(stage_out),
            "weights_sha256": weights_sha256,
            **info,
        })
    header, source = _session_sources(session)
    (out / "merlin_session.h").write_text(header, encoding="utf-8")
    (out / "merlin_session.c").write_text(source, encoding="utf-8")
    manifest = {"version": 1, "session_root": str(session.root),
                "programs": records, "bindings": [binding.__dict__ for binding in session.bindings],
                "quality_program": session.quality_program,
                "quality_output": session.quality_output}
    (out / "session_build.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
