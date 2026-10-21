"""Freestanding execution glue for the existing version-two session ABI.

This emits no model implementation or target instructions. Stage lowering,
object namespacing, allocator/stack budgets and final image admission remain
the ordinary builder's responsibility. Every output is offered to an explicit
caller observer immediately after each completed stage call.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import prod
from pathlib import Path

from . import c_runtime, session_bundle


@dataclass(frozen=True, kw_only=True)
class SessionMemoryLimits:
    """Original caller's address interval and accepted-artifact bounds, not facts."""

    weights_base: int
    address_limit: int
    max_weights_bytes: int
    max_static_io_bytes: int
    max_calls: int

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or not 0 < value < 1 << 63:
                raise ValueError(f"{name} requires a positive signed-64-bit integer")
        if self.weights_base % c_runtime.GENERATED_WEIGHT_ALIGNMENT:
            raise ValueError("weights_base violates the generated weight alignment")
        if self.weights_base >= self.address_limit:
            raise ValueError("weights interval is empty")


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _runner(session, rows, limits, timing=False):
    declarations, observations = [], []
    outputs = []
    for index, (program, row) in enumerate(zip(session.programs, rows, strict=True)):
        declarations += [
            f"extern void *merlin_stage_{index}_output(int);",
            f"extern size_t merlin_stage_{index}_output_bytes(int);",
        ]
        calls = []
        for ordinal, (shape, dtype) in enumerate(program.outputs):
            size = prod(shape) * c_runtime.DT_BYTES[dtype]
            outputs.append(dict(program=index, output=ordinal, shape=shape, dtype=dtype, bytes=size))
            calls.append(f"""
      value = merlin_stage_{index}_output({ordinal});
      if (!value || merlin_stage_{index}_output_bytes({ordinal}) != {size}ULL) return -3;
      if (observe({index}, step, {ordinal}, value, {size}ULL, context)) return -4;""")
        setup = f"if (merlin_session_prepare_program({index})) return -2;"
        invoke = (
            f"merlin_session_run_step({index}, (const void *)(uintptr_t)\n"
            f"        (MERLIN_SESSION_WEIGHTS_BASE_ADDR + {row['offset']}ULL), step, 0)"
        )
        call = f"if ({invoke}) return -2;"
        if timing:
            setup = (
                f"started = merlin_session_clock();\n  status = merlin_session_prepare_program({index});\n"
                f"  elapsed = merlin_session_clock() - started;\n  if (status) return -2;\n"
                f"  merlin_session_record_cycles(1, {index}, -1, elapsed);"
            )
            call = (
                f"started = merlin_session_clock();\n    status = {invoke};\n"
                f"    elapsed = merlin_session_clock() - started;\n    if (status) return -2;\n"
                f"    merlin_session_record_cycles(2, {index}, step, elapsed);"
            )
        observations.append(f"""
  {setup}
  for (long step = 0; step < {program.steps}L; ++step) {{
    {call}
    {"".join(calls)}
  }}""")
    header = """/* Generated freestanding session; observer must consume every output. */
#ifndef MERLIN_SESSION_FREESTANDING_H
#define MERLIN_SESSION_FREESTANDING_H
#include <stddef.h>
typedef int (*merlin_session_observer)(int program, long step, int output,
                                      const void *data, size_t bytes, void *context);
int merlin_session_run_freestanding(merlin_session_observer observe, void *context);
#endif
"""
    timing_declarations = (
        "extern uint64_t merlin_session_clock(void);\n"
        "extern void merlin_session_record_cycles(int kind,int program,long step,uint64_t ticks);"
    )
    declarations_text = "\n".join(declarations) + ("\n" + timing_declarations if timing else "")
    reset = "merlin_session_reset();"
    if timing:
        reset = (
            "uint64_t started = merlin_session_clock(), elapsed; int status;\n"
            "  merlin_session_reset();\n  elapsed = merlin_session_clock() - started;\n"
            "  merlin_session_record_cycles(0, -1, -1, elapsed);"
        )
    source = f"""/* No OS services, hidden validation, allocation or timing policy. */
#include <stddef.h>
#include <stdint.h>
#include "merlin_session.h"
#include "merlin_session_freestanding.h"
#ifndef MERLIN_SESSION_WEIGHTS_BASE_ADDR
#define MERLIN_SESSION_WEIGHTS_BASE_ADDR {limits.weights_base}ULL
#endif
_Static_assert(sizeof(uintptr_t) == 8 && sizeof(long) == 8, "session requires LP64");
{declarations_text}
int merlin_session_run_freestanding(merlin_session_observer observe, void *context) {{
  if (!observe) return -1;
  void *value;
  {reset}
  {"".join(observations)}
  return 0;
}}
"""
    return header, source, outputs


def generate(root: str | Path, out_dir: str | Path, *, limits: SessionMemoryLimits, timing=False) -> dict:
    """Generate existing stage glue and one streamed, unshared weight image.

    The default generated addresses are literal absolute addresses suitable for
    the existing ``weights_blob.o`` / ``.weights`` linker path. A native caller
    may define ``MERLIN_SESSION_WEIGHTS_BASE_ADDR`` to its linked blob symbol.
    Byte ceilings cover generated weights and static I/O only: code, descriptors,
    stack, dynamic activations and physical accessibility need separate budgets.
    Existing stage generation runs before these byte checks; they do not bound
    its earlier memory or disk use.
    Optional timing requires original clock/record functions. Reset, program
    preparation (including cross-stage state copies), and step calls are separate
    intervals; recording/observation is outside them, so their sum is not a
    continuous whole-session latency measurement.
    """
    if type(limits) is not SessionMemoryLimits:
        raise ValueError("an original typed SessionMemoryLimits is required")
    limits.__post_init__()
    if type(timing) is not bool:
        raise ValueError("session timing requires an explicit boolean")
    root = Path(root).resolve()
    original = {path: _digest(path) for path in root.rglob("*") if path.is_file()}
    session = session_bundle.load(root)
    by_name = {program.name: program for program in session.programs}
    for binding in session.bindings:
        manifest = json.loads(
            (by_name[binding.target_program].bundle / "weights.safetensors.manifest.json").read_text()
        )
        if manifest.get(str(binding.target_input), {}).get("kind") != "input":
            raise ValueError("session binding target must be a declared runtime input, not immutable weights")
    if sum(program.steps for program in session.programs) > limits.max_calls:
        raise ValueError("session call count exceeds max_calls")
    for program in session.programs:
        if len(program.inputs) + len(program.outputs) > 4096:
            raise ValueError("session stage exceeds the existing runtime argument limit")
        for shape, dtype in (*program.inputs, *program.outputs):
            if (
                dtype not in c_runtime.DT_BYTES
                or len(shape) > 8
                or any(type(dim) is not int or dim <= 0 for dim in shape)
                or prod(shape) * c_runtime.DT_BYTES[dtype] >= 1 << 63
            ):
                raise ValueError("session requires static bounded tensors in the existing runtime ABI")
    out = Path(out_dir).resolve()
    if out.exists():
        raise ValueError("freestanding session output must be fresh")
    # Never place generated data over original source or child artifacts.
    if out == session.root or session.root in out.parents:
        raise ValueError("freestanding output must be outside the source bundle")
    manifest = session_bundle.generate(root, out)
    static_bytes = sum(row["static_io_bytes"] for row in manifest["programs"])
    if static_bytes > limits.max_static_io_bytes:
        raise ValueError("session static I/O exceeds max_static_io_bytes")
    rows, cursor = [], 0
    alignment = c_runtime.GENERATED_WEIGHT_ALIGNMENT
    for row in manifest["programs"]:
        path = Path(row["generated"]) / "weights.bin"
        size = path.stat().st_size
        offset = cursor + (-cursor % alignment)
        cursor = offset + size
        if cursor > limits.max_weights_bytes or limits.weights_base + cursor > limits.address_limit:
            raise ValueError("session weights exceed the declared byte/address interval")
        if size != row["weights_bytes"] or _digest(path) != row["weights_sha256"]:
            raise ValueError("stage weight bytes differ from generated metadata")
        rows.append(dict(path=str(path), offset=offset, bytes=size, sha256=row["weights_sha256"]))
    blob = out / "weights.bin"
    with blob.open("xb") as destination:
        written = 0
        for row in rows:
            destination.write(bytes(row["offset"] - written))
            digest = hashlib.sha256()
            with Path(row["path"]).open("rb") as source:
                while data := source.read(1 << 20):
                    destination.write(data)
                    digest.update(data)
            written = destination.tell()
            if written != row["offset"] + row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise ValueError("stage weight payload changed while streaming")
    header, source, outputs = _runner(session, rows, limits, timing)
    (out / "merlin_session_freestanding.h").write_text(header)
    (out / "merlin_session_freestanding.c").write_text(source)
    current = {path for path in session.root.rglob("*") if path.is_file()}
    if current != original.keys() or any(_digest(path) != digest for path, digest in original.items()):
        raise ValueError("original session source changed during generation")
    record = dict(
        schema="merlin.session_freestanding.v1",
        limits=asdict(limits),
        session=manifest,
        weights=rows,
        weights_bytes=cursor,
        weights_sha256=_digest(blob),
        static_io_bytes=static_bytes,
        outputs=outputs,
        calls=sum(program.steps for program in session.programs),
        sources=[dict(path=str(path), sha256=digest) for path, digest in sorted(original.items())],
        unbudgeted=["code", "stack", "descriptors", "dynamic_activations"],
    )
    if timing:
        record["timing"] = "separate reset/program-setup/step intervals; observers excluded; not whole-session latency"
    (out / "freestanding_session.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record
