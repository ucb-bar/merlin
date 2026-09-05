"""Buildable synthetic tracer for the MRLNSES2 whole-session C ABI.

The tracer is intentionally small but semantically non-trivial: one compiled
prefill call seeds carried state and three recurrent calls update it while
emitting a trajectory.  Generated C embeds only the public session descriptor.
Request values and output references remain outside the object.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paper_session_abi import SessionDescriptor, descriptor_from_contract

ENTRYPOINT = "merlin_paper_session_v1"


@dataclass(frozen=True)
class TracerPackage:
    root: Path
    descriptor: SessionDescriptor
    source: Path
    object: Path
    runner_source: Path
    runner: Path
    receipt: Path


@dataclass(frozen=True)
class ObjectEvidence:
    path: Path
    elf_class: str
    elf_type: str
    machine: str
    symbols: tuple[str, ...]


def synthetic_prefill_decode_descriptor() -> SessionDescriptor:
    """Public two-program contract used by the production-recipe tracer."""
    root = {
        "version": 2,
        "kind": "synthetic_recurrent",
        "paper_ready": True,
        "stages": ["prefill", "decode"],
        "stage_schedule": [
            {"name": "prefill", "steps": 1, "execution": "compiled", "timed": True},
            {"name": "decode", "steps": 3, "execution": "compiled_recurrent", "timed": True},
        ],
        "programs": [
            {"name": "prefill", "bundle": "stages/prefill", "steps": 1},
            {"name": "decode", "bundle": "stages/decode", "steps": 3},
        ],
        "bindings": [
            {
                "name": "state_seed",
                "from": {"program": "prefill", "output_index": 1},
                "to": {"program": "decode", "input_arg": 1},
            }
        ],
        "states": ["state"],
        "streams": [],
        "quality": {"scope": "trajectory", "program": "decode"},
    }

    def child(name: str, steps: int, *, stream: bool, state: bool) -> dict[str, Any]:
        return {
            "version": 1,
            "kind": "synthetic_recurrent",
            "paper_ready": True,
            "stages": [name],
            "steps": steps,
            "stage_schedule": [
                {
                    "name": name,
                    "steps": steps,
                    "execution": "compiled_recurrent" if steps > 1 else "compiled",
                    "timed": True,
                }
            ],
            "streams": ([{"name": "value", "input_arg": 0, "key": "value"}] if stream else []),
            "states": ([{"name": "state", "input_arg": 1, "output_index": 1}] if state else []),
            "quality": {"scope": "trajectory", "output_index": 0},
        }

    return descriptor_from_contract(
        root,
        child_contracts={
            "prefill": child("prefill", 1, stream=True, state=False),
            "decode": child("decode", 3, stream=True, state=True),
        },
    )


def _validate_tracer_shape(descriptor: SessionDescriptor) -> None:
    expected = synthetic_prefill_decode_descriptor()
    if descriptor.canonical_bytes != expected.canonical_bytes:
        raise ValueError("synthetic tracer requires the exact public prefill+3-decode descriptor")


def _c_bytes(value: bytes) -> str:
    return ",".join(str(byte) for byte in value)


def render_model_source(descriptor: SessionDescriptor) -> str:
    """C source exporting the common whole-session ABI without private constants."""
    _validate_tracer_shape(descriptor)
    public = descriptor.canonical_bytes
    calls = ",".join(f"{{{call.program}U,{call.step}U}}" for call in descriptor.calls)
    return f"""/* Generated public MRLNSES2 production-recipe tracer. */
typedef __SIZE_TYPE__ size_t;
typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;
_Static_assert(sizeof(u32) == 4, "u32 width");
_Static_assert(sizeof(u64) == 8, "u64 width");

static const u8 MAGIC[8] = {{77,82,76,78,83,69,83,50}};
static const u8 DESCRIPTOR[] = {{{_c_bytes(public)}}};
static const u32 CALLS[][2] = {{{calls}}};

static int same(const u8 *a, const u8 *b, size_t n) {{
  size_t i; for (i = 0; i < n; ++i) if (a[i] != b[i]) return 0; return 1;
}}
static int get32(const u8 *p, size_t n, size_t *at, u32 *v) {{
  size_t i = *at; if (i > n || n - i < 4) return -1;
  *v = ((u32)p[i] << 24) | ((u32)p[i+1] << 16) | ((u32)p[i+2] << 8) | p[i+3];
  *at = i + 4; return 0;
}}
static int get64(const u8 *p, size_t n, size_t *at, u64 *v) {{
  size_t i; u64 x = 0; if (*at > n || n - *at < 8) return -1;
  for (i = 0; i < 8; ++i) x = (x << 8) | p[*at + i];
  *at += 8; *v = x; return 0;
}}
static int put32(u8 *p, size_t n, size_t *at, u32 v) {{
  size_t i = *at; if (i > n || n - i < 4) return -1;
  p[i] = (u8)(v >> 24); p[i+1] = (u8)(v >> 16);
  p[i+2] = (u8)(v >> 8); p[i+3] = (u8)v; *at = i + 4; return 0;
}}
static int put64(u8 *p, size_t n, size_t *at, u64 v) {{
  size_t i; if (*at > n || n - *at < 8) return -1;
  for (i = 0; i < 8; ++i) p[*at + 7 - i] = (u8)(v >> (i * 8));
  *at += 8; return 0;
}}
static int frame(const u8 *request, size_t request_size, size_t *at,
                 u32 program, u32 input, u32 step, u64 *value) {{
  u32 got_program, got_input, got_step; u64 bytes;
  if (get32(request, request_size, at, &got_program) ||
      get32(request, request_size, at, &got_input) ||
      get32(request, request_size, at, &got_step) ||
      get64(request, request_size, at, &bytes)) return -1;
  if (got_program != program || got_input != input || got_step != step || bytes != 8) return -1;
  return get64(request, request_size, at, value);
}}
static int output_frame(u8 *response, size_t capacity, size_t *at,
                        u32 program, u32 output, u32 step, u64 value) {{
  return put32(response, capacity, at, program) || put32(response, capacity, at, output) ||
         put32(response, capacity, at, step) || put64(response, capacity, at, 8) ||
         put64(response, capacity, at, value) ? -1 : 0;
}}

int {ENTRYPOINT}(const char *runtime_root,
                 const u8 *request, size_t request_size,
                 u8 *response, size_t response_capacity, size_t *response_size) {{
  size_t at = 0, out = 0, i; u32 descriptor_size, frame_count; u64 seed, delta[3], state[3];
  (void)runtime_root;
  if (!request || !response || !response_size || request_size < 13) return 10;
  if (!same(request, MAGIC, 8) || request[8] != 1) return 11;
  at = 9;
  if (get32(request, request_size, &at, &descriptor_size) ||
      descriptor_size != sizeof(DESCRIPTOR) || at > request_size ||
      request_size - at < descriptor_size || !same(request + at, DESCRIPTOR, descriptor_size)) return 12;
  at += descriptor_size;
  if (get32(request, request_size, &at, &frame_count) || frame_count != 4) return 13;
  if (frame(request, request_size, &at, 0, 0, 0, &seed)) return 14;
  for (i = 0; i < 3; ++i)
    if (frame(request, request_size, &at, 1, 0, (u32)i, &delta[i])) return 15;
  if (at != request_size) return 16;

  /* Prefill is the only source of initial recurrent state. */
  state[0] = seed * 3 + delta[0];
  state[1] = state[0] * 3 + delta[1];
  state[2] = state[1] * 3 + delta[2];

  if (response_capacity < 13 + sizeof(DESCRIPTOR)) return 20;
  for (i = 0; i < 8; ++i) response[out++] = MAGIC[i];
  response[out++] = 2;
  if (put32(response, response_capacity, &out, (u32)sizeof(DESCRIPTOR))) return 21;
  if (out > response_capacity || response_capacity - out < sizeof(DESCRIPTOR)) return 21;
  for (i = 0; i < sizeof(DESCRIPTOR); ++i) response[out++] = DESCRIPTOR[i];
  if (put32(response, response_capacity, &out, 4)) return 22;
  for (i = 0; i < 4; ++i)
    if (put32(response, response_capacity, &out, CALLS[i][0]) ||
        put32(response, response_capacity, &out, CALLS[i][1])) return 22;
  if (put32(response, response_capacity, &out, 3)) return 23;
  for (i = 0; i < 3; ++i)
    if (output_frame(response, response_capacity, &out, 1, 0, (u32)i, state[i])) return 23;
  *response_size = out;
  return 0;
}}
"""


def render_runner_source(*, response_capacity: int | None = None) -> str:
    if response_capacity is not None and response_capacity < 1:
        raise ValueError("session runner response capacity must be positive")
    response_capacity_declaration = (
        f", response_capacity = {response_capacity}ULL" if response_capacity is not None else ""
    )
    response_allocation = "response_capacity" if response_capacity is not None else "used + 4096"
    return f"""/* Host driver for the common whole-session ABI. */
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
extern int {ENTRYPOINT}(const char *, const unsigned char *, size_t,
                        unsigned char *, size_t, size_t *);
int main(int argc, char **argv) {{
  unsigned char *request = 0, *response = 0;
  size_t used = 0, capacity = 4096, output_size = 0{response_capacity_declaration};
  int byte, rc; request = (unsigned char *)malloc(capacity);
  if (!request) return 90;
  while ((byte = fgetc(stdin)) != EOF) {{
    if (used == capacity) {{
      unsigned char *grown; capacity *= 2; grown = (unsigned char *)realloc(request, capacity);
      if (!grown) {{ free(request); return 91; }} request = grown;
    }}
    request[used++] = (unsigned char)byte;
  }}
  response = (unsigned char *)malloc({response_allocation});
  if (!response) {{ free(request); return 92; }}
  rc = {ENTRYPOINT}(argc > 1 ? argv[1] : ".", request, used,
                    response, {response_allocation}, &output_size);
  if (!rc && fwrite(response, 1, output_size, stdout) != output_size) rc = 93;
  free(response); free(request); return rc;
}}
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(argv: Sequence[str], *, where: str) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(list(argv), capture_output=True, timeout=60)
    if completed.returncode:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{where} failed ({completed.returncode}): {stderr}")
    return completed


def build_relocatable_object(
    source: str | Path, output: str | Path, *, compiler: str | Path, flags: Sequence[str] = ()
) -> tuple[str, ...]:
    """Compile one deterministic composite source to a relocatable object."""
    source, output = Path(source).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    argv = (str(Path(compiler).resolve()), *flags, "-std=c11", "-O2", "-c", str(source), "-o", str(output))
    _run(argv, where="relocatable-object compilation")
    return argv


def build_tracer_package(root: str | Path, *, compiler: str | Path) -> TracerPackage:
    """Materialize and locally link the synthetic package through the common ABI."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    descriptor = synthetic_prefill_decode_descriptor()
    source = root / "model_session.c"
    object_path = root / "model_session.o"
    runner_source = root / "runner.c"
    runner = root / "runner"
    receipt = root / "package_receipt.json"
    source.write_text(render_model_source(descriptor), encoding="utf-8")
    runner_source.write_text(render_runner_source(), encoding="utf-8")
    (root / "session_descriptor.json").write_bytes(descriptor.canonical_bytes + b"\n")
    object_argv = build_relocatable_object(source, object_path, compiler=compiler)
    link_argv = (
        str(Path(compiler).resolve()),
        "-std=c11",
        "-O2",
        str(runner_source),
        str(object_path),
        "-o",
        str(runner),
    )
    _run(link_argv, where="tracer runner link")
    runner.chmod(runner.stat().st_mode | 0o100)
    record = {
        "schema": "merlin.paper.synthetic-session-package/v1",
        "entrypoint": ENTRYPOINT,
        "descriptor_sha256": descriptor.sha256,
        "source_sha256": _sha256(source),
        "object_sha256": _sha256(object_path),
        "runner_source_sha256": _sha256(runner_source),
        "runner_sha256": _sha256(runner),
        "object_argv": list(object_argv),
        "link_argv": list(link_argv),
    }
    receipt.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TracerPackage(root, descriptor, source, object_path, runner_source, runner, receipt)


def run_tracer_package(package: TracerPackage, request: bytes) -> bytes:
    completed = subprocess.run(
        [str(package.runner), str(package.root)],
        input=request,
        capture_output=True,
        timeout=30,
        env={"PATH": os.environ.get("PATH", "")},
    )
    if completed.returncode:
        raise RuntimeError(f"tracer rejected session packet with status {completed.returncode}")
    return completed.stdout


def inspect_relocatable_object(
    object_path: str | Path, *, readelf: str | Path, nm: str | Path, expected_machine: str
) -> ObjectEvidence:
    """Prove ELF class/type/machine and the exported common entrypoint."""
    object_path = Path(object_path).resolve()
    header = _run((str(Path(readelf).resolve()), "-h", str(object_path)), where="ELF-header inspection").stdout.decode(
        "utf-8", errors="replace"
    )
    fields: dict[str, str] = {}
    for line in header.splitlines():
        key, separator, value = line.strip().partition(":")
        if separator:
            fields[key] = value.strip()
    symbols_text = _run(
        (str(Path(nm).resolve()), "-g", "--defined-only", str(object_path)), where="symbol inspection"
    ).stdout.decode("utf-8", errors="replace")
    symbols = tuple(parts[-1] for line in symbols_text.splitlines() if len(parts := line.split()) >= 2)
    elf_type = fields.get("Type", "").split(None, 1)[0]
    machine = fields.get("Machine", "")
    if elf_type != "REL":
        raise ValueError(f"object is not relocatable ELF: {fields.get('Type')!r}")
    if expected_machine not in machine:
        raise ValueError(f"object machine differs: expected {expected_machine!r}, got {machine!r}")
    if ENTRYPOINT not in symbols:
        raise ValueError(f"object does not export {ENTRYPOINT}")
    return ObjectEvidence(object_path, fields.get("Class", ""), fields.get("Type", ""), machine, symbols)
