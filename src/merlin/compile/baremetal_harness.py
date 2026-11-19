"""Emit one recorded C-package invocation for a caller-selected platform adapter.

The adapter header includes the application header and supplies
``int harness_platform_bind(context_type *, size_t max_polls)``. It owns device
addresses, memory policy and any platform canary. The application supplies its
parameter binder and invocation ABI through application-c.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

_WIDTHS = {"i1": 1, "i8": 1, "u8": 1, "bf16": 2, "f16": 2,
           "i16": 2, "i32": 4, "f32": 4, "i64": 8, "f64": 8}


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_]\w*", value):
        raise ValueError("invalid C identifier")
    return value


def _bytes(kind):
    shape = kind["shape"]
    if any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("positive static extents required")
    try:
        width = _WIDTHS[kind["dtype"]]
    except KeyError as exc:
        raise ValueError("unsupported recorded dtype") from exc
    return math.prod(shape) * width


def emit_harness(interface_path, output, *, inputs, expected_sha256,
                 adapter_header="platform.h", parameter_address=None,
                 max_polls=200_000_000, math_archive=None):
    """Write main.c and its hash header; return reproducible build requirements.

    inputs maps every recorded input id to its raw byte file. Outputs are hashed
    in interface order, without padding. expected_sha256 is that concatenation's
    authenticated reference digest. No conversion of recorded bytes is allowed.
    """
    interface_path, output = Path(interface_path), Path(output)
    raw = interface_path.read_bytes()
    spec = json.loads(raw)
    binding = spec["platform_binding"]
    if spec["pointer_abi"] != "opaque_native_dense_buffers" or binding["completion"] != "synchronous":
        raise ValueError("synchronous opaque buffer ABI required")
    if set(inputs) != set(spec["inputs"]) or len(spec["inputs"]) != len(spec["input_types"]) or len(spec["outputs"]) != len(spec["output_types"]):
        raise ValueError("complete input and output roster required")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("canonical expected SHA256 required")
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", adapter_header):
        raise ValueError("invalid adapter header")
    if type(max_polls) is not int or max_polls <= 0:
        raise ValueError("positive poll limit required")
    linkage = spec.get("host_linkage", [])
    archive_hash = hashlib.sha256(Path(math_archive).read_bytes()).hexdigest() if math_archive else None
    if any(item["archive_sha256"] != archive_hash for item in linkage):
        raise ValueError("missing or incorrect pinned host archive")
    params = spec.get("model_parameters")
    if params is not None:
        alignment = spec.get("workspace_alignment", 64)
        size = params["payload_bytes"]
        if (type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1)
                or type(parameter_address) is not int or parameter_address <= 0
                or parameter_address % alignment or type(size) is not int or size <= 0
                or parameter_address + size >= 1 << 64):
            raise ValueError("explicit aligned fixed parameter span required")
    lines = [f'#include "{adapter_header}"', '#include "harness_sha256.h"', '#include <stdint.h>', '#include <stdio.h>', '#include <string.h>',
             f'static {_identifier(binding["context_type"])} context;']
    identities = {}
    for i, (name, kind) in enumerate(zip(spec["inputs"], spec["input_types"], strict=True)):
        data = Path(inputs[name]).read_bytes()
        if len(data) != _bytes(kind):
            raise ValueError("recorded input byte extent differs")
        identities[name] = hashlib.sha256(data).hexdigest()
        lines.append(f'static _Alignas(64) const unsigned char input_{i}[]={{'+','.join(map(str,data))+'};')
    sizes = [_bytes(kind) for kind in spec["output_types"]]
    for i, size in enumerate(sizes):
        lines.append(f'static _Alignas(64) unsigned char output_{i}[{size}];')
    lines += ['int main(void) {', '  (void)merlin_output_sha256;',
              f'  if(harness_platform_bind(&context,{max_polls}ULL)) {{ puts("platform: FAIL"); return 1; }}']
    if params is not None:
        lines.append(f'  if({_identifier(binding["bind_parameters"])}(&context,(const void *)(uintptr_t){parameter_address}ULL,{params["payload_bytes"]}ULL)) {{ puts("parameters: FAIL"); return 1; }}')
    lines += ['  const void *inputs[]={'+(','.join(f'input_{i}' for i in range(len(inputs))) or 'NULL')+'};',
              '  void *outputs[]={'+(','.join(f'output_{i}' for i in range(len(sizes))) or 'NULL')+'};',
              f'  if({_identifier(binding["invoke"])}(&context,inputs,outputs)) {{ puts("invoke: FAIL"); return 1; }}',
              '  merlin_sha256 hash; unsigned char digest[32]; char hex[65];', '  merlin_sha256_init(&hash);']
    for i, size in enumerate(sizes):
        lines.append(f'  merlin_sha256_update(&hash,output_{i},{size}ULL);')
    lines += ['  merlin_sha256_final(&hash,digest);',
              '  for(unsigned i=0;i<32;i++) { hex[2*i]="0123456789abcdef"[digest[i]>>4]; hex[2*i+1]="0123456789abcdef"[digest[i]&15]; }',
              '  hex[64]=0; printf("SHA256=%s\\n",hex);',
              f'  int fail=strcmp(hex,"{expected_sha256}")!=0;', '  puts(fail?"generated: FAIL":"generated: PASS"); return fail;', '}']
    output.mkdir(parents=True, exist_ok=True)
    (output/'main.c').write_text('\n'.join(lines)+'\n')
    from merlin.common.paths import runtime_dir
    (output/'harness_sha256.h').write_bytes((runtime_dir()/'baremetal/spike/output_sha256.h').read_bytes())
    record = dict(interface_sha256=hashlib.sha256(raw).hexdigest(), input_sha256=identities,
                  expected_sha256=expected_sha256, parameter_address=parameter_address,
                  max_polls=max_polls, cflags=['-ffp-contract=off', '-frounding-math'],
                  host_archive_sha256=archive_hash, host_linkage=linkage)
    (output/'harness.json').write_text(json.dumps(record,indent=2)+'\n')
    return record
