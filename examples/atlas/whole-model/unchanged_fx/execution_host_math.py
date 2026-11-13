"""Validated component text replacement for native host math execution packaging.

Production C, its RV64 policy, and the native application inventory are immutable.
Complete flattened components are matched and replaced in a separate package.
"""
from hashlib import sha256
import json
from math import prod
from pathlib import Path
import struct

from merlin.compile.c_sources import c_source_pair
from merlin.targetgen.host_linkage_contract import validate_linkage_contract

PRODUCTION_ARCHIVE = '0626df7b1ce6426945820fd03dbd6752931c7594676c94f2043d5673588d178b'


def _hash(data):
    return sha256(data).hexdigest()


def _member(root, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('unsafe execution source member')
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('execution source escapes package')
    return target


def _native_source(name, symbol, count, angles):
    table = ','.join(str(value) + 'u' for value in angles)
    size = count * 4
    return f'''#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <fenv.h>
#include <xmmintrin.h>
#if !defined(__x86_64__) || defined(__FAST_MATH__)
#error validation unary math requires x86_64 without fast-math
#endif
_Static_assert(CHAR_BIT==8 && FLT_RADIX==2 && FLT_MANT_DIG==24 && sizeof(float)==4 && SIZE_MAX>={size}ULL,"validation unary representation");
static const uint32_t {name}_angles[]={{{table}}};
static float (*volatile const {name}_call)(float)={symbol};
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs || !inputs[0] || !outputs[0]) return -1;
  uintptr_t a=(uintptr_t)inputs[0],b=(uintptr_t)outputs[0];
  if (a>UINTPTR_MAX-{size}u || b>UINTPTR_MAX-{size}u || (a<b+{size}u && b<a+{size}u)) return -1;
  if (fegetround()!=FE_TONEAREST || (_mm_getcsr() & ((3u<<13)|(1u<<15)|(1u<<6)))) return -3;
  for(size_t i=0;i<{count}u;++i) {{
    uint32_t bits; memcpy(&bits,(const unsigned char *)inputs[0]+4u*i,4);
    if ((bits&0x7fffffffu)>=0x7f800000u) return -2;
    size_t lo=0,hi={len(angles)}u;
    while(lo<hi) {{ size_t mid=lo+(hi-lo)/2; if({name}_angles[mid]<bits) lo=mid+1; else hi=mid; }}
    if(lo=={len(angles)}u || {name}_angles[lo]!=bits) return -4;
  }}
  for(size_t i=0;i<{count}u;++i) {{
    float x,y; memcpy(&x,(const unsigned char *)inputs[0]+4u*i,4);
    y={name}_call(x); memcpy((unsigned char *)outputs[0]+4u*i,&y,4);
  }}
  return 0;
}}
'''


def _selection(manifest, equivalence_receipt, archive, rv64_output):
    gate_bytes = equivalence_receipt.read_bytes()
    gate = json.loads(gate_bytes)
    native = validate_linkage_contract(gate['linkage'])
    native_archive_sha256 = native['archive_sha256']
    if (gate.get('schema') != 'rotary.execution-host-math-domain-check.v1'
            or gate.get('gate') != 'PASS' or gate.get('rounding') != 'RNE'
            or native['symbols'] != ['cosf', 'sinf']
            or gate.get('execution_host_archive_sha256') != native_archive_sha256
            or _hash(archive.read_bytes()) != native_archive_sha256
            or gate.get('production_rv64_archive_sha256') != PRODUCTION_ARCHIVE
            or gate.get('domain', {}).get('distinct_input_lanes') != 65536
            or gate.get('domain', {}).get('compared_f32_trig_words') != 131072
            or any(gate.get(key) != 0 for key in ('angle_mismatch_count', 'angle_input_mismatch_count',
                                                'f32_mismatch_count', 'bf16_mismatch_count_after_cat'))
            or gate.get('execution_host_output_sha256') != gate.get('rv64_output_sha256')
            or not gate.get('rv64_output_sha256')):
        raise ValueError('native host math equivalence gate or archive differs')
    for op in ('sin', 'cos'):
        if gate.get('results', {}).get(op) != {'f32_mismatch_count': 0, 'bf16_mismatch_count_after_cat': 0}:
            raise ValueError('native host math operation gate differs')
    domain_bytes = rv64_output.read_bytes()
    if len(domain_bytes) != 65536 * 12 or _hash(domain_bytes) != gate['rv64_output_sha256']:
        raise ValueError('qualified angle domain size differs')
    angle_bytes = b''.join(domain_bytes[i:i+4] for i in range(0, len(domain_bytes), 12))
    angle_hash = _hash(angle_bytes)
    if any(gate.get('output_hashes', {}).get(engine, {}).get('angle_sha256') != angle_hash
           for engine in ('rv64', 'execution_host')):
        raise ValueError('qualified angle domain hash differs')
    angles = sorted(set(struct.unpack('<65536I', angle_bytes)))
    if any((bits & 0x7fffffff) >= 0x7f800000 for bits in angles):
        raise ValueError('qualified domain contains nonfinite angles')
    production_linkage = [validate_linkage_contract(value) for value in manifest['application_interface']['host_linkage']]
    expected = {**native, 'archive_sha256': PRODUCTION_ARCHIVE}
    if production_linkage != [expected]:
        raise ValueError('production unary archive pin differs')
    return ({'production_linkage': production_linkage, 'execution_linkage': [native],
             'equivalence_receipt_sha256': _hash(gate_bytes), 'qualified_angle_sha256': angle_hash,
             'qualified_distinct_encodings': len(angles), 'archive': str(archive.resolve()),
             'link_flags': ['-Wl,-Bsymbolic-functions', '-Wl,--exclude-libs,ALL',
                            '-Wl,--trace-symbol=sinf', '-Wl,--trace-symbol=cosf']}, angles)


def validate_execution_host_math_selection(manifest, equivalence_receipt: Path, archive: Path,
                                           rv64_output: Path):
    """Authenticate the explicit selection before any runtime package is copied."""
    return _selection(manifest, equivalence_receipt, archive, rv64_output)[0]


def prepare_execution_host_math(manifest, production_package: Path, execution_package: Path,
                                equivalence_receipt: Path, archive: Path, rv64_output: Path):
    """Replace validated component text; return link/provenance data for this run."""
    if (execution_package.resolve().is_relative_to(production_package.resolve())
            or production_package.resolve().is_relative_to(execution_package.resolve())):
        raise ValueError('validation package must be separate from production')
    selection, angles = _selection(manifest, equivalence_receipt, archive, rv64_output)
    expected = selection['production_linkage'][0]
    source_name = manifest['application_interface']['source']
    original = _member(production_package, source_name).read_bytes()
    inventory = manifest['stage_artifacts']
    if inventory.get('c-package/' + source_name) != _hash(original):
        raise ValueError('production combined source hash differs')
    request_bytes = (production_package.parent / 'source-request.json').read_bytes()
    if inventory.get('source-request.json') != _hash(request_bytes):
        raise ValueError('production source request hash differs')
    source_request = json.loads(request_bytes)
    nodes = {node['id']: node for node in source_request['nodes']}
    selected, components, seen = original.decode(), [], set()
    for region in manifest['regions']:
        desc = region['description']
        if desc.get('arithmetic') != 'selected-library-source-math':
            continue
        op = desc.get('source_intrinsic')
        symbol = {'source_f32_sin_unqualified': 'sinf', 'source_f32_cos_unqualified': 'cosf'}.get(op)
        if (symbol is None or region['placement'] != 'host' or region['native_programs']
                or desc.get('host_isa') != 'rv64gc' or desc.get('host_abi') != 'lp64d'
                or desc.get('numeric_policy') != 'selected-libm-rne-finite-v1'
                or desc.get('effects') != 'enabled-fpu-rne-unobserved-flags-and-errno-v1'
                or desc.get('source_request_digest') != region['source_request_digest']
                or validate_linkage_contract(desc['host_linkage']) != {**expected, 'symbols': [symbol]}):
            raise ValueError('unsupported production unary region or pin')
        name = desc['implementation']
        if not name.isidentifier() or len(region['outputs']) != 1:
            raise ValueError('invalid unary region ABI')
        node = nodes[region['outputs'][0]]
        if (node['op'] != op or node['inputs'] != region['inputs'] or len(node['inputs']) != 1
                or node['type']['dtype'] != 'f32' or node['type'] != nodes[node['inputs'][0]]['type']):
            raise ValueError('unary source request differs')
        count = prod(node['type']['shape'])
        if count <= 0 or count * 4 > 2**63 - 1:
            raise ValueError('invalid unary source extent')
        member = f'regions/{name}/{name}.c'
        header = f'regions/{name}/{name}.h'
        raw = _member(production_package, member).read_bytes()
        if inventory.get('c-package/' + member) != _hash(raw):
            raise ValueError('production region source hash differs')
        # Flatten the complete component, including nested directives, then
        # validate its exact text before replacing that text in the combined C.
        packed = c_source_pair('validation_component', public_headers={header: _member(production_package, header).read_text()},
                               private_headers={}, sources={member: raw.decode()})['validation_component.c']
        marker = '#line 1 ' + json.dumps(member) + '\n'
        flattened = packed[packed.index(marker):].removesuffix('\n\n')
        if selected.count(flattened) != 1:
            raise ValueError('production component does not match combined C exactly')
        native_source = _native_source(name, symbol, count, angles)
        logical = f'validation-native/{name}.c'
        replacement_pack = c_source_pair('validation_component', public_headers={}, private_headers={},
                                         sources={logical: native_source})['validation_component.c']
        replacement_marker = '#line 1 ' + json.dumps(logical) + '\n'
        replacement = replacement_pack[replacement_pack.index(replacement_marker):].removesuffix('\n\n')
        selected = selected.replace(flattened, replacement, 1)
        components.append({'production_component': member, 'execution_component': logical,
                           'production_component_sha256': _hash(flattened.encode()),
                           'execution_component_sha256': _hash(replacement.encode()),
                           'source_request_digest': region['source_request_digest'],
                           'execution_source_request_digest': region['source_request_digest'], 'elements': count,
                           'symbol': symbol})
        seen.add(symbol)
    if seen != {'cosf', 'sinf'}:
        raise ValueError('execution selection must cover both qualified unary symbols')
    destination = _member(execution_package, source_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(selected.encode())
    return {**selection, 'scope': 'validation-only native host unary component text replacement', 'source': source_name,
            'production_source_sha256': _hash(original), 'execution_source_sha256': _hash(selected.encode()),
            'components': components}
