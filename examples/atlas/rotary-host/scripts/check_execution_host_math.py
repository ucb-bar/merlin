"""Fail closed unless native selected-archive trig is bitwise equal to RV64."""
import hashlib
import json
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
from build_execution_host_math import ENV, INV_FREQ_SHA, OUT, PRODUCTION_SHA, ROOT, digest


def bf16(bits):
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff


def main():
    build = json.loads((OUT / 'build-receipt.json').read_text())
    production = ROOT / 'out/core-math/libcore-math-rotary.a'
    archive = OUT / 'libcore-math-execution-host.a'
    executable = OUT / 'execution_host_probe'
    assert digest(production) == build['production_archive_sha256'] == PRODUCTION_SHA
    assert digest(archive) == build['archive_sha256'] == build['linkage']['archive_sha256']
    assert digest(executable) == build['executable_sha256']
    for name, key in [('link-symbol-trace.txt', 'link_symbol_trace_sha256'),
                      ('execution_host_probe.map', 'link_map_sha256')]:
        assert digest(OUT / name) == build[key]
    freq_bytes = bytes.fromhex((ROOT / 'scripts/inv_freq.hex').read_text())
    assert hashlib.sha256(freq_bytes).hexdigest() == digest(OUT / 'inv_freq.bin') == INV_FREQ_SHA
    assert build['linkage']['symbols'] == ['cosf', 'sinf']
    native_path = OUT / 'execution-host-rotary.bin'
    command = ['nice', '-n', '10', str(executable), str(OUT / 'inv_freq.bin')]
    with native_path.open('wb') as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.PIPE, env=ENV)
    with (OUT / 'commands.jsonl').open('a') as log:
        log.write(json.dumps({'label': 'execute domain probe', 'command': command,
                              'returncode': result.returncode, 'stderr': result.stderr.decode(),
                              'OMP_NUM_THREADS': '1'}) + '\n')
    assert result.returncode == 0, result.stderr.decode()
    native = native_path.read_bytes()
    rv_path = ROOT / 'out/rv64-rotary.bin'
    rv = rv_path.read_bytes()
    assert len(native) == len(rv) == 2048 * 32 * 3 * 4
    freqs = struct.unpack('<32f', freq_bytes)
    rv_rows = list(struct.iter_unpack('<III', rv))
    native_rows = list(struct.iter_unpack('<III', native))
    angle_mismatch = 0
    angle_input_mismatch = 0
    counts = {'sin': {'f32_mismatch_count': 0, 'bf16_mismatch_count_after_cat': 0},
              'cos': {'f32_mismatch_count': 0, 'bf16_mismatch_count_after_cat': 0}}
    outputs = {host: {name: bytearray() for name in ('angle', 'sin', 'cos', 'sin_bf16_after_cat', 'cos_bf16_after_cat')}
               for host in ('rv64', 'execution_host')}
    examples = []
    for i, (actual, expected) in enumerate(zip(native_rows, rv_rows)):
        angle_bits = struct.unpack('<I', struct.pack('<f', float(i // 32) * freqs[i % 32]))[0]
        angle_mismatch += actual[0] != expected[0]
        angle_input_mismatch += actual[0] != angle_bits or expected[0] != angle_bits
        for name, lane in [('sin', 1), ('cos', 2)]:
            differs = actual[lane] != expected[lane]
            counts[name]['f32_mismatch_count'] += differs
            counts[name]['bf16_mismatch_count_after_cat'] += 2 * (bf16(actual[lane]) != bf16(expected[lane]))
            if differs and len(examples) < 16:
                examples.append({'position': i // 32, 'frequency': i % 32, 'function': name,
                                 'execution_host_f32': hex(actual[lane]), 'rv64_f32': hex(expected[lane])})
        for host, row in [('rv64', expected), ('execution_host', actual)]:
            for lane, name in enumerate(('angle', 'sin', 'cos')):
                outputs[host][name].extend(struct.pack('<I', row[lane]))
    # Cat duplicates all 32 frequency lanes within each position, before BF16 conversion.
    for host, rows in [('rv64', rv_rows), ('execution_host', native_rows)]:
        for start in range(0, len(rows), 32):
            for name, lane in [('sin', 1), ('cos', 2)]:
                words = [bf16(row[lane]) for row in rows[start:start + 32]] * 2
                outputs[host][name + '_bf16_after_cat'].extend(struct.pack('<64H', *words))
    output_hashes = {}
    for host, streams in outputs.items():
        output_hashes[host] = {}
        for name, data in streams.items():
            path = OUT / (host + '-' + name + '.bin')
            path.write_bytes(data)
            output_hashes[host][name + '_sha256'] = digest(path)
    failed = angle_mismatch or angle_input_mismatch or any(any(v.values()) for v in counts.values())
    receipt = {'schema': 'rotary.execution-host-math-domain-check.v1', 'gate': 'FAIL' if failed else 'PASS',
               'qualification': ('Native selected-archive sinf/cosf bitwise equal to production RV64 over all 65536 captured angles.'
                                 if not failed else 'Native selected-archive trig equivalence failed.'),
               'core_math_revision': build['revision'], 'rounding': 'RNE',
               'execution_host_archive_path': str(archive), 'execution_host_archive_sha256': digest(archive),
               'production_rv64_archive_path': str(production), 'production_rv64_archive_sha256': digest(production),
               'linkage': build['linkage'], 'build_receipt_sha256': digest(OUT / 'build-receipt.json'),
               'execution_host_output_sha256': digest(native_path), 'rv64_output_sha256': digest(rv_path),
               'inv_freq_sha256': INV_FREQ_SHA, 'angle_mismatch_count': angle_mismatch,
               'angle_input_mismatch_count': angle_input_mismatch,
               'domain': {'positions': [0, 2047], 'frequencies': 32, 'distinct_input_lanes': 65536,
                          'compared_f32_trig_words': 131072, 'compared_bf16_words_after_cat': 262144},
               'results': counts, 'first_mismatches': examples, 'output_hashes': output_hashes,
               'f32_mismatch_count': sum(v['f32_mismatch_count'] for v in counts.values()),
               'bf16_mismatch_count_after_cat': sum(v['bf16_mismatch_count_after_cat'] for v in counts.values()),
               'probe_sha256': digest(executable), 'link_symbol_trace_sha256': digest(OUT / 'link-symbol-trace.txt'),
               'link_map_sha256': digest(OUT / 'execution_host_probe.map')}
    (OUT / 'equivalence-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    assert digest(production) == PRODUCTION_SHA
    print(json.dumps(receipt, indent=2))
    assert not failed, 'native selected-archive equivalence gate failed'


if __name__ == '__main__':
    main()
