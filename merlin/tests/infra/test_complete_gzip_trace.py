"""Every encoded/decoded byte remains bound across streamed source observation."""
import gzip
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from merlin.common import invocation_record
from merlin.common.complete_gzip_trace import (
    TraceLimits, capture_gzip_stderr, seal_transport, source_trace_text_view,
)
from merlin.perf.executed_trace import executed_trace
from merlin.perf.streamed_executed_trace import iter_source_commands


def capture(tmp_path, program, limits=TraceLimits()):
    argv = [sys.executable, '-c', program]
    with invocation_record.observe(tmp_path, stage='complete-gzip-test', argv=argv,
                                   inputs=(), dependencies=(), env=dict(os.environ)) as observed:
        completed, stats = capture_gzip_stderr(
            argv, stdout_path=observed.directory/'stdout.bin',
            stderr_path=observed.directory/'stderr.bin', env=dict(os.environ),
            timeout_s=10, limits=limits)
        observed.complete_streams(completed.returncode)
        invocation = observed.path
    seal_transport(invocation, stats, limits)
    return invocation, stats


def test_complete_bytes_roundtrip_and_original_invocation_pin(tmp_path):
    raw = b'core 0: 3 0x1000 (0x00000013) x1 0x01\n' * 1000 + b'final line\n'
    program = "import os;os.write(1,b'complete output\\n');os.write(2," + repr(raw) + ')'
    invocation, statistics = capture(tmp_path, program)
    assert statistics['streams']['stderr']['decoded_sha256'] == hashlib.sha256(raw).hexdigest()
    assert statistics['streams']['stderr']['decoded_bytes'] == len(raw)
    assert gzip.decompress(open(statistics['stderr']['path'],'rb').read()) == raw
    view = source_trace_text_view(invocation)
    assert list(view.splitlines()) == raw.decode().splitlines()
    assert list(source_trace_text_view(invocation,role='stdout').splitlines())==['complete output']
    assert list(view.splitlines()) == raw.decode().splitlines()
    receipt = json.loads(view.transport_path.read_bytes())
    receipt['statistics']['streams']['stderr']['decoded_sha256'] = '0' * 64
    view.transport_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='changed'):
        list(view.splitlines())
    with pytest.raises(ValueError, match='decoded bytes'):
        list(source_trace_text_view(invocation).splitlines())


@pytest.mark.parametrize('field,bound', [('max_decoded_bytes',1024),('max_compressed_bytes',16)])
def test_limits_refuse_complete_capture_without_a_success_receipt(tmp_path, field, bound):
    limits=replace(TraceLimits(), **{field:bound}, max_line_bytes=64)
    with pytest.raises(ValueError, match='byte limit'):
        capture(tmp_path, "import os;os.write(2,b'x'*8192)", limits)
    assert not list(tmp_path.rglob('trace-transport.json'))


def test_partial_gzip_or_oversized_line_refuses(tmp_path):
    invocation, statistics = capture(tmp_path, "import os;os.write(2,b'x'*1000+b'\\n')",
                                    replace(TraceLimits(),max_line_bytes=100))
    with pytest.raises(ValueError, match='line byte limit'):
        list(source_trace_text_view(invocation).splitlines())


def test_stdout_limit_is_enforced_before_payload_write(tmp_path):
    limits=replace(TraceLimits(),max_stdout_bytes=32)
    with pytest.raises(ValueError,match='stdout exceeds decoded byte limit'):
        capture(tmp_path,"import os;os.write(1,b'x'*8192)",limits)
    output=next(tmp_path.rglob('stdout.bin'))
    assert output.stat().st_size<=32


def test_streaming_retirement_replay_matches_original_and_preserves_fences(monkeypatch):
    from merlin.targetgen.rocc import semantics
    monkeypatch.setattr(semantics,'decode_instruction',lambda f,a,b,i:('TEST',dict(a=a['raw'],b=b['raw'])))
    word = (6<<25)|(2<<20)|(1<<15)|0x0b
    log = '\n'.join(['core 0: 3 0x1000 (0x13) x1 0x1234',
                      'core 0: 3 0x1004 (0x13) x2 0x5678',
                      f'core 0: 3 0x1008 (0x{word:x})',
                      'core 0: 3 0x100c (0x0f)',
                      'core 0: 3 0x1010 (0x13) x1 0x9999',
                      f'core 0: 3 0x1014 (0x{word:x})'])
    expected=executed_trace(log.splitlines(),isa={'CUSTOM_OPCODE':0x0b})['instructions']
    actual=list(iter_source_commands(log,isa={'CUSTOM_OPCODE':0x0b}))
    assert [row for row in actual if row['class']!='FENCE']==expected
    assert actual[1]=={'class':'FENCE'}
    with pytest.raises(ValueError,match='retired'):
        list(iter_source_commands(log,isa={'CUSTOM_OPCODE':0x0b},max_retired=2))
    with pytest.raises(ValueError,match='event limit'):
        list(iter_source_commands(log,isa={'CUSTOM_OPCODE':0x0b},max_commands=1))
    with pytest.raises(ValueError,match='single hart'):
        list(iter_source_commands(log.replace('core 0:','core 1:'),isa={'CUSTOM_OPCODE':0x0b}))
