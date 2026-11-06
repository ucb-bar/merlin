"""Strict validation shared by original-model backfill reuse and admission."""
from dataclasses import dataclass, field
import hashlib
import importlib
import json
from pathlib import Path
import struct
import subprocess

PIN = '47b8599d8ffbe5ddd96059ffe2b5be2427a410f5'
EPILOGUE = struct.pack('<6I', 0x13, 0xc1102ff3, 0x13, 0xc10f9073, 0x13, 0x73)
SOURCE_SCHEMA = 'atlas.smol_original_gate_sources.v2'


def safe(value):
    path = Path(value).absolute()
    if '..' in path.parts:
        raise ValueError('restricted path component')
    if any(prefix.is_symlink() for prefix in reversed([path, *path.parents])):
        raise ValueError('symlink paths are unsupported')
    return path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def payload(item):
    raw = safe(item.get('path', item.get('file'))).read_bytes()
    require(len(raw) == item['bytes'] and digest(raw) == item['sha256'], 'captured payload changed')
    return raw


@dataclass(frozen=True)
class Bindings:
    model: dict
    runner_sha256: str
    harness_sha256: str
    helper_source_sha256: dict = field(default_factory=dict)
    source_binding_schema: str = SOURCE_SCHEMA


def helper_sources(runner):
    root = safe(runner).parent
    return {'smol_capture_audit': safe(root / 'smol_capture_audit.py'),
            'smol_capture': safe(root / 'smol_capture/__init__.py'),
            'smol_capture.stats': safe(root / 'smol_capture/stats.py')}


def helper_bindings(runner):
    sources = helper_sources(runner)
    for name, path in sources.items():
        module = importlib.import_module(name)
        require(safe(module.__file__) == path, 'gate helper imported outside selected source: ' + name)
    return {str(path): digest(path.read_bytes()) for path in sources.values()}


def bindings(model_root, runner, harness):
    root = safe(model_root)
    require(subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == PIN,
            'original model revision changed')
    require(not subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip(),
            'original model is dirty')
    return Bindings(dict(commit=PIN, numerics='rtl', root=str(root)),
                    digest(safe(runner).read_bytes()), digest(safe(harness).read_bytes()), helper_bindings(runner))


def validate_receipt(event, frozen, saved, current):
    """Authenticate the event, complete arena, runtime program, DMA, and source bindings."""
    require(json.loads(frozen) == event, 'frozen event differs from current capture')
    require(saved['status'] == 'PASS', 'original receipt did not pass')
    require(saved['index'] == event['index'] and saved['kind'] == event['kind'], 'event identity differs')
    require(saved['event_sha256'] == digest(frozen), 'frozen event hash differs')
    require(saved['runner_sha256'] == current.runner_sha256 and
            saved['harness_sha256'] == current.harness_sha256, 'runner or harness changed')
    require(saved.get('source_binding_schema') == current.source_binding_schema and
            saved.get('helper_source_sha256') == current.helper_source_sha256, 'gate helper sources changed')
    require(saved['model'] == current.model and saved['original_model'] is True, 'original model binding differs')
    require(saved['native_backend'] is False and saved['math_override'] is False, 'development backend receipt')
    require(saved['initial_csrs'] == {'0xc11': event['sequence']}, 'completion sequence differs')
    initial, output = payload(event['initial']), payload(event['output'])
    canonical, program = payload(event['program']), payload(event['runtime_program'])
    require(canonical and len(canonical) % 4 == 0, 'canonical program extent differs')
    words = struct.unpack('<' + 'I' * (len(canonical) // 4), canonical)
    require(words[-1] == 0x73 and 0x73 not in words[:-1], 'canonical terminal ECALL differs')
    require(program == canonical[:-4] + EPILOGUE and event['runtime_epilogue_sha256'] == digest(EPILOGUE),
            'runtime completion epilogue differs')
    payload(event['source'])
    payload(event['runtime_source'])
    offset, size = event['output_offset'], event['output_bytes']
    require(len(output) == size and 0 <= offset <= len(initial) - size, 'output extent differs')
    expected = initial[:offset] + output + initial[offset + size:]
    require(saved['full_arena_exact'] is True and saved['output_exact'] is True, 'full-arena check absent')
    require(saved['initial_sha256'] == digest(initial), 'initial arena differs')
    require(saved['expected_output_sha256'] == digest(output), 'output differs')
    require(saved['expected_arena_sha256'] == saved['actual_arena_sha256'] == event['expected_arena_sha256']
            == digest(expected), 'full arena hash differs')
    require(saved['runtime_program_executed'] is True, 'runtime program was not executed')
    require(saved['canonical_program_sha256'] == digest(canonical), 'canonical program differs')
    execution = saved['execution']
    require(saved['program_sha256'] == execution['program_sha256'] == digest(program), 'runtime program differs')
    require(execution['program_words'] * 4 == len(program) and execution['instructions'] > 0,
            'execution extent differs')
    require(execution['halt_pc'] == execution['program_words'] - 1, 'execution did not reach final ECALL')
    for key in ('load_bytes', 'store_bytes'):
        require(execution['observed'][key] == event['requested_dma'][key] == event['observed_dma'][key],
                'DMA differs: ' + key)
    return saved


def validate_completed(event, frozen, receipt, completion, current):
    """A receipt is reusable only after a successful bound worker process exit."""
    raw = safe(receipt).read_bytes()
    result = json.loads(safe(completion).read_bytes())
    require(type(result.get('returncode')) is int and result['returncode'] == 0, 'worker process did not succeed')
    require(result.get('index') == event['index'] and result.get('event_sha256') == digest(frozen),
            'worker completion event differs')
    require(result.get('receipt_sha256') == digest(raw), 'worker completion receipt differs')
    return validate_receipt(event, frozen, json.loads(raw), current)
