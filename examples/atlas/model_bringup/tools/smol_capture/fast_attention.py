"""Explicit admitted attention development override; never an original-model gate.

Historical original receipts and payloads are validated from a configured portable
archive. This reuses prior admission and does not claim fresh original execution.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading

from smol_capture import model as original
from .fast_ipt import development_only_math_override, BACKEND_NAME, BASELINE_NPU_MODEL_COMMIT
import kernel_library  # noqa: E402,F401
from atlas_attention_v2_program import AttentionV2Config, assembly_v2

_EXECUTION_LOCK = threading.Lock()
_MODEL_FUNCTIONS = ('QuietLogger', 'Execution', 'assemble', '_branch', '_functional', 'execute')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path):
    return original.safe_path(path).read_bytes()


def _screen_model_tree(root):
    pending = [root]
    while pending:
        parent = pending.pop()
        for entry in os.scandir(parent):
            if entry.name == '.git':
                continue
            if entry.is_symlink():
                raise ValueError('symlink model paths are unsupported')
            if entry.is_dir(follow_symlinks=False):
                pending.append(Path(entry.path))


def baseline_binding():
    binding = original.model_binding()
    root = original.safe_path(original.MODEL_ROOT)
    if (binding != dict(commit=BASELINE_NPU_MODEL_COMMIT, numerics='rtl', root=str(root))
        or original.MODEL_PIN != BASELINE_NPU_MODEL_COMMIT):
        raise ValueError('development attention requires original47 RTL model sources')
    if os.environ.get('NPU_MODEL_RTL_NATIVE_MXU0', '0') not in ('', '0'):
        raise ValueError('native math backends are not admitted here')
    _screen_model_tree(root)
    if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip():
        raise ValueError('original model worktree is not clean')
    from npu_model.hardware.arch_state import ArchState
    if original.ArchState is not ArchState:
        raise ValueError('original model factory was already overridden')
    return binding


def kind(config):
    fields=asdict(config)
    # Exponents are constants in the same recipe. All FP8 bytes and BF16 partial
    # encodings are covered by the primitive admission; final data still gates
    # on the original model for every captured launch.
    for field in ('q_exponent','k_exponent','v_exponent','p_exponent','o_exponent'):
        del fields[field]
    return json.dumps(fields,sort_keys=True,separators=(',',':'))


def _execute_development(source,arena,output_offset,output_bytes):
    """Qualification only; process-local factory substitution, never a gate."""
    baseline_binding()
    arch_state=original.ArchState
    class DevelopmentState(arch_state):
        def __init__(self,config):
            super().__init__(config)
            if config.numerics != 'rtl':
                raise ValueError('require explicit RTL arithmetic selection')
            self.math=development_only_math_override(self.math)
    original.ArchState=DevelopmentState
    try:
        result=original.execute(source,arena,output_offset,output_bytes,timed=False)
    finally:
        original.ArchState=arch_state
    return result

def _ast_functions(raw, names):
    tree = ast.parse(raw)
    found = {node.name: ast.dump(node, include_attributes=False) for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names}
    if set(found) != set(names):
        raise ValueError('archived or current runner lacks required functions')
    return found


def _archive_file(root, entry):
    member = Path(entry['path'])
    if member.is_absolute() or '..' in member.parts:
        raise ValueError('archive member must be a contained relative path')
    path = original.safe_path(root / member)
    if not path.is_relative_to(root):
        raise ValueError('archive member escapes data root')
    raw = _read(path)
    if len(raw) != entry['bytes'] or digest(raw) != entry['sha256']:
        raise ValueError('archive payload hash mismatch: ' + str(member))
    return path, raw


@dataclass(frozen=True)
class Admission:
    data_root: Path
    admission_path: Path
    historical_runner_path: Path
    catalogue_path: Path
    original_sources: dict
    kinds: frozenset[str]
    cases: tuple[dict, ...]
    admission: dict
    pins: tuple[tuple[Path, str], ...]

    def validate_bookends(self):
        for path, expected in self.pins:
            if digest(_read(path)) != expected:
                raise ValueError('admission source bookend changed: ' + str(path))
        if baseline_binding() != self.original_sources:
            raise ValueError('selected original model binding changed')


def load_admission(*, data_root, admission_path=None, historical_runner_path=None, catalogue_path=None):
    """Validate explicit portable evidence once; no historical absolute path is read."""
    root = original.safe_path(data_root)
    path = original.safe_path(admission_path if admission_path is not None else root / 'admission.json')
    runner = original.safe_path(historical_runner_path if historical_runner_path is not None else root / 'historical-runner.py')
    catalogue_path = original.safe_path(catalogue_path if catalogue_path is not None else root / 'catalogue.json')
    raw, archived_runner, catalogue_raw = _read(path), _read(runner), _read(catalogue_path)
    admission, catalogue = json.loads(raw), json.loads(catalogue_raw)
    binding = baseline_binding()
    helper = original.safe_path(Path(__file__).with_name('fast_ipt.py'))
    module = original.safe_path(__file__)
    if (admission['status'] != 'ADMITTED_FOR_DEVELOPMENT_ONLY' or admission['gate_of_record'] is not False
        or admission['backend'] != BACKEND_NAME or admission['original_model_commit'] != BASELINE_NPU_MODEL_COMMIT
        or admission['original_binding']['commit'] != BASELINE_NPU_MODEL_COMMIT
        or admission['original_binding']['numerics'] != 'rtl' or admission['all_full_arenas_exact'] is not True
        or digest(_read(helper)) != admission['helper_sha256'] or digest(archived_runner) != admission['runner_sha256']):
        raise ValueError('helper, runner, or original-model admission binding differs')
    names = ('kind', '_execute_development')
    if _ast_functions(_read(module), names) != _ast_functions(archived_runner, names):
        raise ValueError('development execution or kind AST differs from admitted runner')
    if (catalogue['schema'] != 'atlas.smol_attention_admission_archive.v1'
        or catalogue['original_model_commit'] != BASELINE_NPU_MODEL_COMMIT
        or catalogue['helper_sha256'] != admission['helper_sha256']
        or catalogue['reused_historical_original_execution'] is not True
        or catalogue['new_original_math_executed'] is not False):
        raise ValueError('unsupported portable catalogue')
    pins = [(path, digest(raw)), (runner, digest(archived_runner)), (catalogue_path, digest(catalogue_raw)),
            (helper, admission['helper_sha256']), (module, digest(_read(module)))]
    files = {}
    for name, entry in catalogue['files'].items():
        filename, content = _archive_file(root, entry)
        files[name] = (filename, content)
        pins.append((filename, digest(content)))
    if files['admission'][1] != raw or files['runner'][1] != archived_runner:
        raise ValueError('explicit admission/runner differ from catalogue payloads')
    if (digest(files['review'][1]) != admission['source_review_sha256']
        or digest(files['primitive'][1]) != admission['primitive_receipt_sha256']
        or b'PASS_FAST_IPT_PRIMITIVES_ONLY' not in files['primitive'][1]):
        raise ValueError('primitive or source review receipt differs')
    model_path = original.safe_path(original.__file__)
    if _ast_functions(_read(model_path), _MODEL_FUNCTIONS) != _ast_functions(files['model_runner'][1], _MODEL_FUNCTIONS):
        raise ValueError('canonical model execution AST differs from historical original runner')
    pins.append((model_path, digest(_read(model_path))))
    rows, receipts = catalogue['cases'], admission['receipts']
    if len(rows) != len(receipts) or len(rows) != admission['launches'] or not rows:
        raise ValueError('incomplete historical full-arena catalogue')
    seen, kinds = set(), set()
    for index, (row, receipt) in enumerate(zip(rows, receipts)):
        if row['admission_index'] != index or row['name'] in seen or row['kind'] != receipt['kind']:
            raise ValueError('catalogue case identity differs')
        seen.add(row['name'])
        payload = {name: _archive_file(root, entry)[1] for name, entry in row['files'].items()}
        recorded = json.loads(payload['receipt'])
        if (digest(payload['receipt']) != receipt['original_receipt_sha256']
            or digest(payload['arena.bin']) != receipt['initial_sha256']
            or digest(payload['expected-arena.bin']) != receipt['expected_arena_sha256']
            or receipt['full_arena_exact'] is not True or receipt['independent_output_exact'] is not True
            or receipt['gate_of_record'] is not False or receipt['backend'] != BACKEND_NAME
            or receipt['original_binding'] != admission['original_binding']
            or recorded['expected_exact'] is not True or recorded['model'] != admission['original_binding']):
            raise ValueError('historical original receipt or full-arena identity differs')
        config = AttentionV2Config(**recorded['config'])
        offsets = recorded['offsets']
        source = payload['program.S'].decode()
        if kind(config) != receipt['kind'] or source != assembly_v2(config, offsets):
            raise ValueError('current attention recipe differs from admitted case')
        _, program_sha = original.assemble(source)
        functional = recorded['functional']
        development = receipt['development_execution']
        if (program_sha != functional['program_sha256']
            or any(development[key] != functional[key] for key in ('program_sha256', 'observed', 'instructions', 'program_words', 'halt_pc'))):
            raise ValueError('historical program or DMA identity differs')
        initial, expected = payload['arena.bin'], payload['expected-arena.bin']
        start, size = offsets['output'], offsets['output_bytes']
        if (start < 0 or size <= 0 or start + size > len(initial) or len(expected) != len(initial)
            or expected[:start] != initial[:start] or expected[start+size:] != initial[start+size:]
            or expected[start:start+size] != payload['expected-output.bin']):
            raise ValueError('historical complete-arena/output preservation differs')
        kinds.add(receipt['kind'])
    if kinds != set(admission['kinds']):
        raise ValueError('not every admitted recipe has a validated full-arena sample')
    result = Admission(root, path, runner, catalogue_path, binding, frozenset(kinds), tuple(rows), admission, tuple(pins))
    result.validate_bookends()
    return result


def execute_admitted(config, offsets, arena, *, admission):
    """Run only an explicitly configured admitted recipe; all final gates stay original."""
    if not isinstance(admission, Admission):
        raise ValueError('require a validated portable admission')
    with _EXECUTION_LOCK:
        admission.validate_bookends()
        recipe = kind(config)
        if recipe not in admission.kinds:
            raise ValueError('attention recipe is not admitted')
        source = assembly_v2(config, offsets)
        result = _execute_development(source, arena, offsets['output'], offsets['output_bytes'])
        admission.validate_bookends()
    return result, dict(backend=BACKEND_NAME, math_override=True, gate_of_record=False,
        original_sources=dict(admission.original_sources), helper_sha256=admission.admission['helper_sha256'],
        historical_runner_sha256=admission.admission['runner_sha256'],
        runner_sha256=digest(_read(__file__)), admission_sha256=digest(_read(admission.admission_path)),
        catalogue_sha256=digest(_read(admission.catalogue_path)), admitted_recipe=recipe,
        admission_reuse='historical25sample17kind full-arena admission; no fresh original math execution',
        historical_payloads_validated_on_load=True, data_root=str(admission.data_root))


def qualify_cached_case(admission, name):
    """Fast-only port check against the preserved historical original full arena."""
    found = [row for row in admission.cases if row['name'] == name]
    if len(found) != 1:
        raise ValueError('unknown or ambiguous portable case name')
    row = found[0]
    payload = {key: _archive_file(admission.data_root, entry)[1] for key, entry in row['files'].items()}
    recorded = json.loads(payload['receipt'])
    result, provenance = execute_admitted(AttentionV2Config(**recorded['config']), recorded['offsets'], payload['arena.bin'], admission=admission)
    if result.arena != payload['expected-arena.bin'] or result.output != payload['expected-output.bin']:
        raise ValueError('ported fast execution differs from historical original full arena')
    if any(result.metadata()[key] != recorded['functional'][key] for key in ('program_sha256', 'observed', 'instructions', 'program_words', 'halt_pc')):
        raise ValueError('ported execution program/DMA metadata differs')
    return dict(case=name, full_arena_exact=True, output_exact=True,
        initial_sha256=digest(payload['arena.bin']), expected_arena_sha256=digest(result.arena),
        expected_output_sha256=digest(result.output), execution=result.metadata(), provenance=provenance,
        original_math_executed=False, historical_original_receipt_sha256=digest(payload['receipt']))
