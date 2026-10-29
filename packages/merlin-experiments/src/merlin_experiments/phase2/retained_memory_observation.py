"""Operator-only same-ELF output export using existing HTIF symbol supplements.

No simulator, qualification or fit authority is issued here. The original public
native product reopener owns source/ELF custody. Only one bounded dense integer
output is supported initially; unknown layouts refuse rather than export gaps.
The supplemental ELF has no load segments, executable sections or guest bytes.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

MAX_BYTES = 64 << 10


def pin(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError('observation input must be an ordinary canonical file')
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def signature_symbols(base, size):
    """A fixed ELF64 little-endian symbol-only image, never a replacement payload."""
    if (type(base) is not int or type(size) is not int or base < 0
            or not 0 < size <= MAX_BYTES or base + size >= 1 << 64):
        raise ValueError('signature export exceeds its bounded physical region')
    strings = b'\0begin_signature\0end_signature\0'
    names = b'\0.strtab\0.symtab\0.shstrtab\0'
    symbols = bytes(24) + struct.pack('<IBBHQQ', 1, 0x10, 0, 0xfff1, base, 0)
    symbols += struct.pack('<IBBHQQ', 17, 0x10, 0, 0xfff1, base + size, 0)
    # No program table means no PT_LOAD and no guest-memory content at all.
    stringoff, symboloff = 64, 64 + len(strings)
    nameoff = symboloff + len(symbols)
    shoff = nameoff + len(names)
    header = b'\x7fELF\x02\x01\x01' + bytes(9)
    header += struct.pack('<HHIQQQIHHHHHH', 2, 243, 1, 0, 0, shoff, 0, 64, 56, 0, 64, 4, 3)
    sections = bytes(64)
    sections += struct.pack('<IIQQQQIIQQ', 1, 3, 0, 0, stringoff, len(strings), 0, 0, 1, 0)
    sections += struct.pack('<IIQQQQIIQQ', 9, 2, 0, 0, symboloff, len(symbols), 1, 1, 1, 24)
    sections += struct.pack('<IIQQQQIIQQ', 17, 3, 0, 0, nameoff, len(names), 0, 0, 1, 0)
    return header + strings + symbols + names + sections


def signature_bytes(raw, size):
    """Decode exact granularity-one HTIF output, with no truncation or extra bytes."""
    if type(size) is not int or not 0 < size <= MAX_BYTES or len(raw) != size * 3:
        raise ValueError('signature output has a different byte count')
    lines = raw.splitlines()
    if len(lines) != size or any(len(line) != 2 or any(c not in b'0123456789abcdef' for c in line) for line in lines):
        raise ValueError('signature output is not complete granularity-one bytes')
    return bytes(int(line, 16) for line in lines)


def _flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from _flatten(item)
    elif type(value) is int:
        yield value
    else:
        raise ValueError('only exact integer output values are supported')


def output_region(elf_bytes, cb):
    """Reproduce the native dense caller roster and independently validate ELF storage."""
    from merlin.runtime.harness_render import _roster, _tensor
    from ..phase1.feedback.native_output_readback import _elf_output_symbols

    if elf_bytes[16:18] != b'\x02\x00':
        raise ValueError('export requires fixed-address ET_EXEC')
    normalized, slots = _roster(cb)
    outputs = normalized['kernel_abi']['outputs']
    if len(outputs) != 1:
        raise ValueError('prototype requires exactly one output, without inter-object gaps')
    name = outputs[0]
    spec = normalized['tensors'][name]
    shape, dtype, extent = _tensor(spec)
    if spec['dtype'] not in {'i8', 'i16', 'i32', 'i64'} or not 0 < extent <= MAX_BYTES:
        raise ValueError('unsupported or excessive output storage')
    symbol = f'tensor_{list(slots).index(name)}'
    addresses, _ = _elf_output_symbols(elf_bytes, {symbol: extent})
    # Existing signature symbols could silently defeat the supplement through
    # std::map::merge. Refuse either, even if only one is present or malformed.
    if b'begin_signature\0' in elf_bytes or b'end_signature\0' in elf_bytes:
        raise ValueError('original ELF already declares signature metadata')
    expected = cb.get('_oracle_expected_outputs', {})
    if set(expected) != {name}:
        raise ValueError('original effective buffer omits the exact output oracle')
    values = list(_flatten(expected[name]))
    width = dtype.element_bits // 8
    if len(values) != math.prod(shape) or len(values) * width != extent:
        raise ValueError('output oracle or dense physical extent differs')
    try:
        data = b''.join(value.to_bytes(width, 'little', signed=True) for value in values)
    except OverflowError as error:
        raise ValueError('output oracle exceeds its declared integer type') from error
    return {'tensor': name, 'symbol': symbol, 'base': addresses[symbol], 'bytes': extent,
            'shape': list(shape), 'dtype': spec['dtype'], 'elements': len(values)}, data


def reopen(result_path):
    """Retained native custody plus original dense numerical contract; no new verdict."""
    from ..phase1.feedback import tier_boundary_join as J
    from ..phase1.feedback.candidate_execution_products import reopen_operator_tier_products
    from merlin.perf.whole_model_gsim import read_dump

    result_path = Path(result_path)
    result_pin = pin(result_path)
    boundary = J.observe(result_path, tier='L3')
    grade = json.loads(result_path.read_bytes())
    if grade['numeric'].get('policy') != 'exact_int':
        raise ValueError('prototype requires original exact integer numerics')
    generated = result_path.parent / 'generated'
    products, custody = reopen_operator_tier_products(generated, grade, tier='L3')
    elf, cb_path = products['elf'], products['command_buffer']
    cb = json.loads(cb_path.read_bytes())
    region, expected = output_region(elf.read_bytes(), cb)
    matches = []
    # Bound discovery to direct native readback owners, not any candidate tree.
    children = list(generated.iterdir())
    if len(children) > 1024:
        raise ValueError('native output-owner roster exceeds budget')
    for child in children:
        if not child.name.startswith('neutral_readback_') or child.is_symlink() or not child.is_dir():
            continue
        saved = child / 'command_buffer.json'
        if saved.is_file() and saved.stat().st_size <= 16 << 20 and json.loads(saved.read_bytes()) == cb:
            matches.append(child)
    if len(matches) != 1:
        raise ValueError('original dense memory dump owner absent or ambiguous')
    dump = matches[0] / 'output.dump'
    dump_pin = pin(dump)
    if dump.stat().st_size != 64 + region['bytes']:
        raise ValueError('original coherent dump extent differs')
    parsed = read_dump(dump, [(region['base'], region['bytes'])])
    if parsed['source'] != 'coherent' or parsed['regions'][(region['base'],region['bytes'])] != expected:
        raise ValueError('retained coherent values differ from original numerical contract')
    if pin(result_path) != result_pin or pin(dump) != dump_pin:
        raise ValueError('retained numerical evidence changed while reopening')
    return {'result': result_pin, 'boundary': boundary, 'custody': custody,
            'elf': pin(elf), 'caller': pin(products['caller']), 'command_buffer': pin(cb_path),
            'coherent_dump': dump_pin, 'output': region,
            'physical_cycles': grade['tiers']['L3']['cycles'],
            'physical_engine': grade['tiers']['L3']['sim_provenance'],
            'output_sha256': hashlib.sha256(expected).hexdigest(),
            'guards': {'status': 'not_declared', 'claim': 'all declared output bytes; no invented guard coverage'}}, expected


def observe(result_path, *, runtime, bwrap_pin, directory, tool_pins=(), remaining_s=120):
    """Bounded operator prototype: unchanged executable, private exact output export.

    The namespace contains only the numerically admitted ELF, two generated
    signature symbols, scope metadata, and frozen simulator leaves. Golden and
    physical output bytes are compared by the parent and never mounted. Uses the
    existing census guardian/cleanup owner and auxiliary shared CPU leases.
    """
    import os
    import shutil
    import time
    from dataclasses import replace
    from merlin.common.cpu_pool import AUXILIARY, cpu_priority
    from merlin.runtime import _native_files as F
    from ..phase1 import annotated_spike_evidence as E
    from . import archive_count_observation as O, census_launch as BOX
    from . import _counter_symbol_aliases as A
    from .archive_spike_runtime import ArchiveSpikeRuntime
    from .contracts import document_sha256, write_json

    if type(runtime) is not ArchiveSpikeRuntime or not 0 < remaining_s <= 120:
        raise ValueError('requires frozen runtime and bounded operator observation')
    runtime.verify()
    before, expected = reopen(result_path)
    model = json.loads(runtime.model_json)
    if before['physical_engine']['sha256'] != model['domain']['label_engine_sha256']:
        raise ValueError('physical and model hardware engine domains differ')
    bwrap = Path(bwrap_pin['path'])
    if pin(bwrap) != bwrap_pin:
        raise ValueError('operator namespace launcher changed')
    tools = tuple(Path(row['path']) for row in tool_pins)
    if any(pin(path) != row for path, row in zip(tools, tool_pins, strict=True)):
        raise ValueError('selected runtime helper changed')
    directory = Path(directory)
    if directory.resolve() != directory or directory.exists() or any(p.is_symlink() for p in directory.parents):
        raise ValueError('observation requires a fresh canonical private destination')
    directory.mkdir(mode=0o700, parents=True)
    workspace = directory / 'workspace'; workspace.mkdir(mode=0o700)
    elf = workspace / 'program.elf'; shutil.copyfile(before['elf']['path'], elf)
    if pin(elf)['sha256'] != before['elf']['sha256']:
        raise ValueError('copied original executable changed')
    supplement = workspace / 'output-symbols.elf'
    supplement.write_bytes(signature_symbols(before['output']['base'], before['output']['bytes']))
    scope = workspace / 'scope.json'
    prefix = [str(bwrap), '--unshare-pid', '--unshare-net', '--die-with-parent', '--new-session',
              '--clearenv', '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
              '--bind', str(workspace), str(workspace), '--chdir', str(workspace)]
    for path in (*runtime.execution_files, *tools):
        prefix += ['--ro-bind', str(path), str(path)]
        stem, sep, version = path.name.partition('.so.')
        if sep and '.' in version:
            alias = path.with_name(stem + sep + version.split('.')[0])
            prefix += ['--ro-bind', str(path), str(alias)]
    # The selected dynamic loader is already frozen in runtime dependencies.
    loaders = [p for p in runtime.dependencies if p.name == 'ld-linux-x86-64.so.2']
    if len(loaders) != 1:
        raise ValueError('prototype requires one selected native dynamic loader')
    prefix += ['--ro-bind', str(loaders[0]), '/lib64/ld-linux-x86-64.so.2']
    lookup = ':'.join(dict.fromkeys(str(p.parent) for p in (*runtime.libraries, *runtime.dependencies, *tools)))
    prefix += ['--setenv', 'LD_LIBRARY_PATH', lookup, '--setenv', 'PATH',
               ':'.join(dict.fromkeys(str(path.parent) for path in tools))]
    prefix, backing = BOX._private_backing(tuple(prefix), directory)
    prefix = (*prefix, '--ro-bind', str(elf), str(elf), '--ro-bind', str(supplement), str(supplement))
    limits = replace(runtime.observation_limits(remaining_s), max_line_bytes=1 << 20, max_output_bytes=16 << 20)
    dependencies = tuple(Path(p) for p, _ in runtime.files) + (Path(__file__).resolve(), bwrap)
    ownership = {'closed': True}; started = time.monotonic()
    def run(argv, name, done):
        seconds = remaining_s - (time.monotonic() - started)
        if seconds < 1:
            raise ValueError('operator observation deadline exhausted')
        with cpu_priority(AUXILIARY):
            return O._run(argv, prefix=prefix, directory=directory/name, workspace=workspace,
                backing=backing, limits=replace(limits, wall_s=seconds), dependencies=dependencies,
                require_done=done, ownership=ownership)
    reviewed, actual, listing = run([str(runtime.disassembler), '-d', '--no-show-raw-insn', str(elf)], 'review', False)
    if actual.get('returncode') != 0:
        raise ValueError('boxed exact executable disassembly failed')
    window, schema = A.window(E.parse_disassembly(listing.decode()), elf_bytes=elf.read_bytes(),
        counter_asm=json.loads(runtime.timing_json)['counter_asm'], entry_symbol=runtime.entry_symbol,
        completion_symbol=runtime.completion_symbol)
    review = {'schema': schema, 'elf_sha256': before['elf']['sha256'], **window,
              'disassembly_sha256': hashlib.sha256(listing).hexdigest()}
    write_json(directory/'counter-review.json', review)
    binding = {'schema':'merlin.spike_scope_binding.v1', 'elf_sha256':before['elf']['sha256'],
        'input_sha256':before['elf']['sha256'], 'scope_sha256':model['domain']['scope_sha256'],
        'begin_pc':window['begin_pc'], 'end_pc':window['end_pc'], 'occurrence':window['occurrence'],
        'evidence_sha256':document_sha256(review)}
    write_json(scope, binding)
    prefix = (*prefix, '--ro-bind', str(scope), str(scope))
    report_path = workspace/'report.json'
    commands = runtime.commands(elf, report_path, scope)
    consoles, records = [], []
    for index, argv in enumerate(commands):
        signature = workspace / f'output-{index}.signature'
        # HTIF options precede the primary ELF; unlike payload,
        # symbol-elf is read through nop_memif and cannot modify guest memory.
        invocation, actual, console = run([*argv[:-1], '+symbol-elf='+str(supplement),
            '+signature='+str(signature), '+signature-granularity=1', argv[-1]], f'run-{index}', True)
        if actual.get('returncode') != 0 or signature_bytes(signature.read_bytes(), len(expected)) != expected:
            raise ValueError('same-ELF Spike output differs from original complete numerical contract')
        consoles.append(console)
        records.append({'invocation':pin(invocation),'signature':pin(signature),
                        'complete_output_match':True,'bytes':len(expected)})
    if consoles[0] != consoles[1]:
        raise ValueError('annotated observation changed the functional console')
    report = json.loads(report_path.read_bytes())
    if (report.get('scope_binding') != binding or report.get('model_sha256') != pin(runtime.model_path)['sha256']
            or report.get('loaded_library_sha256') != model['libraries']):
        raise ValueError('model report changed selected executable scope or runtime')
    after, current_expected = reopen(result_path)
    runtime.verify()
    if any(pin(path) != row for path, row in zip(tools, tool_pins, strict=True)):
        raise ValueError('runtime helper changed during observation')
    if before != after or expected != current_expected or pin(elf)['sha256'] != before['elf']['sha256']:
        raise ValueError('original evidence changed during observation')
    cycles = report.get('estimated_cycles') if report.get('timing_complete') is True else None
    value = {'schema':'operator.retained_same_elf_observation.v1','original':before,
        'runtime_sha256':runtime.sha256,'model':pin(runtime.model_path),'observer':pin(runtime.observer),
        'namespace_launcher':bwrap_pin,'runtime_helpers':list(tool_pins),'transport':'existing_htif_symbol_elf_signature',
        'symbol_supplement':pin(supplement),'scope_binding':binding,'report':pin(report_path),
        'runs':records,'functional_console_equal':True,'physical_cycles':before['physical_cycles'],
        'estimated_cycles':cycles,'relative_error':None if cycles is None else cycles/before['physical_cycles']-1,
        'timing_complete':report.get('timing_complete'),'unresolved':report.get('unresolved'),
        'physical_runs':0,'calibration_eligible':False,'qualification':'provisional diagnostic only',
        'elapsed_s':time.monotonic()-started,'implementation':pin(Path(__file__).resolve())}
    write_json(directory/'receipt.json',value)
    return value
