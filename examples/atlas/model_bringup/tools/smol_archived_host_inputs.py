"""Admit unchanged Smol host operands from the exact delivered immutable ELF.

This explicit provider preserves compiled host input identity after the machine
move. It does not claim fresh checkpoint, Torch, or observation-archive access.
"""
import hashlib
import json
import struct

import numpy as np
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked


DELIVERED_SUMS_SHA256 = '901451eb57dc313deb1931974d80d01d5f28be6e97e9e2cd5f89a0c09d62706d'


def require(value, message):
    if not value:
        raise ValueError(message)


def admit_archived_host_inputs(root, read):
    """Admit every immutable archived operand against its exact ELF symbol."""
    from elf_image import Image
    from smol_full_contract import parameter_geometry, readonly_span
    root = checked(root)
    sums_raw = read(root/'SHA256SUMS', DELIVERED_SUMS_SHA256)
    sums = {}
    from bundle import relative_name
    for line in sums_raw.decode().splitlines():
        digest, name = line.split(None, 1)
        name = name.lstrip('*')
        relative_name(name)
        require(name not in sums and len(digest) == 64, 'Archived host checksum inventory')
        sums[name] = digest

    def member(name):
        relative_name(name)
        require(name in sums, 'Unlisted archived host input '+name)
        return read(root/name, sums[name])

    provenance = json.loads(member('provenance.json'))
    require(provenance['schema'] == 'atlas.full_bundle.v1' and provenance['workload'] == 'smol-full',
            'Delivered Smol host provider schema')
    contract_raw = member(provenance['contract'])
    require(hashlib.sha256(contract_raw).hexdigest() == provenance['contract_sha256'], 'Archived contract binding')
    contract = json.loads(contract_raw)
    build = json.loads(member('host/build.json'))
    require(build['actual_returncode'] == 0 and build['source_bookends_verified'] is True
            and build['contract_sha256'] == provenance['contract_sha256'], 'Archived host build binding')
    elf_raw = member('host/host.elf')
    require(hashlib.sha256(elf_raw).hexdigest() == build['elf']['sha256'], 'Archived ELF identity')
    image = Image(elf_raw)
    values = {}
    for name, item in contract['host_parameters'].items():
        raw = member(item['file'])
        require(len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'],
                'Archived host parameter identity '+name)
        symbol = image.symbols[item['symbol']]
        require(symbol['size'] == len(raw), 'Archived host symbol extent '+name)
        readonly_span(image, symbol['address'], len(raw))
        require(image.bytes_at(symbol['address'], len(raw)) == raw, 'Archived embedded operand differs '+name)
        value = np.frombuffer(raw, np.dtype(item['dtype'])).copy().reshape(item['shape'])
        if value.dtype == np.uint16:
            require(not np.any((value & 0x7f80) == 0x7f80), 'Nonfinite archived BF16 host input '+name)
        elif value.dtype.kind == 'f':
            require(np.isfinite(value).all(), 'Nonfinite archived FP32 host input '+name)
        values[name] = value

    reference_raw = member(contract['source_reference'])
    require(hashlib.sha256(reference_raw).hexdigest() == contract['source_reference_sha256'],
            'Archived host source-reference binding')
    shapes = parameter_geometry(json.loads(reference_raw))
    require(set(values) == set(shapes), 'Complete archived host parameter inventory')
    for name, (shape, dtype) in shapes.items():
        require(list(values[name].shape) == shape and values[name].dtype == np.dtype(dtype),
                'Archived host input geometry '+name)

    observation = image.symbols['smol_full_driver_observation']
    require(observation['size'] == 56, 'Archived observation ABI')
    readonly_span(image, observation['address'], 56)
    pointers = struct.unpack('<7Q', image.bytes_at(observation['address'], 56))
    for pointer, name in zip(pointers[:6], ('images', 'state', 'noise', 'image_masks', 'language_mask', 'language_tokens')):
        require(pointer == image.symbols[contract['host_parameters']['observation.'+name]['symbol']]['address'],
                'Archived observation descriptor pointer '+name)
    vocabulary = pointers[-1]
    tokens = values['observation.language_tokens']
    require(0 < vocabulary and np.all(tokens >= 0) and np.all(tokens < vocabulary), 'Archived language token domain')
    receipt = dict(provider='EXACT_DELIVERED_COMPILED_HOST_INPUTS', bundle=str(root),
                   checksums_sha256=DELIVERED_SUMS_SHA256, elf_sha256=build['elf']['sha256'],
                   host_parameters=len(values), all_embedded_parameters_exact=True,
                   checkpoint_file_reopened=False, observation_archive_reopened=False,
                   torch_executed=False, historical_source_weights=contract['source_weights'])
    return values, vocabulary, receipt


def load_host_inputs(root, replay, replay_root, trajectory, read):
    """Bind admitted archived operands to the actual regenerated capture."""
    from bundle import relative_name
    values, vocabulary, receipt = admit_archived_host_inputs(root, read)
    source = receipt['historical_source_weights']
    require(source['sha256'] == replay['source']['weights']['sha256']
            and source['bytes'] == replay['source']['weights']['bytes'],
            'Archived checkpoint identity differs from capture provenance')
    for module, item in replay['parameters'].items():
        if 'bias' in item:
            row = item['bias']
            name = 'bias.'+module
            relative_name(row['file'])
            require(name in values and values[name].tobytes() == read(checked(replay_root/row['file']), row['sha256']),
                    'Archived bias differs from current captured parameter '+module)
    times = np.array([trajectory[f'step_{step:02d}_timestep'][0] for step in range(10)], np.float32)
    require(times.tobytes() == values['timestep'].tobytes(), 'Archived timestep differs from regenerated trajectory')
    calls = [row for row in replay['calls'] if row['module'] == 'action_time_mlp_in']
    require(len(calls) == 10, 'Regenerated timestep call inventory')
    for step, call in enumerate(calls):
        row = call['input']
        relative_name(row['file'])
        raw = read(checked(replay_root/row['file']), row['sha256'])
        actual = np.frombuffer(raw, '<f4').reshape(row['shape']).reshape(50, 1440)[:, 720:]
        expected = np.broadcast_to(values['time.embedding'][step], (50, 720)).copy()
        require(actual.tobytes() == expected.tobytes(), 'Archived timestep embedding differs from actual new predecessor')
    expected_biases = {'bias.'+module for module, item in replay['parameters'].items() if 'bias' in item}
    require({name for name in values if name.startswith('bias.')} == expected_biases, 'Current captured bias inventory')
    return values, vocabulary, receipt
