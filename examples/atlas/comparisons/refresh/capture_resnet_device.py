#!/usr/bin/env python3
"""Capture ResNet50 fused device launches from exact staged RTL-model arenas.

Frozen v1 tensors select compile-time scales only. Every v2 graph edge comes
from the new device result, with host byte gathers, maxpool, average, and bias.
"""

import argparse
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from partition_resnet import epilogue_partition


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unrestricted(path):
    if any(token in part.lower() for part in Path(path).parts for token in ('hammer', 'vlsi')):
        raise ValueError('restricted path')


def checked(path, *, symlinks=True):
    """Resolve each symlink only after admitting its target's path components."""
    path = Path(path).absolute()
    unrestricted(path)
    pending = list(path.parts[1:])
    current = Path(path.anchor)
    links = 0
    while pending:
        part = pending.pop(0)
        if part in ('', '.'):
            continue
        if part == '..':
            current = current.parent
            continue
        candidate = current / part
        if candidate.is_symlink():
            if not symlinks:
                raise ValueError('output symlink')
            links += 1
            if links > 40:
                raise ValueError('source symlink loop')
            target = Path(os.readlink(candidate))
            unrestricted(target)
            if target.is_absolute():
                current = Path(target.anchor)
                pending = list(target.parts[1:]) + pending
            else:
                pending = list(target.parts) + pending
        else:
            current = candidate
    return current


def path_remaps(specifications):
    result = []
    for specification in specifications:
        old, separator, new = specification.partition('=')
        if not separator or not old or not new or not Path(old).is_absolute() or not Path(new).is_absolute():
            raise ValueError('--path-remap requires absolute OLD=NEW roots')
        unrestricted(old)
        unrestricted(new)
        old = Path(old)
        if any(old == previous for previous, _ in result):
            raise ValueError('duplicate --path-remap root')
        result.append((old, Path(new)))
    return sorted(result, key=lambda item: len(item[0].parts), reverse=True)


def source_path(path, manifest=None, remaps=()):
    path = Path(path)
    unrestricted(path)
    if path.is_absolute():
        for old, new in remaps:
            if path.is_relative_to(old):
                path = new / path.relative_to(old)
                break
    elif manifest is not None:
        path = Path(manifest).parent / path
    return checked(path)


def read_source(path, pins, expected=None, manifest=None, remaps=()):
    path = source_path(path, manifest, remaps)
    raw = path.read_bytes()
    digest = sha(raw)
    if expected is not None and digest != expected:
        raise ValueError('source digest differs: ' + str(path))
    if str(path) in pins and pins[str(path)] != digest:
        raise ValueError('source changed: ' + str(path))
    pins[str(path)] = digest
    return raw


def plan_dimensions(plan):
    """Admit the single-image graph geometry supported by this capture driver."""

    def shape(value, rank):
        if len(value) != rank or any(type(x) is not int or x <= 0 for x in value):
            raise ValueError('invalid plan shape')
        return value

    def square_pair(value):
        return len(value) == 2 and all(type(x) is int for x in value) and value[0] == value[1]

    image = shape(plan['image']['shape'], 4)
    if image[0] != 1 or plan['image']['dtype'] != 'float32':
        raise ValueError('capture requires one float32 image')
    nodes = {node['node']: node for node in plan['nodes']}
    if len(nodes) != len(plan['nodes']):
        raise ValueError('duplicate plan node')
    if nodes['x']['output_shape'] != image:
        raise ValueError('image and graph input shapes differ')
    pool = nodes['maxpool']
    pool_shape = shape(pool['output_shape'], 4)
    source_shape = shape(nodes[pool['inputs'][0]]['output_shape'], 4)
    params = pool['host_params']
    kernel, stride, padding = [params[name] for name in ('kernel', 'stride', 'padding')]
    if not all(square_pair(value) for value in (kernel, stride, padding)):
        raise ValueError('capture requires square maxpool geometry')
    if kernel[0] <= 0 or stride[0] <= 0 or not 0 <= padding[0] <= kernel[0] // 2:
        raise ValueError('invalid maxpool geometry')
    if params['dilation'] != [1, 1] or params['ceil_mode']:
        raise ValueError('capture requires undilated floor maxpool')
    expected = [1, source_shape[1], *[(x + 2 * padding[0] - kernel[0]) // stride[0] + 1 for x in source_shape[2:]]]
    if pool_shape != expected:
        raise ValueError('maxpool shape differs from geometry')
    average = shape(nodes['avgpool']['output_shape'], 4)
    average_input = shape(nodes[nodes['avgpool']['inputs'][0]]['output_shape'], 4)
    if average != [1, average_input[1], 1, 1] or nodes['avgpool']['host_params']['output_size'] != [1, 1]:
        raise ValueError('capture requires global average pooling')
    classifier = nodes['fc']
    output = shape(classifier['output_shape'], 2)
    logical = shape(classifier['matrix_mkn'], 3)
    padded = shape(classifier['padded_mkn'], 3)
    if logical != [1, average[1], output[1]] or output[0] != 1:
        raise ValueError('classifier shape differs from pooling')
    for node in nodes.values():
        if node['operator'] not in ('Conv2d', 'Linear'):
            continue
        actual = shape(node['matrix_mkn'], 3)
        aligned = shape(node['padded_mkn'], 3)
        if any(a > p or p % 32 for a, p in zip(actual, aligned)):
            raise ValueError('invalid padded contraction shape')
        if node['operator'] == 'Conv2d':
            geometry = node['geometry']
            square = all(square_pair(geometry[name]) for name in ('kernel_size', 'stride', 'padding'))
            if not square or geometry['dilation'] != [1, 1]:
                raise ValueError('capture requires square undilated convolutions')
            incoming = shape(nodes[node['inputs'][0]]['output_shape'], 4)
            outgoing = shape(node['output_shape'], 4)
            kh, st, pad = [geometry[name][0] for name in ('kernel_size', 'stride', 'padding')]
            if kh <= 0 or st <= 0 or pad < 0:
                raise ValueError('invalid convolution geometry')
            spatial = [(x + 2 * pad - kh) // st + 1 for x in incoming[2:]]
            expected_matrix = [spatial[0] * spatial[1], incoming[1] * kh * kh, outgoing[1]]
            if outgoing != [1, outgoing[1], *spatial] or actual != expected_matrix:
                raise ValueError('convolution shape differs from geometry')
    return dict(
        image=image[1:], maxpool=pool_shape[1:], average_channels=average[1], classes=output[1],
        classifier_padded=padded, pool_geometry=(kernel[0], stride[0], padding[0])
    )


def tiled_fp8(matrix):
    m, k = matrix.shape
    return matrix.reshape(m // 32, 32, k // 32, 32).transpose(0, 2, 1, 3).copy().tobytes()


def bf16_tiles(matrix):
    m, n = matrix.shape
    raw = matrix.contiguous().view(torch.uint16).numpy()
    return b''.join(raw[r:r + 32, c:c + 16].copy().tobytes() + raw[r:r + 32, c + 16:c + 32].copy().tobytes()
                    for r in range(0, m, 32) for c in range(0, n, 32))


def bf16_decode(raw, m, n):
    data = np.frombuffer(raw, dtype='<u2')
    out = np.empty((m, n), np.uint16)
    p = 0
    for r in range(0, m, 32):
        for c in range(0, n, 32):
            out[r:r + 32, c:c + 16] = data[p:p + 512].reshape(32, 16)
            out[r:r + 32, c + 16:c + 32] = data[p + 512:p + 1024].reshape(32, 16)
            p += 1024
    return (out.astype(np.uint32) << 16).view(np.float32)


def byte_gather(raw, desc, launch):
    c, h, w = (desc['channels'], desc['h'], desc['w'])
    m, k = (launch['m'], launch['k'])
    kernel, stride, pad = (launch['kernel'], launch['stride'], launch['padding'])
    oh, ow = ((h + 2 * pad - kernel) // stride + 1, (w + 2 * pad - kernel) // stride + 1)
    source = np.frombuffer(raw, np.uint8).reshape(-1, desc['padded_channels'])[:h * w, :c].reshape(h, w, c)
    result = np.zeros((m, k), np.uint8)
    positions = np.arange(oh * ow)
    for channel in range(c):
        for kr in range(kernel):
            for kc in range(kernel):
                ih, iw = (positions // ow * stride + kr - pad, positions % ow * stride + kc - pad)
                valid = (ih >= 0) & (ih < h) & (iw >= 0) & (iw < w)
                result[positions[valid], (channel * kernel + kr) * kernel + kc] = source[ih[valid], iw[valid], channel]
    return tiled_fp8(result)


def maxpool(raw, desc, geometry):
    h, w, c = (desc['h'], desc['w'], desc['padded_channels'])
    source = np.frombuffer(raw, np.uint8).reshape(-1, c)[:h * w].reshape(h, w, c)
    if np.any(source & 128):
        raise ValueError('byte maxpool requires the fused nonnegative ReLU output')
    kernel, stride, padding = geometry
    oh, ow = ((h + 2 * padding - kernel) // stride + 1, (w + 2 * padding - kernel) // stride + 1)
    result = np.zeros(((oh * ow + 31) // 32 * 32, c), np.uint8)
    for r in range(oh):
        for col in range(ow):
            ih, iw = (r * stride - padding, col * stride - padding)
            patch = source[max(0, ih):min(h, ih + kernel), max(0, iw):min(w, iw + kernel)]
            result[r * ow + col] = patch.max(axis=(0, 1))
    return result.tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--atlas-tools', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--input-image', type=Path, required=True, help='Single f32 NCHW image, bound by the plan hash')
    parser.add_argument('--v1-logits', type=Path, required=True, help='Baseline f32 logits for this image')
    parser.add_argument('--expected-plan-sha256', help='Optional caller-supplied host-plan digest')
    parser.add_argument('--expected-reference-sha256', help='Optional caller-supplied reference digest')
    parser.add_argument('--expected-v1-logits-sha256', help='Optional caller-supplied baseline logits digest')
    parser.add_argument(
        '--path-remap', action='append', default=[], metavar='OLD=NEW',
        help='Remap a legacy absolute payload root; longest component prefix wins'
    )
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.atlas_tools = checked(args.atlas_tools)
    args.model_root = checked(args.model_root)
    sys.path[:0] = [str(args.atlas_tools), str(args.model_root)]
    from atlas_epilogue import Epilogue, Step, TileInput
    from atlas_program import assembly, dma_load_bytes
    from atlas_resnet_input import DirectInput, relocate_assembly
    from atlas_rtl_reference import RtlReference
    from atlas_tile_arithmetic import tensor_scale_exponent
    from npu_model.configs.numerics import RtlNumerics
    from npu_model.util.converter import input_to_program
    out = checked(args.out, symlinks=False)
    out.mkdir(parents=True, exist_ok=args.resume)
    pins = {}
    remaps = path_remaps(args.path_remap)

    def read(path, expected=None, manifest=None):
        return read_source(path, pins, expected, manifest, remaps if manifest is not None else ())

    def write(path, raw):
        path = out / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return str(path)

    args.plan = source_path(args.plan, remaps=remaps)
    args.reference = source_path(args.reference, remaps=remaps)
    args.input_image = source_path(args.input_image, remaps=remaps)
    args.v1_logits = source_path(args.v1_logits, remaps=remaps)
    plan_raw = read(args.plan, args.expected_plan_sha256)
    plan = json.loads(plan_raw)
    dimensions = plan_dimensions(plan)
    old = json.loads(read(args.reference, args.expected_reference_sha256))
    baseline_raw = read(args.v1_logits, args.expected_v1_logits_sha256)
    if len(baseline_raw) != dimensions['classes'] * 4:
        raise ValueError('baseline logits size differs from classifier')
    nodes = plan['nodes']
    by_node = {node['node']: node for node in nodes}
    old_nodes = {node['node']: node for node in old['nodes']}
    fusion = epilogue_partition(nodes)
    old_sites = {item['site']: item for item in old['kernels']}
    position = {node['node']: i for i, node in enumerate(nodes)}
    tensors = {}
    actions = []
    image_raw = read(args.input_image, plan['image']['sha256'])
    if len(image_raw) != int(np.prod(dimensions['image'])) * 4:
        raise ValueError('input image size differs from plan')
    input_path = write(Path('input.f32.bin'), image_raw)
    model = dict(
        schema='atlas.resnet50.device_fp8.v2', arena_base=0x90000000,
        input=dict(path=input_path, shape=dimensions['image'], dtype='f32_nchw', tensor='input'), tensors=[],
        nodes=actions, launches=[], expected_logits=str(out / 'logits.f32.bin'), output_tensor='fc'
    )
    launches = {}
    for fused in fusion:
        node = by_node[fused['node']]
        baseline_index = fused['baseline_index']
        is_conv = node['operator'] == 'Conv2d'
        tail = by_node[fused['output_node']] if is_conv else node
        shape = tail['output_shape']
        channels, h, w = (shape[1], shape[2], shape[3]) if is_conv else (shape[1], 1, 1)
        m, k, n = node['padded_mkn']
        old_site = old_sites[node['target']]
        input_node = {'x': 'input', 'flatten': 'avgpool'}.get(fused['input_node'], fused['input_node'])
        scale = 0
        if is_conv:
            saved = old_nodes[fused['output_node']]
            if 'payload' in saved:
                value = np.frombuffer(
                    read(saved['payload']['path'], saved['payload']['sha256'], args.reference), np.float32
                )
            else:
                payload = old_site['expected_physical_output']
                raw = read(payload['path'], payload['sha256'], args.reference)
                value = bf16_decode(raw, m, n)[:node['matrix_mkn'][0], :channels].T.reshape(-1)
                bn = by_node[fused['bn_node']]['host_params']['scale_then_bias']
                coeff = np.frombuffer(read(bn['path'], bn['sha256'], args.plan), np.float32)
                value = np.add(
                    np.multiply(value.reshape(channels, -1), coeff[:channels, None], dtype=np.float32),
                    coeff[channels:, None], dtype=np.float32
                ).reshape(-1)
                if sha(value.tobytes()) != saved['output']['sha256']:
                    raise ValueError('reconstructed BN scale source differs')
            if not np.isfinite(value).all():
                raise ValueError('nonfinite captured scale source')
            bits = torch.from_numpy(value.copy()).to(torch.bfloat16).view(torch.uint16).numpy()
            scale = tensor_scale_exponent([int((bits & 32767).max())])
        output_id = fused['output_node']
        tensors[output_id] = dict(
            id=output_id, h=h, w=w, channels=channels, padded_channels=n, scale_exponent=scale,
            dtype='fp8_nhwc' if is_conv else 'bf16_tiles', storage_bytes=m * n * (1 if is_conv else 2), physical_rows=m
        )
        geometry = node.get('geometry', {})
        kernel = geometry.get('kernel_size', [1])[0]
        stride = geometry.get('stride', [1])[0]
        padding = geometry.get('padding', [0])[0]
        launch = dict(
            site=node['target'], baseline_index=baseline_index, input_tensor=input_node, output_tensor=output_id, m=m,
            k=k, n=n, logical_m=node['matrix_mkn'][0], logical_k=node['matrix_mkn'][1],
            logical_n=node['matrix_mkn'][2], kernel=kernel, stride=stride, padding=padding, a_offset=0, a_bytes=m * k,
            weight_offset=m * k, b_scale_exponent=old_site['b_scale_exponent'],
            a_scale_exponent=old_site['a_scale_exponent'], output_offset=m * k + k * n + 256,
            output_bytes=tensors[output_id]['storage_bytes'], output_dtype='fp8' if is_conv else 'bf16',
            output_layout='channels_last' if is_conv else 'tiles', output_scale_exponent=scale, dynamic_regions=[],
            arena_regions=[], _fusion=fused, _node=node, _position=position[node['node']]
        )
        launches[fused['node']] = launch
        dependencies = [input_node] + ([fused['residual_node']] if fused['residual_node'] else [])
        actions.append(dict(
            op='launch', _key=fused['node'], _deps=dependencies, _output=output_id, _position=position[node['node']]
        ))
    stem = next(item for item in fusion if item['node'] == 'conv1')['output_node']
    pool_channels, pool_h, pool_w = dimensions['maxpool']
    pool_rows = (pool_h * pool_w + 31) // 32 * 32
    pool_padded = tensors[stem]['padded_channels']
    tensors['maxpool'] = dict(
        id='maxpool', h=pool_h, w=pool_w, channels=pool_channels, padded_channels=pool_padded,
        scale_exponent=tensors[stem]['scale_exponent'], dtype='fp8_nhwc', storage_bytes=pool_rows * pool_padded,
        physical_rows=pool_rows
    )
    average_channels = dimensions['average_channels']
    classes = dimensions['classes']
    tensors['avgpool'] = dict(
        id='avgpool', h=1, w=1, channels=average_channels, padded_channels=average_channels, scale_exponent=0,
        dtype='f32_vector', storage_bytes=average_channels * 4
    )
    tensors['fc'] = dict(
        id='fc', h=1, w=1, channels=classes, padded_channels=classes, scale_exponent=0, dtype='f32_vector',
        storage_bytes=classes * 4
    )
    pool_kernel, pool_stride, pool_padding = dimensions['pool_geometry']
    actions += [
        dict(
            op='maxpool', input=stem, output='maxpool', kernel=pool_kernel, stride=pool_stride, padding=pool_padding,
            _deps=[stem], _output='maxpool', _position=position['maxpool']
        ),
        dict(
            op='avgpool', input=by_node['avgpool']['inputs'][0], output='avgpool',
            _deps=[by_node['avgpool']['inputs'][0]], _output='avgpool', _position=position['avgpool']
        )
    ]
    bias = by_node['fc']['host_bias']
    bias_path = write(Path('fc-bias.f32.bin'), read(bias['path'], bias['sha256'], args.plan))
    if len(read(bias_path)) != classes * 4:
        raise ValueError('classifier bias size differs from plan')
    model['fc_bias'] = bias_path
    actions.append(dict(
        op='fc_bias', input='fc_acc', output='fc', path=bias_path, _deps=['fc_acc'], _output='fc',
        _position=position['fc'] + 1
    ))
    remaining = list(actions)
    actions.clear()
    available = {'input'}
    while remaining:
        ready = sorted(
            (item for item in remaining if all(dep in available for dep in item['_deps'])),
            key=lambda item: item['_position']
        )
        if not ready:
            raise ValueError('fused graph has an unresolved dependency')
        item = ready[0]
        remaining.remove(item)
        available.add(item['_output'])
        if item['op'] == 'launch':
            launch = launches[item['_key']]
            launch['index'] = len(model['launches'])
            model['launches'].append(launch)
            item['launch'] = launch['index']
        actions.append({k: v for k, v in item.items() if not k.startswith('_')})
    model['tensors'] = list(tensors.values())
    model['partition'] = fusion
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        roots = (args.atlas_tools, args.model_root, Path(__file__).parent)
        if name and str(name).endswith('.py') and any(str(name).startswith(str(root) + '/') for root in roots):
            read(name)
    model['source_inputs_sha256'] = pins
    (out / 'partition.json').write_text(json.dumps(fusion, indent=2) + '\n')
    values = {}
    manifest = dict(schema='atlas.resnet50.device_fp8.launches.v2', launches=[])
    engine = None if args.prepare_only else RtlReference(model['arena_base'])
    started = time.monotonic()
    try:
        for action in actions:
            op = action['op']
            if op == 'maxpool':
                if not args.prepare_only:
                    values[action['output']] = maxpool(
                        values[action['input']], tensors[action['input']], dimensions['pool_geometry']
                    )
                continue
            if op == 'avgpool':
                if not args.prepare_only:
                    desc = tensors[action['input']]
                    raw = np.frombuffer(values[action['input']], np.uint8).reshape(-1, desc['padded_channels'])
                    raw = raw[:desc['h'] * desc['w'], :desc['channels']]
                    decoded = RtlNumerics.from_fp8(
                        torch.from_numpy(raw.copy()).view(torch.float8_e4m3fn), 127 + desc['scale_exponent']
                    ).float().numpy()
                    total = np.zeros(desc['channels'], np.float32)
                    for row in decoded:
                        total = np.add(total, row, dtype=np.float32)
                    values[action['output']] = np.divide(total, np.float32(len(decoded)), dtype=np.float32).tobytes()
                continue
            if op == 'fc_bias':
                if not args.prepare_only:
                    raw = values[action['input']]
                    value = bf16_decode(
                        raw, dimensions['classifier_padded'][0], dimensions['classifier_padded'][2]
                    )[0, :classes]
                    logits = np.add(value, np.frombuffer(read(bias_path), np.float32), dtype=np.float32)
                    values['fc'] = logits.tobytes()
                    write(Path('logits.f32.bin'), logits.tobytes())
                continue
            launch = model['launches'][action['launch']]
            fused, node = (launch['_fusion'], launch['_node'])
            index, m, k, n = (launch['index'], launch['m'], launch['k'], launch['n'])
            is_conv = node['operator'] == 'Conv2d'
            directory = Path(f'kernel-{index:02d}')
            weight_id = node['weight']['fp8_tiled']
            weights = read(weight_id['path'], weight_id['sha256'], args.plan)
            if len(weights) != k * n:
                raise ValueError('weight bytes differ from padded contraction')
            launch['weights'] = write(directory / 'weight.bin', weights)
            initial = bytearray(bytes(m * k) + weights + bytes([0xA5]) * 256
                                + bytes([0xCD]) * launch['output_bytes'] + bytes([0x5A]) * 256)
            steps = []

            def region(raw, name):
                initial.extend(bytes(-len(initial) % 64))
                offset = len(initial)
                initial.extend(raw)
                path = write(directory / name, raw)
                launch['arena_regions'].append(dict(offset=offset, path=path))
                return offset
            if is_conv:
                bn = by_node[fused['bn_node']]['host_params']['scale_then_bias']
                coefficients = torch.frombuffer(
                    bytearray(read(bn['path'], bn['sha256'], args.plan)), dtype=torch.float32
                ).to(torch.bfloat16)
                for kind, vector in [('mul', coefficients[:n]), ('add', coefficients[n:])]:
                    offset = region(bf16_tiles(vector.expand(32, n).contiguous()), 'bn-' + kind + '.bin')
                    steps.append(Step(kind, TileInput(offset, 'channel')))
            incoming = tensors.get(launch['input_tensor'])
            direct = None
            if launch['input_tensor'] == 'input':
                launch['input_mode'] = 'stem_f32'
                launch['geometry'] = [*dimensions['image'], launch['kernel'], launch['stride'], launch['padding']]
                if not args.prepare_only:
                    image = np.frombuffer(read(input_path), np.float32).reshape(dimensions['image'])
                    source = torch.from_numpy(image.copy()).to(torch.bfloat16)
                    encoded = RtlNumerics.to_fp8(source, 127 + launch['a_scale_exponent']).view(torch.uint8).numpy()
                    c, h, w = dimensions['image']
                    desc = dict(channels=c, padded_channels=c, h=h, w=w)
                    initial[:m * k] = byte_gather(encoded.transpose(1, 2, 0).copy().tobytes(), desc, launch)
            elif not is_conv:
                launch['input_mode'] = 'fc_f32'
                launch['geometry'] = [average_channels, 1, 1, 1, 1, 0]
                if not args.prepare_only:
                    value = torch.frombuffer(
                        bytearray(values[launch['input_tensor']]), dtype=torch.float32
                    ).to(torch.bfloat16)
                    encoded = RtlNumerics.to_fp8(value, 127 + launch['a_scale_exponent']).view(torch.uint8).numpy()
                    matrix = np.zeros((m, k), np.uint8)
                    matrix[0, :average_channels] = encoded
                    initial[:m * k] = tiled_fp8(matrix)
            elif launch['kernel'] == 1:
                launch['input_mode'] = 'device_gather_fp8'
                source = bytes(incoming['storage_bytes']) if args.prepare_only else values[launch['input_tensor']]
                source_offset = region(source, 'source-placeholder.bin')
                # Dynamic source bytes are overwritten by the live host.
                launch['arena_regions'].pop()
                zero_offset = region(bytes(32), 'zero-line.bin')
                direct = DirectInput(
                    source_offset, incoming['padded_channels'], incoming['h'], incoming['w'], launch['stride'],
                    zero_offset
                )
                launch['direct_input'] = asdict(direct)
                launch['input_source_offset'] = source_offset
                launch['geometry'] = [
                    incoming['channels'], incoming['h'], incoming['w'], launch['kernel'], launch['stride'],
                    launch['padding']
                ]
                launch['a_scale_exponent'] = incoming['scale_exponent']
            else:
                launch['input_mode'] = 'host_gather_fp8'
                launch['geometry'] = [
                    incoming['channels'], incoming['h'], incoming['w'], launch['kernel'], launch['stride'],
                    launch['padding']
                ]
                launch['a_scale_exponent'] = incoming['scale_exponent']
                if not args.prepare_only:
                    initial[:m * k] = byte_gather(values[launch['input_tensor']], incoming, launch)
            launch['input_scale_exponent'] = launch['a_scale_exponent']
            if fused['residual_node']:
                residual_desc = tensors[fused['residual_node']]
                residual = (bytes(residual_desc['storage_bytes']) if args.prepare_only
                            else values[fused['residual_node']])
                offset = region(residual, 'residual-placeholder.bin')
                launch['arena_regions'].pop()
                launch['dynamic_regions'].append(dict(offset=offset, tensor=fused['residual_node']))
                steps.append(Step('add', TileInput(offset, 'channels_last', 'fp8', residual_desc['scale_exponent'])))
            if fused['relu']:
                steps.append(Step('relu'))
            e = Epilogue(
                m, n, launch['output_offset'], tuple(steps), 'fp8', launch['output_scale_exponent'], 'channels_last'
            ) if is_conv else None
            launch['epilogue'] = asdict(e) if e else None
            launch['arena_bytes'] = len(initial)
            source = assembly(
                m, k, n, launch['output_offset'], launch['a_scale_exponent'] + launch['b_scale_exponent'],
                traffic_reuse=True, epilogue=e, direct_input=direct
            )
            source = relocate_assembly(source, model['arena_base'], arena_bytes=len(initial))
            program = input_to_program(io.StringIO(source))
            raw_program = b''.join((word.to_bytes(4, 'little') for word in program.assemble()))
            if len(raw_program) + 20 > 131072:
                raise ValueError('patched program exceeds IMEM')
            launch['source'] = write(directory / 'program.S', source.encode())
            launch['program'] = write(directory / 'program.bin', raw_program)
            launch['initial'] = write(directory / 'initial-dram.bin', bytes(initial))
            launch['expected'] = str(out / directory / 'expected.bin')
            launch['requested_dma'] = dict(
                load_bytes=dma_load_bytes(m, k, n, traffic_reuse=True, epilogue=e, direct_input=direct),
                store_bytes=launch['output_bytes']
            )
            launch['program_words'] = len(raw_program) // 4
            if not args.prepare_only:
                expected, stats = engine.run(
                    source, bytes(initial), launch['output_offset'], launch['output_bytes'], launch['requested_dma'],
                    fp8_scale=launch['output_scale_exponent'] if is_conv else None
                )
                write(directory / 'expected.bin', expected)
                values[launch['output_tensor']] = expected
                launch['reference'] = stats
                print(
                    json.dumps(dict(index=index, site=launch['site'], seconds=time.monotonic() - started, **stats)),
                    flush=True
                )
            manifest['launches'].append(dict(
                name=launch['site'], source=str(directory / 'program.S'), program=str(directory / 'program.bin'),
                input=str(directory / 'initial-dram.bin'), expected=str(directory / 'expected.bin'),
                arena_base=model['arena_base'], output_offset=launch['output_offset'],
                requested_dma=launch['requested_dma']
            ))
            model['arena_bytes'] = max(model.get('arena_bytes', 0), len(initial))
            public = {key: val for key, val in model.items()}
            public['launches'] = [{
                key: val for key, val in item.items() if not key.startswith('_')
            } for item in model['launches']]
            public['status'] = 'PREPARED_NOT_EXECUTED' if args.prepare_only else 'CAPTURING'
            (out / 'model.json').write_text(json.dumps(public, indent=2) + '\n')
        if not args.prepare_only:
            baseline = np.frombuffer(baseline_raw, np.float32)
            current = np.frombuffer(values['fc'], np.float32)
            model['comparison_v1'] = dict(
                images=1, v1_top1=int(baseline.argmax()), v2_top1=int(current.argmax()),
                max_abs_logit_error=float(np.abs(current - baseline).max()),
                l2_logit_error=float(np.linalg.norm(current.astype(np.float64) - baseline)),
                changed_logits=int((current.view(np.uint32) != baseline.view(np.uint32)).sum()),
                v1_logits_sha256=sha(baseline.tobytes()), v2_logits_sha256=sha(current.tobytes())
            )
        for name, digest in pins.items():
            if sha(checked(name).read_bytes()) != digest:
                raise ValueError('capture source bookend changed')
        model['source_bookends_equal'] = True
        model['status'] = 'PREPARED_NOT_EXECUTED' if args.prepare_only else 'CAPTURED_RTL_FUNCTIONAL'
        model['launches'] = [{
            key: val for key, val in item.items() if not key.startswith('_')
        } for item in model['launches']]
        model['command'] = [sys.executable, *sys.argv]
        model['reference_mode'] = ('npu_model physical instructions, numerics=rtl, exact staged arenas; '
                                   'final timed QEMU gate required')
        (out / 'model.json').write_text(json.dumps(model, indent=2) + '\n')
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    finally:
        if engine:
            engine.close()


if __name__ == '__main__':
    main()
