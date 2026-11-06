"""Fixed captured SmolVLA policy graph and existing128-row kernel panels."""
VISION = 'vlm_with_expert.vlm.model.vision_model.'
TEXT = 'vlm_with_expert.vlm.model.text_model.'
EXPERT = 'vlm_with_expert.lm_expert.'
ROLES = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
         'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')


def calls():
    result = []

    def add(module, shape, dtype, site, layer=0, role=0, image=0):
        rows, k, n = shape
        result.append(dict(index=len(result), module=module, rows=rows, input_width=k,
            output_width=n, input_dtype=dtype, output_dtype=dtype, site=site, layer=layer,
            role=role, image=image))

    for image in range(2):
        add(VISION+'embeddings.patch_embedding', (1024, 768, 768), 0, 7, image=image)
        for layer in range(12):
            prefix = VISION+f'encoder.layers.{layer}.'
            for role, name in enumerate(('q_proj', 'k_proj', 'v_proj', 'out_proj'), 1):
                add(prefix+'self_attn.'+name, (1024, 768, 768), 0, 7, layer, role, image)
            add(prefix+'mlp.fc1', (1024, 768, 3072), 0, 7, layer, 5, image)
            add(prefix+'mlp.fc2', (1024, 3072, 768), 0, 7, layer, 6, image)
        add('vlm_with_expert.vlm.model.connector.modality_projection.proj',
            (64, 12288, 960), 0, 7, 12, 7, image)
    add('state_proj', (1, 32, 960), 1, 0)
    for layer in range(16):
        widths = ((960, 960), (960, 320), (960, 320), (960, 960), (960, 2560), (960, 2560), (2560, 960))
        for role, (k, n) in enumerate(widths):
            add(TEXT+f'layers.{layer}.'+ROLES[role], (177, k, n), 0, 5, layer, role)
    for step in range(10):
        begin = len(result)
        add('action_in_proj', (50, 32, 720), 1, 1)
        add('action_time_mlp_in', (50, 1440, 720), 1, 2)
        add('action_time_mlp_out', (50, 720, 720), 1, 3)
        for layer in range(16):
            widths = ((50, 720, 960), (177, 320, 320) if layer % 2 else (50, 720, 320),
                (177, 320, 320) if layer % 2 else (50, 720, 320), (50, 960, 720),
                (50, 720, 2048), (50, 720, 2048), (50, 2048, 720))
            for role, shape in enumerate(widths):
                add(EXPERT+f'layers.{layer}.'+ROLES[role], shape,
                    1 if layer % 2 and role in (1, 2) else 0, 6, layer, role)
        add('action_out_proj', (50, 720, 32), 1, 4)
        for row in result[begin:]: row['step'] = step
    return result


def panels():
    result = []
    for call in calls():
        for start in range(0, call['rows'], 128):
            rows = min(128, call['rows']-start)
            k, n = call['input_width'], call['output_width']
            result.append(dict(index=len(result), call=call['index'], module=call['module'],
                logical_row_start=start, logical_mkn=[rows, k, n],
                physical_mkn=[(rows+31)//32*32, (k+31)//32*32, (n+31)//32*32]))
    return result


def reference_geometry():
    result = {
        'prefix_embeddings': ([1, 177, 960], 'float32'),
        'prefix_pad_mask': ([1, 177], 'bool'),
        'prefix_attention_mask': ([1, 177], 'bool'),
        'prefix_key_cache': ([16, 1, 177, 5, 64], 'uint16'),
        'prefix_value_cache': ([16, 1, 177, 5, 64], 'uint16'),
        'final_full_state': ([1, 50, 32], 'float32'),
        'final_actions': ([1, 50, 6], 'float32'),
    }
    for step in range(10):
        for field in ('input_state', 'velocity', 'output_state'):
            result[f'step_{step:02d}_{field}'] = ([1, 50, 32], 'float32')
        result[f'step_{step:02d}_timestep'] = ([1], 'float32')
    return result


def validate_reference(replay, manifest):
    """Admit this exact architecture's complete ordered reference, not a subset."""
    if (replay.get('status') != 'PASS_ATLAS_ARITHMETIC_FULL_POLICY_REPLAY'
        or replay.get('fresh_vision_executed') is not True
        or replay.get('input_state_rng_bookends_equal') is not True
        or manifest.get('schema') != 'atlas.full_kernel_assets.v1'
        or manifest.get('workload') != 'smol-full'
        or manifest.get('source_bookends_equal') is not True
        or manifest.get('finite_trajectory_verified') is not True):
        raise ValueError('Require complete finite full-policy arithmetic reference')
    expected_calls, expected_panels = calls(), panels()
    fusion_set = manifest.get('fusion_set', 'none')
    if fusion_set not in ('none', 'vision-bias', 'vision-bias-gelu', 'fc1-bias-gelu', 'connector',
                          'vision-bias-gelu-connector', 'fc1-bias-gelu-fp8'):
        raise ValueError('Unsupported explicit Smol fusion profile')

    def fusion_operations(call):
        if fusion_set == 'none' or call['site'] != 7: return []
        if call['role'] == 7:
            return ['connector_scale'] if fusion_set in ('connector', 'vision-bias-gelu-connector') else []
        if fusion_set in ('fc1-bias-gelu', 'fc1-bias-gelu-fp8'):
            return ['bias', 'gelu_tanh'] if call['role'] == 5 else []
        if fusion_set == 'connector': return []
        return ['bias', 'gelu_tanh'] if call['role'] == 5 and fusion_set != 'vision-bias' else ['bias']
    actual_calls, actual_panels = replay.get('calls'), manifest.get('kernels')
    if (not isinstance(actual_calls, list) or len(actual_calls) != len(expected_calls)
        or not isinstance(actual_panels, list) or len(actual_panels) != len(expected_panels)
        or manifest.get('kernel_count') != len(expected_panels)):
        raise ValueError('Incomplete full-policy call/panel count')
    for expected, actual in zip(expected_calls, actual_calls):
        rows, k, n = expected['rows'], expected['input_width'], expected['output_width']
        input_shape, output_shape = ([1, k], [1, n]) if expected['site'] == 0 else ([1, rows, k], [1, rows, n])
        if expected['site'] == 7 and expected['role'] == 0:
            input_shape, output_shape = [1, 3, 512, 512], [1, 768, 32, 32]
        dtype = 'torch.bfloat16' if expected['input_dtype'] == 0 else 'torch.float32'
        if (type(actual.get('index')) is not int or actual['index'] != expected['index']
            or actual.get('module') != expected['module']
            or actual['input'].get('shape') != input_shape or actual['output'].get('shape') != output_shape
            or actual['input'].get('dtype') != dtype or actual['output'].get('dtype') != dtype):
            raise ValueError('Full-policy native call order/dtype/geometry differs')
        if fusion_set == 'none' and (actual.get('input_fp8') or actual.get('output_fp8') or actual.get('fused_operations')):
            raise ValueError('Typed or fused call requires an explicit fusion profile')
        if fusion_set != 'none':
            if actual.get('fused_operations', []) != fusion_operations(expected):
                raise ValueError('Full-policy call fusion operations differ from explicit profile')
            for side, width, wanted in (('input', k, expected['site'] == 7 and expected['role'] == 6),
                                       ('output', n, expected['site'] == 7 and expected['role'] == 5)):
                wanted = wanted and fusion_set == 'fc1-bias-gelu-fp8'
                typed = actual.get(side+'_fp8')
                if bool(typed) != wanted:
                    raise ValueError('Full-policy typed FP8 view differs from explicit profile')
                if wanted and (typed.get('dtype') != 'uint8' or typed.get('shape') != [rows//32, width//32, 32, 32]
                    or typed.get('bytes') != rows*width or actual.get(side+'_source_format') != 'fp8_tiles'
                    or not isinstance(typed.get('sha256'), str) or len(typed['sha256']) != 64
                    or any(character not in '0123456789abcdef' for character in typed['sha256'])
                    or actual.get(side+'_diagnostic_only') is not True
                    or type(actual.get(side+'_decode_exponent')) is not int
                    or not -126 <= actual[side+'_decode_exponent'] <= 127):
                    raise ValueError('Full-policy FP8 extent/layout/scale or diagnostic marker differs')
            if actual.get('input_fp8'):
                previous = actual_calls[expected['index']-1]
                if (previous.get('output_fp8', {}).get('sha256') != actual['input_fp8'].get('sha256')
                    or previous.get('output_decode_exponent') != actual['input_decode_exponent']
                    or actual.get('a_scale') != actual['input_decode_exponent']):
                    raise ValueError('Full-policy FP8 producer/consumer bytes or fixed scale differ')
    import kernel_library  # noqa: E402,F401
    from atlas_program import dma_load_bytes
    scales = {}
    for expected, actual in zip(expected_panels, actual_panels):
        if any(type(actual.get(name)) is not type(value) or actual.get(name) != value
               or isinstance(value, list) and any(type(item) is not int for item in actual[name])
               for name, value in expected.items()):
            raise ValueError('Full-policy panel order or zero-padding geometry differs')
        for key in ('a_scale', 'b_scale', 'output_rescale_exponent'):
            if type(actual.get(key)) is not int or not -126 <= actual[key] <= 127:
                raise ValueError('Full-policy finite scale range differs')
        if actual['output_rescale_exponent'] != actual['a_scale'] + actual['b_scale']:
            raise ValueError('Full-policy rescale differs from global input/weight scales')
        if actual['call'] in scales and scales[actual['call']] != (actual['a_scale'], actual['b_scale']):
            raise ValueError('Full-policy operator scale changes between row panels')
        scales[actual['call']] = actual['a_scale'], actual['b_scale']
        m, k, n = actual['physical_mkn']
        loads = (m*k*(n//32)+k*n if k*n <= 262144 else m*k+k*n*(m//32)
                 if m*k <= 262144 else (m//32)*(n//32)*(2*k//32*1024))
        if actual.get('recipe') == 'traffic-reuse-v1': loads = dma_load_bytes(m, k, n, traffic_reuse=True)
        operations = fusion_operations(expected_calls[actual['call']])
        if operations:
            meta = actual.get('fusion', {})
            fp8 = fusion_set == 'fc1-bias-gelu-fp8'
            output_bytes = m*n if fp8 else 2*m*n
            extra_bytes = n*64 if 'bias' in operations else 0
            extra_offset = actual['output_offset'] + output_bytes + 256
            total = extra_offset + (extra_bytes+256 if extra_bytes else 0)
            if (meta.get('operations') != operations or meta.get('host_ops_on_success') != operations
                or meta.get('output_dtype') != ('fp8' if fp8 else 'bf16') or meta.get('output_layout') != 'tiles'
                or meta.get('output_offset') != m*k+k*n+256 or meta.get('output_bytes') != output_bytes
                or actual.get('output_offset') != meta.get('output_offset')
                or meta.get('extra_offset') != extra_offset or meta.get('extra_bytes') != extra_bytes
                or meta.get('total_dram_bytes') != total or actual.get('total_dram_bytes') != total
                or actual.get('dram_base') != 0x90000000 or actual.get('recipe') != 'traffic-reuse-v1'
                or meta.get('requested_dma') != actual.get('requested_dma')
                or actual['requested_dma'].get('store_bytes') != output_bytes
                or type(actual['requested_dma'].get('load_bytes')) is not int or actual['requested_dma']['load_bytes'] <= 0
                or fp8 and meta.get('output_scale_exponent') != actual_calls[actual['call']]['output_decode_exponent']):
                raise ValueError('Full-policy fusion output, extras, or DMA geometry differs')
            continue
        if actual.get('fusion'):
            raise ValueError('Unselected full-policy panel declares fusion')
        if (actual.get('dram_base') != manifest.get('placement', {}).get('arena_base', 0x100000000) or actual.get('output_offset') != m*k+k*n+256
            or actual.get('total_dram_bytes') != m*k+k*n+512+2*m*n
            or actual.get('requested_dma') != {'load_bytes': loads, 'store_bytes': 2*m*n}):
            raise ValueError('Full-policy panel staging or DMA differs')
    return expected_calls
