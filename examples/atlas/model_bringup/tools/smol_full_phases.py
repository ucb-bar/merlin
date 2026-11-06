"""Fixed semantic phase partitions and native reference live inventory."""
from itertools import groupby
from dataclasses import dataclass
from smol_full_plan import calls, panels, reference_geometry


@dataclass(frozen=True)
class ExecutionProfile:
    name: str
    attention_count: int

    @property
    def launch_count(self): return 2715 + self.attention_count

    @property
    def cycle_bytes(self): return 4 * self.launch_count

    def events(self):
        if self.attention_count:
            from smol_attention_assets import event_plan
            return event_plan()
        return [dict(index=p['index'], kind='projection', kernel=p['index'], call=p['call']) for p in panels()]


V1 = ExecutionProfile('v1', 0)
V2 = ExecutionProfile('v2', 200)


def execution_profile(manifest):
    """Select only a complete, internally consistent manifest execution inventory."""
    kernels = manifest.get('kernels')
    if (type(manifest.get('kernel_count')) is not int or manifest['kernel_count'] != 2715
        or not isinstance(kernels, list) or len(kernels) != 2715
        or any(not isinstance(p, dict) or type(p.get('index')) is not int or p['index'] != i
               for i,p in enumerate(kernels))):
        raise ValueError('Incomplete Smol projection profile')
    attention = manifest.get('attention_kernels', [])
    if not isinstance(attention, list): raise ValueError('Invalid Smol attention profile')
    profile = V2 if attention else V1
    if attention:
        expected = [e for e in profile.events() if e['kind'] == 'attention']
        if (len(attention) != 200 or type(manifest.get('attention_count')) is not int or manifest['attention_count'] != 200
            or type(manifest.get('device_event_count')) is not int or manifest['device_event_count'] != profile.launch_count
            or manifest.get('device_events') != profile.events()
            or any(not isinstance(row, dict) or type(row.get('index')) is not int or row['index'] != i
                   or type(row.get('phase')) is not int or row['phase'] != i
                   or row.get('before_call') != expected[i]['before_call'] for i,row in enumerate(attention))):
            raise ValueError('Incomplete Smol v2 attention profile')
    elif (type(manifest.get('attention_count', 0)) is not int or manifest.get('attention_count', 0) != 0
          or type(manifest.get('device_event_count', 2715)) is not int or manifest.get('device_event_count', 2715) != 2715
          or manifest.get('device_events') is not None):
        raise ValueError('Attention-free Smol profile has attention events')
    if manifest.get('execution_profile', profile.name) != profile.name:
        raise ValueError('Smol execution profile differs from inventory')
    return profile


def segments(profile=V1):
    sequence, physical = calls(), panels()
    if [x['index'] for x in sequence] != list(range(len(sequence))):
        raise ValueError('call indices are not in complete source order')
    def key(call):
        site, layer = call['site'], call['layer']
        if site == 7: return ('vision', call['image'], min(layer, 11))
        if site in (0, 5): return ('prefix', 0, layer if site == 5 else 0)
        return ('expert', call['step'], layer if site == 6 else 15 if site == 4 else 0)
    result = []
    for identity, grouped in groupby(sequence, key):
        group = list(grouped)
        start, end = group[0]['index'], group[-1]['index'] + 1
        covered = [p['index'] for p in physical if start <= p['call'] < end]
        if covered != list(range(covered[0], covered[-1] + 1)):
            raise ValueError('noncontiguous physical phase')
        result.append(dict(index=len(result), kind=identity[0], ordinal=identity[1], layer=identity[2],
                           call_start=start, call_end=end, panel_start=covered[0], panel_end=covered[-1]+1,
                           launch_start=covered[0]+(len(result) if profile.attention_count else 0),
                           launch_end=covered[-1]+1+(len(result)+1 if profile.attention_count else 0)))
    if len(result) != 200 or result[0]['launch_start'] != 0 or result[-1]['launch_end'] != profile.launch_count:
        raise ValueError('incomplete fixed phase partition')
    for left, right in zip(result, result[1:]):
        if left['launch_end'] != right['launch_start'] or left['call_end'] != right['call_start']:
            raise ValueError('overlap or gap between phases')
    return result


EXTRA_LIVE = {'smol_v_tokens':1572864, 'smol_v_output':122880, 'smol_full_image_output':245760,
              'smol_b_output':339840, 'smol_g_velocity':6400, 'smol_full_cycles':10860,
              'smol_full_graph_state':16, 'smol_full_vision_states':32, 'smol_full_phase_state':40}


def extra_live(profile=V1):
    return dict(EXTRA_LIVE, smol_full_cycles=profile.cycle_bytes)


def live_regions(data, image):
    expected = reference_geometry()
    records = data['reference_buffers']
    if len(records) != 47 or {x['name'] for x in records} != set(expected):
        raise ValueError('exact complete reference inventory required')
    result = extra_live(execution_profile(data))
    for row in records:
        shape, dtype = expected[row['name']]
        count = 1
        for dim in shape: count *= dim
        size = count * {'float32':4, 'uint16':2, 'bool':1}[dtype]
        if row['bytes'] != size: raise ValueError('reference inventory size')
        result[row['actual_symbol']] = size
    for name, size in result.items():
        if name not in image.symbols or image.symbols[name]['size'] != size:
            raise ValueError('live symbol extent differs: '+name)
    return result


def scratch_regions(image, live):
    result = {}
    exact = {'smol_full_input_copy','smol_full_activation_copy','smol_full_image'}
    for name, row in image.symbols.items():
        if name not in live and (name.startswith(('smol_v_','smol_b_','smol_g_')) or name in exact):
            if row['size']: result[name] = row['size']
    return result


def check_observations(values, start, end, profile=V1):
    """Validate only genuinely executed current-range deltas before zeroing.

    Native values are synthetic protocol observations, never measured timing.
    Real segment admission must additionally match each value to Device receipts.
    Prior and future slots are canonical zero; no fake prior nonzero seeds exist.
    """
    count = profile.launch_count
    if (len(values) != count or type(start) is not int or type(end) is not int
        or not 0 <= start < end <= count
        or any(type(x) is not int or not 0 <= x < 1 << 32 for x in values)):
        raise ValueError('invalid segment observation array/range')
    if any(values[:start]) or any(values[end:]) or not all(values[start:end]):
        raise ValueError('observations must cover only the actual current range')
    return values[start:end]
