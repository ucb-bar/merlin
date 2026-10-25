"""Compact public workload requirements; no execution, oracle or grade authority.

The selected public roster is the denominator. Grouping helps compiler construction
but never replaces individual numerical/physical obligations or grants a schedule.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .capability_source import analyze

SCHEMA = 'merlin.public_capability_inventory.v1'
FILENAME = 'PUBLIC_CAPABILITIES.json'
SUMMARY = 'PUBLIC_CAPABILITIES.md'
QUERY = 'capability_query.py'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _names(policy, section, key):
    names = policy.get(section, {}).get(key, [])
    if (not isinstance(names, list) or any(not isinstance(n, str) for n in names)
            or len(set(names)) != len(names)):
        raise ValueError('capability inventory requires unique declared capsule IDs')
    return set(names)


def build(public, *, scalability=(), policy):
    """Project only explicit public requirements, preserving the complete roster.

    Inputs are ordinary native-discovered declarations from admitted public views.
    No defaults can transform hidden, absent or excluded sources into public cases.
    """
    core, scale = list(public), list(scalability)
    original = _names(policy, 'correctness_curriculum', 'original_capsules')
    physical = _names(policy, 'gsim_correctness', 'capsules')
    witnesses = _names(policy, 'gsim', 'witness_capsules')
    all_caps = core + scale
    if not all_caps or len(all_caps) > 10000:
        raise ValueError('capability inventory needs a bounded nonempty public roster')
    names = [cap.get('name') for cap in all_caps]
    if (any(not isinstance(n, str) or not n or Path(n).name != n for n in names)
            or len(set(names)) != len(names)):
        raise ValueError('capability inventory has missing or duplicate public IDs')
    if any(cap.get('label') not in {'public', 'dev'} for cap in all_caps):
        raise ValueError('capability inventory refuses non-public capsule declarations')
    core_names, scale_names = {c['name'] for c in core}, {c['name'] for c in scale}
    if not original <= core_names or not physical <= core_names:
        raise ValueError('capability inventory omits a declared correctness/calibration member')
    if original and physical != original:
        raise ValueError('capability inventory must preserve every original physical obligation')
    groups, members = defaultdict(list), {}
    for cap in sorted(all_caps, key=lambda c: c['name']):
        name = cap['name']
        semantic, operation = cap.get('semantic') or {}, cap.get('operation') or {}
        family = str(semantic.get('semantic_family') or operation.get('op') or cap.get('kind') or 'unknown')
        cohort = ('scalability_performance' if name in scale_names else
                  'original_correctness' if name in original else
                  'calibration' if name in witnesses else 'public_development')
        details = analyze(cap)
        tiers = list(cap.get('required_oracle_tiers') or [])
        entry = {
            'cohort': cohort, 'family': family,
            'declared_operation': operation.get('op'),
            'source_sha256': _digest({k: v for k, v in cap.items() if k != '__dir__'}),
            'input_tensors': [{k: t[k] for k in ('name', 'shape', 'dtype', 'layout', 'strides', 'role') if k in t}
                              for t in cap.get('inputs', [])],
            'numeric_policy': cap.get('numeric_policy') or {},
            'layout_requirements': {k: v for k, v in (operation.get('attributes') or {}).items()
                if k in {'layout', 'stride', 'strides', 'padding', 'dilation', 'transpose', 'permutation'}},
            'semantic_requirements': {k: semantic[k] for k in
                ('generalization_axis', 'must_accelerate', 'eligible') if k in semantic},
            'declared_tiers': tiers,
            'development_engines': ['spike', 'gsim'] if name in physical else ['spike'],
            'gsim_role': ('required_correctness' if name in physical else
                          'question_selected_calibration' if name in witnesses else 'not_selected'),
            'performance_priority': 'primary' if name in scale_names else 'diagnostic',
            'source_analysis': details,
        }
        members[name] = entry
        groups[family].append(name)
    capabilities = _capabilities(members)
    return {'schema': SCHEMA, 'total': len(members), 'capability_groups': capabilities,
            'cohorts': dict(sorted(Counter(v['cohort'] for v in members.values()).items())),
            'families': dict(sorted(groups.items())), 'members': members,
            'construction_boundaries': {
                'graph': 'typed graph patterns, region legality, fusion and live outputs',
                'numeric_layout': 'numeric contracts, layout/panel plans and preparation reuse',
                'schedule_storage': 'dependencies, storage lifetimes, resource schedules and pass policy',
                'isa_lowering': 'generic ISA generation, target emission and command legality',
                'host_runtime': 'shared host codegen, dispatch, completion and complete-call execution',
                'integration': 'shared interfaces, committed compiler changes and immutable evaluations'},
            'witnesses_outside_public_roster': len(witnesses - core_names),
            'policy': {
                'construction_unit': 'semantic capability; capsule IDs are traceability only',
                'testing': 'early end-to-end canary per capability, then rolling eligible cases',
                'concurrency': 'correctness, calibration and scalability may advance concurrently',
                'performance': 'prioritize scalability after numerical execution; originals are regressions',
                'calibration': 'eligible exact gSIM evidence only; witness membership is not fit authority',
                'final_obligations': 'unchanged; hidden and complete-model inventories are not exposed here'},
            'qualification_authority': False}


def _capabilities(members):
    """Group source requirements across shapes; canaries are recommendations only."""
    groups = {}
    for name, member in members.items():
        source = member['source_analysis']
        signature = {'family': member['family'], 'source_status': source['status'],
                     'operations': sorted(source.get('operations', {})),
                     'input_dtypes': sorted({str(t.get('dtype', 'unknown')) for t in member['input_tensors']}),
                     'numeric_policy': member['numeric_policy'],
                     'layout_requirements': member['layout_requirements']}
        key = _digest(signature)
        group = groups.setdefault(key, {'requirements': signature, 'members': []})
        group['members'].append(name)
    for group in groups.values():
        def volume(name):
            import math
            shapes = [t.get('shape') for t in members[name]['input_tensors']]
            if not shapes or any(not isinstance(s, list) or any(type(d) is not int or d <= 0 for d in s) for s in shapes):
                return float('inf')
            return sum(math.prod(shape) for shape in shapes)
        ordered = sorted(group['members'], key=lambda n: (volume(n), n))
        group['small_canary_candidate'] = ordered[0]
        group['large_generalization_candidate'] = ordered[-1]
        group['selection_basis'] = 'declared input volume; advisory, no execution or qualification'
    return dict(sorted(groups.items()))


def summary(inventory):
    """Small initial context; the full per-case record stays on demand."""
    lines = ['# Public compiler capability inventory', '',
             f"{inventory['total']} public cases across {len(inventory['families'])} semantic families "
             f"and {len(inventory['capability_groups'])} source/numeric/layout capability groups.", '',
             '| Cohort | Cases | Purpose |', '| --- | ---: | --- |']
    descriptions = {'original_correctness': 'Correctness; retain required Spike and gSIM checks',
                    'calibration': 'Model evidence; selective, question-labelled gSIM',
                    'scalability_performance': 'Spike correctness and primary performance targets',
                    'public_development': 'Declared public requirements; no inferred physical permission'}
    lines.extend(f'| {name} | {count} | {descriptions[name]} |'
                 for name, count in inventory['cohorts'].items())
    lines += ['', '| Semantic family | Cases | Source-analysis unknowns |', '| --- | ---: | ---: |']
    for family, names in list(inventory['families'].items())[:24]:
        unknown = sum(inventory['members'][name]['source_analysis']['status'] != 'parsed' for name in names)
        lines.append(f'| {family} | {len(names)} | {unknown} |')
    if len(inventory['families']) > 24:
        lines.append(f"Additional families: {len(inventory['families']) - 24}; inspect the JSON inventory.")
    lines += ['', f'Per-case shapes, dtypes, graph requirements and engine obligations: `{FILENAME}`.',
              f'List groups: `python {QUERY} --groups`; inspect `--group PREFIX` or `--capsule ID`.',
              'Use --offset/--limit for bounded pages; do not paste the entire catalog into context.',
              'Unknown source analysis remains an implementation/readiness gap, not evidence of an empty graph.',
              'Family grouping grants no grade, physical dispatch, model fit or capsule exemption.', '']
    return '\n'.join(lines)


def stage(workspace, run_dir, inventory):
    """Stage fresh author inputs before the existing task seal; never on resume."""
    data = json.dumps(inventory, sort_keys=True, indent=2, allow_nan=False) + '\n'
    brief = summary(inventory)
    query_script = Path(__file__).with_name(QUERY).read_bytes()
    block = ('\n\n## Build against the whole public corpus\n\n'
             f'Read `{SUMMARY}` first. It summarizes {inventory["total"]} selected public cases; '
             f'Use `python {QUERY} --groups`, `--group PREFIX` or `--capsule ID` '
             f'for bounded queries over `{FILENAME}`. '
             'Build one reusable compiler around the required semantic capabilities and compiler boundaries: '
             'graph, numeric/layout, schedule/storage, ISA lowering, host/runtime, integration. '
             'Assign bounded disjoint edits, integrate shared interfaces early, and let admission gate only tests, '
             'never construction in an independent capability. '
             'For reproducible action sequences use `python construction_actions.py PLAN.json --out OUTPUT`; '
             'it invokes ordinary commands and agent_selfcheck.py, just as an offline study does. '
             'Test a tiny end-to-end case as each capability becomes executable; continue independent '
             'construction while immutable snapshots are evaluated. Calibration prerequisites and large '
             'shape cases can proceed before all original correctness cases pass. The scalability cohort '
             'is the primary performance target; original small cases remain mandatory correctness regressions.\n')
    task = Path(workspace) / 'TASK.md'
    existing = task.read_text()
    for root in (Path(workspace), Path(run_dir)):
        (root / FILENAME).write_text(data)
        (root / SUMMARY).write_text(brief)
        (root / QUERY).write_bytes(query_script)
        (root / QUERY).chmod(0o444)
        action_client = Path(__file__).with_name('construction_actions.py').read_bytes()
        (root / 'construction_actions.py').write_bytes(action_client)
        (root / 'construction_actions.py').chmod(0o444)
        (root / 'TASK.md').write_text(existing + block)
    from .feedback.construction_plan import stage_example
    stage_example(workspace, run_dir, inventory, data.encode())
    return {'schema': SCHEMA, 'sha256': hashlib.sha256(data.encode()).hexdigest(),
            'total': inventory['total'], 'cohorts': inventory['cohorts']}


def stage_selected(workspace, run_dir, *, context, public_roots, contract, excluded):
    """Use the same native public and frozen scalability selections as grading."""
    from merlin.targetgen.capsule_common import discover_capsules, load_capsule
    from .joint_control import selected
    from .feedback.rolling_scalability import Selection

    state = selected()
    if state is None:
        return None
    native = discover_capsules(public_roots, labels={'public', 'dev'}, contract=contract)
    exclusions = set(excluded(cap['name'] for cap in native))
    native = [cap for cap in native if cap['name'] not in exclusions]
    additional = Selection(context)
    scale = [load_capsule(path, contract=contract) for _, path in sorted(additional.cases.items())]
    inventory = build(native, scalability=scale, policy=state['policy'])
    additional.verify()
    return stage(workspace, run_dir, inventory)
