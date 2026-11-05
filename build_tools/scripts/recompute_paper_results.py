#!/usr/bin/env python3
"""Verify saved result inputs and recompute Merlin paper statistics without execution.

The registry selects immutable CSV/JSON records. This command runs no compiler,
agent, simulator, FPGA job or original producer script and never changes inputs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def gm(values):
    values = list(values)
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('a geometric mean needs finite positive observations')
    return math.exp(statistics.fmean(map(math.log, values)))


def check_close(observed, recorded, name):
    if not math.isclose(observed, float(recorded), rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError(f'{name}: recomputed {observed} differs from recorded {recorded}')


def truth(value):
    return str(value).lower() == 'true'


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    lo = math.floor(index)
    return values[lo] + (values[math.ceil(index)] - values[lo]) * (index - lo)


class Inputs:
    def __init__(self, root, registry):
        self.root = root.resolve()
        self.registry = registry
        self.verified = {}

    def path(self, key):
        row = self.registry['inputs'][key]
        relative = PurePosixPath(row['file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('input path must stay within artifact root')
        path = self.root / relative
        if not path.resolve().is_relative_to(self.root) or not path.is_file():
            raise ValueError(f'missing or escaping input: {row["file"]}')
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError(f'input checksum differs: {row["file"]}')
        self.verified[key] = row
        return path

    def json(self, key):
        return json.loads(self.path(key).read_bytes())

    def csv(self, key):
        with self.path(key).open(newline='') as source:
            return list(csv.DictReader(source))


def validated_cycles(row):
    if not truth(row.get('rankable')):
        return None
    if row.get('correctness_status') not in (None, '', 'pass', 'passed'):
        return None
    for field in ('mismatches', 'mismatch_elements', 'mismatch_bytes', 'guard_failures'):
        if row.get(field) not in (None, '', '0'):
            raise ValueError(f'rankable row has nonzero {field}')
    raw = row.get('median_cycles') or row.get('validated_cycles')
    if not raw:
        return None
    cycles = float(raw)
    if not math.isfinite(cycles) or cycles <= 0:
        raise ValueError('invalid cycle count')
    if row.get('cycle_samples'):
        samples = json.loads(row['cycle_samples'])
        check_close(statistics.median(samples), cycles, 'saved cycle median')
    return cycles


def same_image_pair(left, right):
    for field in ('case_id', 'hardware_id', 'elf_sha256', 'input_profile', 'input_sha256s', 'reference_sha256s'):
        if not left.get(field) or left.get(field) != right.get(field):
            raise ValueError(f'same-image comparison has different or missing {field}')
    return validated_cycles(left), validated_cycles(right)


def master_native(data):
    rows = data.csv('frozen-master_long')
    return {(r['system'], r['case_id']): r for r in rows
            if r['record_origin'] == 'native_measured' and r['measurement_profile'] == 'warm_repeat'}


def image_rows(data, key, system=None):
    return [r for r in data.csv(key)
            if r['cache_profile'] == 'warm_repeat' and (system is None or r['system'] == system)]


def supplied_comparison(data, cells, taxonomy):
    """Check the complete development panel against supplied baseline observations."""
    selected = {case for case, row in taxonomy.items() if row['case_family'] == 'main16'}
    supplied = {}
    for row in data.csv('frozen-master_long'):
        case = row['case_id']
        if row['system'] != 'agentic_external' or case not in selected or not row['reported_cycles']:
            continue
        value = float(row['reported_cycles'])
        if not math.isfinite(value) or value <= 0:
            raise ValueError('invalid supplied baseline cycles')
        if case in supplied and supplied[case] != value:
            raise ValueError('conflicting supplied baseline observations')
        supplied[case] = value
    by_key = {(row['system'], row['case']): row['cycles'] for row in cells}
    if not selected or set(supplied) != selected:
        raise ValueError('supplied baseline does not cover the declared development panel')
    initial, refined = 'F10-FULL-NOREF-NOCCA', 'CCA-EXO-R3'
    comparisons = {}
    for system in (initial, refined):
        values = {case: by_key.get((system, case)) for case in selected}
        if any(value is None for value in values.values()):
            raise ValueError('compiler does not cover the supplied comparison panel')
        ratios = {case: supplied[case] / values[case] for case in sorted(selected)}
        comparisons[system] = dict(cases=len(ratios), speed_geomean=gm(ratios.values()), per_case=ratios)
    improvement = gm(by_key[(initial, case)] / by_key[(refined, case)] for case in selected)
    return dict(comparisons=comparisons, refinement_over_initial=improvement,
                scope='Saved supplied cycle values only. External implementations and arithmetic, timing and budget matching are not independently qualified.')


def frozen(data):
    rows = master_native(data)
    extra = ['cca-exo-r3-full-v1', 'cca-tinyllama-r2-full-v1', 'cca-all-r2-full-v1',
             'cca-resnet50-r2-full-v1', 'cca-exo-r2-full-v1', 'full-ref-full-v1', 'full-ref-4h-full-v1']
    same_image = {}
    for name in extra:
        measured = image_rows(data, name)
        for row in measured:
            same_image[(name, row['system'], row['case_id'])] = row
            if row['system'] != data.registry['systems']['automatic_library']:
                rows[(row['system'], row['case_id'])] = row
    protocol = data.json('frozen-protocol')
    peak = float(protocol['peak_macs_per_cycle'])
    cells = []
    for (system, case), row in sorted(rows.items()):
        cycles = validated_cycles(row)
        useful = float(row.get('expected_useful_macs') or row.get('useful_macs') or 0)
        cells.append(dict(system=system, case=case, cycles=cycles,
                          utilization_pct=100 * useful / peak / cycles if cycles else None,
                          compiler_source_sha256=row.get('compiler_source_sha256'),
                          kernel_sha256=row.get('kernel_sha256'), elf_sha256=row.get('elf_sha256'),
                          status='measured' if cycles else 'unrankable'))
    by_key = {(r['system'], r['case']): r for r in cells}
    cases = sorted({c for s, c in by_key if s == 'CCA-EXO-R3'})
    taxonomy = {r['case_id']: r for r in data.csv('figure-data-cases_taxonomy.csv')}
    declared_cases = {r['case_id'] for r in data.json('frozen-cases')['cases']}
    if set(cases) != declared_cases or len(cases) != protocol['total_case_ids']:
        raise ValueError('frozen compiler rows do not cover the declared case roster')
    summaries = {}
    for reference in ('F10-FULL-NOREF-NOCCA', data.registry['systems']['automatic_library'], data.registry['systems']['primitive_library']):
        matched = [(c, by_key[(reference, c)]['cycles'] / by_key[('CCA-EXO-R3', c)]['cycles'])
                   for c in cases if (reference, c) in by_key and by_key[(reference, c)]['cycles']
                   and by_key[('CCA-EXO-R3', c)]['cycles']]
        summaries[reference] = dict(cases=len(matched), speed_geomean=gm(x[1] for x in matched),
                                   per_case=dict(matched), comparison='original common reference cohort',
                                   by_panel={panel: dict(cases=len(items), speed_geomean=gm(v for _, v in items))
                                             for panel in ('main16', 'shape_sweep')
                                             if (items := [(c, v) for c, v in matched if (taxonomy[c]['case_family'] == 'main16') == (panel == 'main16')])})
    controls = []
    for c in cases:
        left = same_image.get(('cca-exo-r3-full-v1', data.registry['systems']['automatic_library'], c))
        right = same_image.get(('cca-exo-r3-full-v1', 'CCA-EXO-R3', c))
        if left and right:
            lhs, rhs = same_image_pair(left, right)
            if lhs is None or rhs is None:
                continue
            controls.append(dict(case=c, fsm_over_merlin=lhs / rhs,
                                 elf_sha256=right['elf_sha256']))
    return dict(peak_macs_per_cycle=peak, cases=len(cases), cells=cells,
                geometric_mean_speed=summaries, same_image_stock_fsm_controls=controls,
                supplied_iterative_baseline=supplied_comparison(data, cells, taxonomy),
                note='Common-reference and same-image results are separate. Adapted/latest FSM controls live in the later-fsm-controls artifact. Utilization is useful peak attainment, not a PE-busy hardware counter.')


def cca(data):
    native = master_native(data)
    seed = {c: validated_cycles(r) for (s, c), r in native.items() if s == 'F10-FULL-NOREF-NOCCA'}
    taxonomy = {r['case_id']: r['op_family'] for r in data.csv('figure-data-cases_taxonomy.csv')}
    def cycles(keys, system):
        grouped = defaultdict(list)
        for key in keys:
            for row in image_rows(data, key, system):
                value = validated_cycles(row)
                if value is not None:
                    grouped[row['case_id']].append(value)
        return {case: statistics.median(values) for case, values in grouped.items()}
    raw = cycles(['cca-raw-full-v1-s1', 'cca-raw-full-v1-s2', 'cca-raw-full-v1-s3'], 'CCA-RAW')
    arms = [('exo', 'CCA-EXO-R3', 'cca-exo-r3-full-v1', 'CCA-EXO', 'cca-exo-full-v1'),
            ('resnet50', 'CCA-RESNET50-R2', 'cca-resnet50-r2-full-v1', 'CCA-RESNET50', 'cca-resnet50-full-v1'),
            ('tinyllama', 'CCA-TINYLLAMA-R2', 'cca-tinyllama-r2-full-v1', 'CCA-TINYLLAMA', 'cca-tinyllama-full-v1')]
    out = {}
    for label, on_name, on_file, off_name, off_file in arms:
        on, off = cycles([on_file], on_name), cycles([off_file], off_name)
        values = {}
        for family in sorted(set(taxonomy.values())):
            keys = sorted(c for c in on if taxonomy[c] == family and seed.get(c))
            if keys:
                values[family] = dict(cases=len(keys), on_over_seed_speed=gm(seed[c] / on[c] for c in keys),
                                     earlier_off_over_seed_speed=gm(seed[c] / off[c] for c in keys),
                                     raw_over_seed_speed=gm(seed[c] / raw[c] for c in keys))
        matched = sorted(c for c in on if seed.get(c))
        out[label] = dict(cases=len(matched), speed_vs_seed=gm(seed[c] / on[c] for c in matched),
                          speed_vs_raw=gm(raw[c] / on[c] for c in matched), families=values)
    return dict(arms=out, confound='CCA-on also receives additional operator notes and a newer export. This is not a matched ablation.')


def scaffolding(data):
    prefix = 'figure-data-el_ladder-'
    correct = data.csv(prefix + 'correctness_timeseries.csv')
    timeline = data.csv(prefix + 'token_timeline.csv')
    latency = data.csv(prefix + 'selfcheck_latency.csv')
    failures = data.csv(prefix + 'failures_by_hour.csv')
    tokens = data.csv(prefix + 'tokens_by_type.csv')
    recorded = data.json(prefix + 'summary.json')
    out = {}
    for run in ('EL1', 'EL2', 'EL3'):
        curve = [r for r in correct if r['run'] == run and r['suite'] == 'cases']
        at_hour = max([int(r['cum_l2_pass_cases']) for r in curve if float(r['elapsed_s']) <= 3600] or [0])
        token_hour = max([int(r['cum_uncached_input']) for r in timeline if r['run'] == run and float(r['elapsed_s']) <= 3600] or [0])
        failed = defaultdict(int)
        for row in failures:
            if row['run'] == run and row['window'] == 'author' and row['failure_class'] != 'expected_refusal':
                failed[row['failure_class']] += int(row['count'])
        times = [float(r['latency_s']) for r in latency if r['run'] == run and truth(r['author']) and r['latency_s']]
        check_close(at_hour, recorded['runs'][run]['correctness']['cases_by_60m'], 'E1-E3 recorded hourly coverage')
        totals = {r['token_type']: int(r['tokens']) for r in tokens
                  if r['run'] == run and r['role'] == 'all' and r['window'] == 'author' and r['hour'] == 'all'}
        out[run] = dict(cases_by_60m=at_hour, uncached_input_by_60m=token_hour,
                        failed_grade_results=sum(failed.values()), failure_classes=dict(failed),
                        selfcheck_latency_s=dict(n=len(times), median=statistics.median(times),
                                                q25=quantile(times, .25), q75=quantile(times, .75)),
                        author_window_tokens=totals)
    return dict(runs=out, uncached_input_reduction_e3_vs_e2=1 - out['EL3']['uncached_input_by_60m'] / out['EL2']['uncached_input_by_60m'],
                note='Historical successes span compiler revisions. Failures count grade results, not distinct defects. Request mixes differ.')


def scheduling(data):
    figure = data.json('scheduling-figure')
    original = data.json('scheduling-events')
    out = []
    for row in figure['panel_b']['rows']:
        control, treatment = row['control_s'], row['tool_s']
        if control <= 0 or treatment <= 0 or max(control, treatment) > row['budget_s']:
            raise ValueError('invalid scheduling milestone')
        ratio = control / treatment
        check_close(ratio, row['speedup'], 'scheduling time ratio')
        out.append(dict(study=row['id'], budget_s=row['budget_s'], control_s=control,
                        checked_scheduler_s=treatment, sooner_ratio=ratio))
    anchor = next(s for s in original['studies'] if s['id'] == figure['panel_a']['study'])
    for index, field in [(0, 'without'), (1, 'with_tool')]:
        check_close(anchor['arms'][index]['events']['g10'], figure['panel_a']['first_10pct_s'][field], 'anchor feedback milestone')
    return dict(studies=out, anchor_seed_modeled_cycles=original['seed_cycles'],
                note='Recomputed from extracted feedback-availability milestones. Author logs and executable byte identity are separate evidence. These are modeled improvements, not measured FireSim speedups.')


def convolution(data):
    figure, raw = data.json('convolution-figure'), data.json('convolution-raw')
    results = []
    for shape in figure['firesim']['shapes']:
        case = shape['case']
        row = dict(case=case)
        for right, output in [('seed', 'library_over_seed'), ('oracle', 'library_over_oracle')]:
            ratios = []
            for layout in raw['layouts']:
                contrast = next(c for c in layout['contrasts'][case] if c['left'] == 'run-01' and c['right'] == right)
                if not contrast['usable']:
                    raise ValueError('plotted convolution contrast lacks a usable layout')
                ratio = contrast['left_median'] / contrast['right_median']
                check_close(ratio, contrast['left_over_right'], 'convolution within-layout ratio')
                ratios.append(ratio)
            if len(ratios) != 4:
                raise ValueError('convolution requires all four layouts')
            value = gm(ratios)
            check_close(value, shape[output], 'convolution plotted geometric mean')
            row[output] = value
            row[output + '_by_layout'] = ratios
        results.append(row)
    return dict(shapes=results, baseline='hangs on hardware, not a valid finite speed ratio',
                strict_physical_qualification=raw['strict_physical_qualification'],
                note='Saved same-image median contrasts are checked. Consumed-ELF/domain qualification is not established by this recomputation.')


def controlled(data):
    figure, raw = data.json('controlled-cca-figure'), data.json('controlled-cca-raw')
    output = {}
    for key, selection in figure['pairs'].items():
        cohort = next(c for c in raw['cohorts'] if c['cohort'] == selection['id'])
        by_case = defaultdict(list)
        boots = []
        for boot in cohort['boots']:
            ratios = []
            for row in boot['case_observations']:
                pair = [row['cells'].get(selection[x]) for x in ('cca_subject', 'no_cca_subject')]
                if not all(c and c['physical_state'] == 'pass' for c in pair):
                    continue
                medians = []
                for cell in pair:
                    samples = cell['samples']
                    if not samples or any(s['mismatched_bytes'] != 0 for s in samples):
                        raise ValueError('numerically invalid controlled CCA cell')
                    median = statistics.median(s['cycles'] for s in samples)
                    check_close(median, cell['median_cycles'], 'controlled CCA median')
                    medians.append(median)
                ratio = medians[0] / medians[1]
                ratios.append(ratio)
                by_case[row['case']].append(ratio)
            value = 100 * (gm(ratios) - 1)
            saved = next(b for b in selection['boots'] if b['boot'] == boot['boot'])
            check_close(value, saved['all_matched_pct'], 'controlled CCA boot mean')
            boots.append(dict(boot=boot['boot'], measured_cases=len(ratios), cycle_change_pct=value))
        if len(boots) != 4 or any(len(v) != 4 for v in by_case.values()):
            raise ValueError('controlled CCA contrasts require all four boots per measured case')
        value = 100 * (gm(gm(v) for v in by_case.values()) - 1)
        check_close(value, selection['all_matched_pct'], 'controlled CCA pair mean')
        output[key] = dict(cycle_change_pct=value, measured_cases=len(by_case),
                           boots=boots, case_ratios={case: gm(v) for case, v in sorted(by_case.items())})
    return dict(pairs=output, strict_physical_qualification=raw['strict_physical_qualification'],
                note='Four boots preserve hardware contrasts, not additional independent author replicates. Unmeasured cases remain outside the reported measured subset.')


def fsm(data):
    latest = data.json('fsm-rerun-verified-rerun-results.json')
    summary = data.json('fsm-rerun-hardware-validation-summary.json')
    if any(summary[field] != 0 for field in ('numerical_mismatches', 'guard_failures', 'input_failures', 'done_code')):
        raise ValueError('latest FSM summary contains failed checks')
    paired = defaultdict(list)
    for row in latest:
        for system in ('fsm', 'merlin'):
            check_close(statistics.median(row[system + '_samples']), row[system + '_median'], 'latest FSM sample median')
        ratio = row['fsm_median'] / row['merlin_median']
        check_close(ratio, row['fsm_over_merlin'], 'latest FSM per-order ratio')
        paired[row['case_id']].append(dict(pass_id=row['pass_id'], fsm_over_merlin=ratio))
    if len(paired) != summary['cases'] or any(len(v) != 2 for v in paired.values()):
        raise ValueError('latest FSM requires two opposite orders for every case')
    adapted = {}
    for cohort in ('fsm-counterfactual-execution', 'fsm-decode-strides'):
        rows = data.json(cohort + '-verified-results.json')
        observation = data.json(cohort + '-hardware-validation-summary.json')
        if observation.get('all_checks_pass') is False or observation.get('done_code', observation.get('done', 0)) != 0:
            raise ValueError('adapted FSM observation contains failed checks')
        grouped = defaultdict(dict)
        for row in rows:
            median = statistics.median(row['samples'])
            check_close(median, row['median_cycles'], 'adapted FSM sample median')
            grouped[(row['case_id'], row['pass_id'])][row['system']] = median
        result = []
        for (case, order), systems in sorted(grouped.items()):
            for system, median in sorted(systems.items()):
                result.append(dict(case=case, pass_id=order, implementation=system,
                                   median_cycles=median,
                                   implementation_over_merlin=median / systems['CCA-EXO-R3'],
                                   stock_over_implementation=systems['fsm-current-auto'] / median))
        adapted[cohort] = result
    return dict(latest_stock={case: dict(orders=v, geometric_mean_speed=gm(r['fsm_over_merlin'] for r in v))
                              for case, v in sorted(paired.items())},
                adapted_controls=adapted,
                note='Ratios greater than one in implementation_over_merlin favor Merlin. Stock automatic policy, manual tile trials and grouped-head calling are distinct comparisons. Two orders share one boot.')


FUNCTIONS = dict(frozen=frozen, cca=cca, scaffolding=scaffolding, scheduling=scheduling,
                 convolution=convolution, controlled=controlled, fsm=fsm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--recipe', action='append', help='stable recipe id, repeatable, default all')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_bytes())
    if registry.get('schema') != 'merlin.paper_result_recipes.v1':
        raise ValueError('unsupported recipe registry schema')
    available = {row['id']: row for row in registry['recipes']}
    selected = args.recipe or list(available)
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f'unknown recipes: {sorted(unknown)}')
    data = Inputs(args.artifact_root, registry)
    results = {}
    # Verify inputs before calculation. No missing recipe is silently skipped.
    for name in selected:
        for key in available[name]['inputs']:
            data.path(key)
    for name in selected:
        recipe = available[name]
        results[name] = dict(metrics=FUNCTIONS[recipe['function']](data),
                             limitations=recipe['missing'], scope=recipe['scope'])
    output = args.output.absolute()
    if output.resolve().is_relative_to(data.root):
        raise ValueError('write output outside the immutable artifact input tree')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite a previous recomputation receipt.
    with output.open('x') as target:
        json.dump(dict(schema='merlin.paper_results_recomputed.v1', registry_sha256=digest(args.registry),
                       script_sha256=digest(Path(__file__)), execution_started=False,
                       verified_inputs=data.verified, results=results), target, indent=2)
        target.write('\n')
    print(json.dumps(dict(recipes=selected, verified_inputs=len(data.verified),
                          output=str(output), execution_started=False), sort_keys=True))


if __name__ == '__main__':
    main()
