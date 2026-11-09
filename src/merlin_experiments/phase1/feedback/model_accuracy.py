"""Descriptive absolute-error cards over independently reopened public receipts.

The native evidence owner supplies normalized model/physical records *after*
producer/runtime/output verification. This reporting utility does not certify an
input dictionary, fit parameters, select a model, or authorize measurements.
Legacy case-name/execution-digest ratios alone cannot establish these joins.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import statistics

SCHEMA = "merlin.public_model_accuracy_card.v1"
BINDINGS = ("source_group_sha256", "capsule_sha256", "execution_digest", "elf_sha256",
            "input_sha256", "hardware_config_sha256", "scope_sha256", "physical_domain_sha256")
MAX_RECORDS = 4096


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _positive(value):
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def _size(macs):
    # Published descriptive buckets, not learned performance boundaries.
    if type(macs) is not int or macs < 0:
        return "unknown_work"
    return "small" if macs <= 1_048_576 else "medium" if macs <= 16_777_216 else "large"


def _quantile(values, q):
    values = sorted(values)
    p = (len(values)-1)*q
    lo = int(p);hi = min(lo+1,len(values)-1)
    return values[lo] + (values[hi]-values[lo])*(p-lo)


def _summary(rows):
    errors = [abs(r["modeled"]-r["measured"])/r["measured"] for r in rows]
    return dict(program_pairs=len(rows), distinct_capsules=len({r['capsule_sha256'] for r in rows}),
        independent_source_groups=len({r['source_group_sha256'] for r in rows}),
        median_absolute_relative_error=statistics.median(errors) if errors else None,
        p95_absolute_relative_error=_quantile(errors,.95) if errors else None,
        maximum_absolute_relative_error=max(errors) if errors else None,
        pairs_within_5_percent=sum(e <= .05 for e in errors),
        fraction_within_5_percent=sum(e <= .05 for e in errors)/len(errors) if errors else None,
        # A small signed median can hide large positive/negative errors.
        median_measured_over_modeled=statistics.median(r['measured']/r['modeled'] for r in rows) if rows else None)


def _pair(record):
    model, physical = record.get('model') or {}, record.get('physical') or {}
    a, b = model.get('binding') or {}, physical.get('binding') or {}
    if any(not _digest(a.get(k)) or a.get(k) != b.get(k) for k in BINDINGS):
        return None, 'missing_or_mismatched_source_execution_input_config_scope_binding'
    if model.get('public_source') is not True or physical.get('public_source') is not True:
        return None, 'nonpublic_or_unclassified_source'
    if (model.get('timing_complete') is not True or model.get('unresolved') != []
            or physical.get('engine') != 'gsim'
            or physical.get('numerical_status') != 'pass' or physical.get('structural_status') != 'pass'):
        return None, 'incomplete_model_or_unqualified_physical_product'
    if (not _digest(model.get('model_sha256')) or not _digest(model.get('comparison_domain_sha256'))
            or not _positive(model.get('cycles')) or not _positive(physical.get('cycles'))):
        return None, 'missing_model_domain_or_positive_cycles'
    row = dict(a, modeled=model['cycles'], measured=physical['cycles'],
        model_sha256=model['model_sha256'], comparison_domain_sha256=model['comparison_domain_sha256'],
        family=record.get('family') if isinstance(record.get('family'),str) else 'unclassified',
        size=_size(record.get('useful_macs')), split=record.get('split','unsplit'))
    if row['split'] not in {'train','held','unsplit'}:
        return None, 'unknown_source_split'
    return row, None


def card(records):
    """Partition by evaluator/domain/family/work; never issue qualification.

    Repeated versions of one source group do not become independent witnesses.
    Duplicate exact pairs are counted once; contradictory cycles or split
    assignments remain explicit exclusions. Held means a caller-provided frozen
    split, not a split inferred retrospectively from favorable errors.
    """
    if not isinstance(records, (list,tuple)) or len(records) > MAX_RECORDS:
        raise ValueError('accuracy reporting requires at most 4096 bounded public receipt pairs')
    excluded = Counter();rows = {};conflicts = set();group_splits = defaultdict(set)
    for record in records:
        if not isinstance(record,dict):
            excluded['invalid_record'] += 1;continue
        row, reason = _pair(record)
        if row is None:
            excluded[reason] += 1;continue
        key = tuple(row[k] for k in BINDINGS) + (row['model_sha256'],row['comparison_domain_sha256'])
        if key in rows and rows[key] != row:
            conflicts.add(key)
        rows.setdefault(key,row)
        if row['split'] != 'unsplit':
            group_splits[(row['model_sha256'],row['comparison_domain_sha256'],row['source_group_sha256'])].add(row['split'])
    usable = []
    for key,row in rows.items():
        if key in conflicts:
            excluded['contradictory_duplicate_pair'] += 1
        elif len(group_splits[(row['model_sha256'],row['comparison_domain_sha256'],row['source_group_sha256'])]) > 1:
            excluded['source_group_crosses_train_held'] += 1
        else:
            usable.append(row)
    partitions = defaultdict(list)
    for row in usable:
        partitions[(row['model_sha256'],row['comparison_domain_sha256'],row['physical_domain_sha256'])].append(row)
    output = []
    for (model,domain,physical_domain), members in sorted(partitions.items()):
        strata = defaultdict(list)
        for row in members:
            strata[(row['family'],row['size'],row['split'])].append(row)
        output.append(dict(model_sha256=model,comparison_domain_sha256=domain,
            physical_domain_sha256=physical_domain,summary=_summary(members),
            covered_size_buckets=sorted({r['size'] for r in members}),
            missing_size_buckets=sorted({'small','medium','large'}-{r['size'] for r in members}),
            strata=[dict(family=f,size=s,split=split,**_summary(group))
                    for (f,s,split),group in sorted(strata.items())]))
    return dict(schema=SCHEMA,status='DESCRIPTIVE' if usable else 'UNKNOWN',
        supplied_records=len(records),usable_program_pairs=len(usable),excluded=dict(sorted(excluded.items())),
        partitions=output,qualification_issued=False,model_activation_authority=False,
        broad_under_5_percent_claim=False,
        basis='absolute(model-measured)/measured; exact public source/program/input/config/scope joins; gSIM domain only',
        unknown_coverage_included_in_accuracy=False,
        note='Missing/unsupported cases are exclusions, not zero error; small probes do not establish large or FireSim accuracy.')


def legacy_card(rows):
    """Expose the existing ratio overlay's missing custody instead of qualifying it."""
    return dict(schema=SCHEMA,status='UNKNOWN',qualification_issued=False,
        broad_under_5_percent_claim=False,legacy_program_versions=len(rows),
        legacy_distinct_capsules=len({r.get('test_case') for r in rows}),
        usable_program_pairs=0,partitions=[],
        reason='Legacy case/execution-digest ratio pairs lack source-group, model/domain, input/config and exact counter-scope joins; use card() with verified native receipt records.')
