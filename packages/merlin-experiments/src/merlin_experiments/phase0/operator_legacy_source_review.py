"""Prospective source-only inventory review for an explicitly selected old bundle.

This does not manufacture a producer receipt or a fresh numerical seal. The
original launch, selected private spec, complete old leaf index and source-role
catalogues must independently close. Only source and ownership metadata is read;
unrelated operand/result arrays are neither decoded nor freshly certified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import yaml

SCHEMA = 'merlin.prospective_legacy_source_inventory_review.v1'


def _load(record):
    from . import operator_exclusion_intake as I
    from merlin.model_check.seal import _files

    if (set(record) != {'schema', 'review_id', 'manifest', 'sums', 'original_launch',
                       'private_spec', 'qualification_issued', 'meaning'}
            or record.get('schema') != SCHEMA or not record.get('review_id')
            or record.get('qualification_issued') is not False):
        raise ValueError('prospective source review lacks exact owner scope')
    manifest_path, raw = I.pinned(record['manifest'])
    manifest = json.loads(raw)
    sums_path, sums_raw = I.pinned(record['sums'])
    launch_path, launch_raw = I.pinned(record['original_launch'])
    launch = json.loads(launch_raw)
    spec_path, spec_raw = I.pinned(record['private_spec'])
    spec = yaml.safe_load(spec_raw)
    argv = launch.get('argv')
    if (launch.get('execution_started') is not True or not isinstance(argv, list)
            or argv.count('--private-full-model-spec') != 1
            or argv.index('--private-full-model-spec') + 1 >= len(argv)
            or argv[argv.index('--private-full-model-spec') + 1] != str(spec_path)
            or (launch.get('environment') or {}).get('MERLIN_MODEL_CHECK_BUNDLE') != str(manifest_path.parent)):
        raise ValueError('legacy source bundle is not the exact original launch selection')
    if (manifest.get('schema') != 'merlin.model_check.bundle.v1'
            or not isinstance(manifest.get('models'), dict) or not manifest['models']
            or spec.get('target') != manifest.get('target')
            or not isinstance(spec.get('models'), list)):
        raise ValueError('legacy source bundle lacks selected model ownership')
    models = {}
    for row in spec['models']:
        if (not isinstance(row, dict) or not isinstance(row.get('id'), str)
                or row['id'] in models or not isinstance(row.get('capture'), str)):
            raise ValueError('private model selection is ambiguous')
        capture = Path(row['capture'])
        if not capture.is_absolute() or '..' in capture.parts:
            raise ValueError('private model capture is not explicitly selected')
        models[row['id']] = capture
    if set(models) != set(manifest['models']):
        raise ValueError('legacy source model roster differs from original selection')
    sums = I._sums(sums_raw)
    seal = manifest.get('seal') or {}
    if (sums_path.parent != manifest_path.parent or sums_path.name != seal.get('sums')
            or record['sums']['sha256'] != seal.get('sums_sha256')
            or seal.get('files') != len(sums)
            or set(_files(manifest_path.parent)) != set(sums)):
        raise ValueError('legacy complete leaf index or current namespace changed')
    binding = manifest.get('binding') or {}
    relative = PurePosixPath(binding.get('receipt', ''))
    if (binding.get('status') != 'pass' or len(relative.parts) != 1
            or relative.is_absolute() or relative.as_posix() not in sums
            or binding.get('receipt_sha256') != sums[relative.as_posix()]):
        raise ValueError('legacy original binding receipt is absent from its seal')
    I.pinned({'path':str(manifest_path.parent / relative), 'sha256':binding['receipt_sha256']})
    expected, catalogues = set(), {}
    for name, model in manifest['models'].items():
        programs = model.get('programs') or {}
        if model.get('model') != name or not programs:
            raise ValueError('legacy model program roster is incomplete')
        for program, metadata in programs.items():
            if (not isinstance(program, str) or PurePosixPath(program).parts != (program,)
                    or not isinstance(metadata, dict)
                    or not isinstance(metadata.get('source_stage'), str)):
                raise ValueError('legacy source program has no declared original stage')
            stage = Path(metadata['source_stage'])
            if not stage.is_absolute() or '..' in stage.parts or not stage.is_relative_to(models[name]):
                raise ValueError('legacy source stage differs from selected private capture')
            prefix = f'{name}/programs/{program}/'
            sources = {member for member in sums if member.startswith(prefix)
                       and member.endswith('/model.prepared.mlir')}
            if not sources:
                raise ValueError('legacy program has no complete prepared source')
            for leaf in ('catalog.json', 'census.json'):
                member = prefix + leaf
                if member not in sums:
                    raise ValueError('legacy source ownership metadata is missing')
                path, content = I.pinned({'path':str(manifest_path.parent / member), 'sha256':sums[member]})
                if leaf == 'catalog.json':
                    catalogue = json.loads(content)
                    if catalogue.get('schema') != 'merlin.model_check.catalog.v1':
                        raise ValueError('legacy source role catalogue is unsupported')
                    catalogues[prefix] = (path, catalogue)
            expected.update(sources)
    actual = {member for member in sums if member.endswith('/model.prepared.mlir')}
    if actual != expected:
        raise ValueError('legacy source inventory omits a sealed prepared source')
    result = {}
    for member in sorted(expected):
        path = manifest_path.parent / member
        role = path.parent.name
        prefix = path.parent.parent.relative_to(manifest_path.parent).as_posix() + '/'
        _, catalogue = catalogues[prefix]
        if role not in {'ref','dev'} or catalogue.get(role + '_prepared') != str(path):
            raise ValueError('legacy source has no exact original catalogue role')
        # This adapter intentionally accepts ordinary original leaves only.
        # Historical aliases require the separate existing alias-aware intake.
        I.ordinary(path)
        result[str(path)] = dict(sha256=sums[member], read_path=str(path), alias=None)
    # Reopen all selected metadata; the streaming issuer hashes every complete
    # source before publishing exclusions and repeats this inventory afterwards.
    for key in ('manifest','sums','original_launch','private_spec'):
        I.pinned(record[key])
    return result


def resolve(pin):
    from . import operator_exclusion_intake as I
    return _load(json.loads(I.pinned(pin)[1]))


def prepare(*, manifest, sums, original_launch, private_spec, review_id, output):
    """Prepare a NEW owner review, without claiming historical producer status."""
    record = dict(schema=SCHEMA, review_id=review_id, manifest=manifest, sums=sums,
                  original_launch=original_launch, private_spec=private_spec,
                  qualification_issued=False,
                  meaning='prospective complete source-only exclusion intake; no numerical seal or historical producer receipt issued')
    sources = _load(record)
    path = Path(output)
    if not path.is_absolute() or path.exists() or path.parent.stat().st_mode & 0o077:
        raise ValueError('prospective source review needs a fresh owner-only leaf')
    raw = (json.dumps(record, indent=2, sort_keys=True) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
    path.chmod(0o600)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()), len(sources)
