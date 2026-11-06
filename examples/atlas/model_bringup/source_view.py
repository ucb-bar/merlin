"""Build and verify the byte-preserving model/compiler/library execution view."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess

MAP = Path(__file__).with_name("source-map.json")
LIBRARY = "third_party/atlas-kernels"
FORBIDDEN = ("ham" + "mer", "vl" + "si")


def checked_name(value):
    path = Path(value)
    if any(word in part.lower() for part in path.parts for word in FORBIDDEN):
        raise ValueError("forbidden path component")
    return path


def relative(value):
    checked_name(value)
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts or '.git' in path.parts:
        raise ValueError("invalid source-relative path")
    return Path(*path.parts)


def unlinked_path(value):
    path = checked_name(value).absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError("canonical root symlink refused: " + str(current))
    return path


def canonical_root(value):
    return unlinked_path(value).resolve(strict=True)


def source_file(root, name):
    """Reject links before reading either their targets or their descendants."""
    root = checked_name(root)
    name = relative(name)
    current = root
    for part in name.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("source symlink refused: " + str(current))
    if not current.is_file():
        raise ValueError("required canonical source missing: " + str(current))
    return current


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def pinned_library_digest(root, revision, name):
    """Authenticate only the admitted file against the pinned Git blob."""
    name = relative(name).as_posix()
    blob = subprocess.check_output(['git', '-C', str(root), 'cat-file', 'blob',
                                    revision + ':' + name])
    return hashlib.sha256(blob).hexdigest()


def load_map(path=MAP):
    data = json.loads(checked_name(path).read_text())
    if data['schema_version'] != 2 or set(data['sources']) != {'merlin', 'atlas-mlir', 'atlas-kernels'}:
        raise ValueError('unsupported source ownership map')
    entries, seen = [], set()
    for owner, names in data['sources'].items():
        for name in names:
            relative(name)
            dest = LIBRARY + '/' + name if owner == 'atlas-kernels' else name
            if dest in seen:
                raise ValueError('duplicate view destination: ' + dest)
            seen.add(dest)
            if owner == 'atlas-mlir' and not name.startswith(('runtime/', 'tools/')):
                raise ValueError('non-generic compiler mapping: ' + name)
            source = 'examples/atlas/model_bringup/' + name if owner == 'merlin' else name
            entries.append(dict(relative_path=dest, owner=owner, source=source,
                                new=name in data['new_paths']))
    if not set(data['new_paths']).issubset(data['sources']['merlin']):
        raise ValueError('new source absent from Merlin ownership')
    return dict(data, entries=entries)


def test_modules(manifest=None):
    """The compact registration is authoritative; source membership must match it."""
    data = json.loads(MAP.with_name('test-manifest.json').read_text())
    if data['schema_version'] != 2:
        raise ValueError('unsupported test manifest')
    names = data['moved_suites']
    split = data['split_cases']
    if len(names) != len(set(names)) or len(split) != len({e['original_id'] for e in split}):
        raise ValueError('duplicate test registration')
    if len(split) != len({e['destination_id'] for e in split}):
        raise ValueError('duplicate split destination')
    modules = ['test/' + n + '.py' for n in names]
    modules += sorted({'test/' + e['destination_id'].split('.')[0] + '.py' for e in split})
    mapped = {e['relative_path'] for e in (manifest or load_map())['entries']
              if e['relative_path'].startswith('test/test_') and e['relative_path'].endswith('.py')}
    if len(modules) != len(set(modules)) or set(modules) != mapped:
        raise ValueError('registered test membership differs from mapped modules')
    for name in modules:
        relative(name)
    return sorted(modules)


def expected_roots(atlas, merlin, kernels):
    return {name: str(canonical_root(path)) for name, path in
            [('atlas-mlir', atlas), ('merlin', merlin), ('atlas-kernels', kernels)]}


def inventory_path(view):
    return Path(str(view) + '.inventory.json')


def inventory_digest(entries):
    payload = json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def build_view(atlas, merlin, kernels, view, mode='relocated', source_map=MAP):
    if mode not in ('baseline', 'relocated'):
        raise ValueError('invalid source mode')
    roots = {name: Path(path) for name, path in expected_roots(atlas, merlin, kernels).items()}
    view = unlinked_path(view)
    view.parent.mkdir(parents=True, exist_ok=True)
    canonical_root(view.parent)
    if view.exists() or view.is_symlink() or inventory_path(view).exists():
        raise ValueError('execution view must be fresh: ' + str(view))
    for root in roots.values():
        if view.is_relative_to(root):
            raise ValueError('execution view must be outside canonical repositories')
    manifest = load_map(source_map)
    if git(roots['atlas-kernels'], 'rev-parse', 'HEAD') != manifest['library_revision']:
        raise ValueError('library revision differs from admitted pin')
    entries = []
    # Validate every source before creating any part of the view.
    for entry in manifest['entries']:
        owner, name = entry['owner'], entry['source']
        if mode == 'baseline' and owner == 'merlin' and not entry.get('new'):
            owner, name = 'atlas-mlir', entry['relative_path']
        source = source_file(roots[owner], name)
        actual = digest(source)
        if owner == 'atlas-kernels' and actual != pinned_library_digest(
                roots[owner], manifest['library_revision'], name):
            raise ValueError('library source differs from admitted pin: ' + name)
        entries.append(dict(relative_path=entry['relative_path'], source_path=str(source),
                            owner=owner, sha256=actual))
    view.mkdir(parents=True)
    # The bootstrap requires a real pinned library Git identity, but V has no .git.
    library = view / LIBRARY
    library.parent.mkdir(parents=True)
    subprocess.run(['git', 'clone', '--quiet', '--no-checkout', '--shared',
                    str(roots['atlas-kernels']), str(library)], check=True)
    subprocess.run(['git', '-C', str(library), 'read-tree', manifest['library_revision']], check=True)
    for entry in entries:
        dest = view / relative(entry['relative_path'])
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry['source_path'], dest)
        shutil.copymode(entry['source_path'], dest)
        if digest(dest) != entry['sha256']:
            raise ValueError('staged digest differs: ' + str(dest))
    receipt = dict(schema_version=1, mode=mode, view=str(view), roots={k: str(v) for k,v in roots.items()},
                   map_path=str(Path(source_map).resolve()), map_sha256=digest(Path(source_map)),
                   library_revision=manifest['library_revision'], entries=entries,
                   inventory_sha256=inventory_digest(entries))
    inventory_path(view).write_text(json.dumps(receipt, indent=2) + '\n')
    verify_view(view, atlas, merlin, kernels, mode)
    return view


def verify_view(view, atlas, merlin, kernels, mode='relocated'):
    view = unlinked_path(view)
    inv = inventory_path(view)
    receipt = json.loads(source_file(inv.parent, inv.name).read_text())
    if receipt['roots'] != expected_roots(atlas, merlin, kernels) or receipt['mode'] != mode:
        raise ValueError('view canonical roots or mode differ from request')
    if receipt['view'] != str(view):
        raise ValueError('view location differs from inventory')
    if view.is_symlink() or (view / '.git').exists():
        raise ValueError('top-level view Git metadata or symlink refused')
    if digest(Path(receipt['map_path'])) != receipt['map_sha256']:
        raise ValueError('source map changed')
    manifest = load_map(receipt['map_path'])
    if receipt['library_revision'] != manifest['library_revision']:
        raise ValueError('inventory library revision differs from admitted pin')
    if git(Path(receipt['roots']['atlas-kernels']), 'rev-parse', 'HEAD') != manifest['library_revision']:
        raise ValueError('library revision differs from admitted pin')
    expected = {e['relative_path']: e for e in manifest['entries']}
    entries = receipt['entries']
    if inventory_digest(entries) != receipt['inventory_sha256']:
        raise ValueError('inventory digest differs')
    if len(entries) != len(expected) or {e['relative_path'] for e in entries} != set(expected):
        raise ValueError('inventory membership differs from whitelist')
    for entry in entries:
        spec = expected[entry['relative_path']]
        owner, name = spec['owner'], spec['source']
        if receipt['mode'] == 'baseline' and owner == 'merlin' and not spec.get('new'):
            owner, name = 'atlas-mlir', spec['relative_path']
        source = source_file(Path(receipt['roots'][owner]), name)
        if owner != entry['owner'] or str(source) != entry['source_path']:
            raise ValueError('inventory canonical owner differs')
        if owner == 'atlas-kernels' and entry['sha256'] != pinned_library_digest(
                Path(receipt['roots'][owner]), manifest['library_revision'], name):
            raise ValueError('library source differs from admitted pin: ' + name)
        if digest(source) != entry['sha256']:
            raise ValueError('canonical source digest changed: ' + str(source))
        if digest(source_file(view, entry['relative_path'])) != entry['sha256']:
            raise ValueError('staged source digest changed: ' + entry['relative_path'])
    actual = set()
    for directory, subdirs, files in os.walk(view, followlinks=False):
        for name in subdirs + files:
            checked_name(name)
            child = Path(directory) / name
            if child.is_symlink():
                raise ValueError('view symlink refused')
        subdirs[:] = [n for n in subdirs if not (Path(directory) == view / LIBRARY and n == '.git')]
        for name in files:
            actual.add(str((Path(directory) / name).relative_to(view)))
    if actual != set(expected):
        raise ValueError('view contains missing or unlisted files: ' + repr(sorted(actual ^ set(expected))))
    if git(view / LIBRARY, 'rev-parse', 'HEAD') != receipt['library_revision']:
        raise ValueError('staged library revision changed')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('atlas', 'merlin', 'kernels'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--view', type=Path, required=True)
    parser.add_argument('--mode', choices=('baseline', 'relocated'), default='relocated')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if any(getattr(args, n) is None for n in ('atlas', 'merlin', 'kernels')):
        parser.error('--atlas, --merlin, and --kernels are required')
    if args.verify:
        result = verify_view(args.view, args.atlas, args.merlin, args.kernels, args.mode)
        print(result['inventory_sha256'])
    else:
        print(build_view(args.atlas, args.merlin, args.kernels, args.view, args.mode))


if __name__ == '__main__':
    main()
