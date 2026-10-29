"""Clean-source replay using the selected environment, with no Merlin fallback.

Import policy is harness plumbing only. Every loaded implementation is the exact
copied source; missing files cause a recorded failure rather than a stub/fallback.
No old project proof object is supplied to the ordinary fresh checker.
"""
import argparse
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import traceback


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_root', type=Path)
    parser.add_argument('record', type=Path)
    parser.add_argument('--mode', choices=['prepare', 'check', 'replay'], required=True)
    parser.add_argument('--lean', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    root = args.source_root.resolve()
    record = args.record.resolve()
    record.mkdir(parents=True, exist_ok=False)
    source = root / 'src'
    os.environ['MERLIN_REPO_ROOT'] = str(root)
    os.environ.pop('LEAN_PATH', None)
    sys.dont_write_bytecode = True
    if any(n == 'merlin' or n.startswith('merlin.') for n in sys.modules):
        raise RuntimeError('Merlin already imported before isolation')
    imported = []
    source_manifest = json.loads((root / 'SOURCE-MANIFEST.json').read_text())
    expected_sources = {row['path']: row for row in source_manifest}
    def check_sources():
        for row in source_manifest:
            data = Path(row['path']).read_bytes()
            assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
    check_sources()

    class SourceOnly(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != 'merlin' and not fullname.startswith('merlin.'):
                return None
            choices = [str(source)] if path is None else [p for p in path if Path(p).resolve().is_relative_to(source)]
            spec = importlib.machinery.PathFinder.find_spec(fullname, choices, target)
            if spec is None or spec.origin is None:
                imported.append(dict(module=fullname, status='missing_copied_source', searched=choices))
                raise ModuleNotFoundError('missing exact copied Merlin source: ' + fullname)
            origin = Path(spec.origin).resolve()
            if not origin.is_relative_to(source) or origin.suffix != '.py':
                imported.append(dict(module=fullname, status='refused_nonlocal_source', origin=str(origin)))
                raise ImportError('Merlin fallback refused: ' + fullname)
            imported.append(dict(module=fullname, status='selected', source=str(origin),
                                 sha256=hashlib.sha256(origin.read_bytes()).hexdigest()))
            return spec

    sys.meta_path.insert(0, SourceOnly())
    sys.path.insert(0, str(source))
    outcome = dict(status='started', mode=args.mode, source_root=str(root),
                   python=sys.executable, source_only_copy=True, new_host_qualification=False,
                   old_project_objects_supplied=False,
                   harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   supplemental_source_count=len(source_manifest))
    write(record / 'INTENT.json', outcome)
    try:
        from merlin.verify.buffer_reuse_intake import ALLOCATOR, prepare_chain
        from merlin.verify import buffer_reuse_gate as gate
        cases = []
        for folder in sorted((root / 'cases').iterdir()):
            if not folder.is_dir():
                continue
            raw = tuple((folder / f'stage{i}.mlir').read_text() for i in range(4))
            item = prepare_chain(raw, allocator=ALLOCATOR)
            assert item.query == (folder / 'CanonicalBufferQuery.lean').read_text()
            assert list(item.roots) == json.loads((folder / 'ROOTS.json').read_text())
            assert json.loads(item.query_parts_json) == json.loads((folder / 'PARTS.json').read_text())
            ledger = json.loads(item.sources_json)
            assert {row['path']: row['sha256'] for row in ledger} == {row['path']: row['sha256'] for row in source_manifest}
            cases.append(dict(case=folder.name, source_count=len(ledger), roots=len(item.roots),
                exact_query_parts_roots=True, query_sha256=hashlib.sha256(item.query.encode()).hexdigest(),
                raw4_sha256=[hashlib.sha256(text.encode()).hexdigest() for text in raw]))
        assert len(cases) == 5
        write(record / 'ALL-CASES.json', cases)
        buffers = tuple((root / 'inputs' / f'stage{i}.mlir').read_text() for i in range(4))
        candidate = prepare_chain(buffers, allocator=ALLOCATOR)
        assert candidate.query == (root / 'expected/CanonicalBufferQuery.lean').read_text()
        assert list(candidate.roots) == json.loads((root / 'expected/ROOTS.json').read_text())
        assert json.loads(candidate.query_parts_json) == json.loads((root / 'expected/PARTS.json').read_text())
        refs = json.loads(candidate.sources_json)
        assert len(refs) == 191
        assert {row['path']: row['sha256'] for row in refs} == {row['path']: row['sha256'] for row in source_manifest}
        assert all(Path(row['path']).resolve().is_relative_to(root) for row in refs)
        write(record / 'PREPARATION.json', candidate.describe())
        write(record / 'ACTUAL-LOCAL-SOURCES.json', refs)
        outcome.update(status='prepared', roots=len(candidate.roots), local_source_refs=len(refs), exact_q5_query=True)
        if args.mode != 'prepare':
            assert args.lean is not None
            if args.mode == 'check':
                result = gate.check(candidate, lean=args.lean, destination=record / 'fresh-check', timeout=120)
            else:
                assert args.report is not None
                result = gate.replay(args.report, lean=args.lean, destination=record / 'fresh-replay', timeout=120)
            outcome.update(status=result.status, reason=result.reason)
            if result.status != 'checked' or result.completion is None:
                raise RuntimeError('fresh proof check did not complete: ' + result.reason)
            # replay creates its own live candidate, so obtain only the exact
            # candidate owned by this live completion, not a saved authority.
            selected = candidate if args.mode == 'check' else gate._LIVE[result.completion][0]
            seen = []
            def consumer(actual):
                assert actual == buffers[-1]
                seen.append(hashlib.sha256(actual.encode()).hexdigest())
                return 'exact-final-consumed'
            delivered = gate.deliver(result.completion, selected, consumer)
            assert delivered == 'exact-final-consumed' and len(seen) == 1
            assert gate.validate_completion(result.completion, selected)
            outcome.update(report=result.completion.report, consumer_called=True,
                           final_sha256=seen[0], live_revalidated=True)
        check_sources()
        write(record / 'RESULT.json', outcome)
        print(json.dumps(outcome), flush=True)
        return 0
    except BaseException as error:
        outcome.update(status='failed', error=type(error).__name__, reason=str(error))
        write(record / 'RESULT.json', outcome)
        (record / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        loaded = []
        for name, module in sorted(sys.modules.items()):
            if name == 'merlin' or name.startswith('merlin.'):
                path = Path(module.__file__).resolve()
                assert path.is_relative_to(source) and path.suffix == '.py'
                assert str(path) in expected_sources
                if 'refs' in locals():
                    assert str(path) in {row['path'] for row in refs}
                assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sources[str(path)]['sha256']
                loaded.append(dict(module=name, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        write(record / 'IMPORTS.json', dict(resolutions=imported, loaded=loaded, fallback_permitted=False))


if __name__ == '__main__':
    raise SystemExit(main())
