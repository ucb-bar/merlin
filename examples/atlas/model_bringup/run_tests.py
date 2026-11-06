"""Required model suite and separate source-join gate, executed from a verified view."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

import source_view


class RequiredResult(unittest.TextTestResult):
    def addSkip(self, test, reason):
        self.addFailure(test, (AssertionError, AssertionError('required check skipped: ' + reason), None))


def configure(view, merlin, model):
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(view / p) for p in ('test', 'tools', 'tools/host_iss', 'tools/tiny_full')]
    os.environ['PYTHONPATH'] = os.pathsep.join(sys.path[:4])
    os.environ['MERLIN_ROOT'] = str(merlin)
    os.environ['SMOL_MERLIN'] = str(merlin / 'examples/atlas/comparisons')
    os.environ['SMOL_RUNTIME'] = str(view / 'runtime')
    if model:
        sys.path.insert(4, str(model))
        os.environ['PYTHONPATH'] += os.pathsep + str(model)
    if model:
        for key in ('ATLAS_NPU_MODEL_ROOT', 'ATLAS_EXECUTION_MODEL_ROOT', 'TINY_MODEL_ROOT'):
            os.environ[key] = str(model)


def assert_imports(view, receipt):
    local_names = {Path(e['relative_path']).stem for e in receipt['entries']
                   if e['relative_path'].endswith('.py') and Path(e['relative_path']).stem != '__init__'}
    local_names.update(Path(e['relative_path']).parts[1] for e in receipt['entries']
                       if e['relative_path'].startswith('tools/') and e['relative_path'].endswith('.py')
                       and len(Path(e['relative_path']).parts) > 2)
    canonical = [Path(receipt['roots']['atlas-mlir']),
                 Path(receipt['roots']['merlin']) / 'examples/atlas/model_bringup']
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename or name in ('__main__', 'source_view'):
            continue
        path = Path(filename).absolute()
        # Some installed packages expose synthetic relative __file__ values.
        if not Path(filename).is_absolute() and not path.exists() and name.split('.', 1)[0] not in local_names:
            continue
        if path in (Path(__file__).absolute(), Path(source_view.__file__).absolute()):
            continue
        if any(path.is_relative_to(root) for root in canonical):
            raise AssertionError('import escaped execution view: ' + str(path))
        if name.split('.', 1)[0] in local_names and not path.is_relative_to(view):
            raise AssertionError('import used an unadmitted package: ' + str(path))


def ids(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from ids(test)
        else:
            yield test.id()


def run_file(args, receipt):
    spec = importlib.util.spec_from_file_location(Path(args._test).stem, args.view / args._test)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    membership = list(ids(suite))
    assert_imports(args.view, receipt)
    args.out.write_text(json.dumps(dict(source=args._test, ids=membership, successful=False, status='RUNNING'), indent=2) + '\n')
    result = unittest.TextTestRunner(verbosity=2, resultclass=RequiredResult).run(suite)
    assert_imports(args.view, receipt)
    record = dict(source=args._test, ids=membership, tests_run=result.testsRun,
                  failures=[(t.id(), message) for t, message in result.failures],
                  errors=[(t.id(), message) for t, message in result.errors],
                  successful=result.wasSuccessful())
    args.out.write_text(json.dumps(record, indent=2) + '\n')
    return 0 if result.wasSuccessful() else 1


def run_faithfulness(args, receipt):
    path = args.view / 'tools/check_faithfulness.py'
    spec = importlib.util.spec_from_file_location('check_faithfulness', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original_runner = unittest.TextTestRunner
    record = dict(command='faithfulness', successful=False, ids=[])

    class RecordingRunner(original_runner):
        def run(self, suite):
            record['ids'] = list(ids(suite))
            result = super().run(suite)
            record.update(tests_run=result.testsRun,
                          failures=[(t.id(), message) for t, message in result.failures],
                          errors=[(t.id(), message) for t, message in result.errors])
            return result

    unittest.TextTestRunner = RecordingRunner
    try:
        code = module.main()
        assert_imports(args.view, receipt)
    finally:
        unittest.TextTestRunner = original_runner
    record.update(returncode=code, successful=code == 0)
    args.out.write_text(json.dumps(record, indent=2) + '\n')
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('tests', 'faithfulness'))
    for key in ('view', 'atlas', 'merlin', 'kernels', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--model-root', type=Path)
    parser.add_argument('--codec-model-root', type=Path, help='separate pinned model for test_tiny_full_export')
    parser.add_argument('--mode', choices=('baseline', 'relocated'), default='relocated')
    parser.add_argument('--_test', help=argparse.SUPPRESS)
    parser.add_argument('--timeout', type=float, default=1800, help='maximum seconds per test module (socket regression uses 180)')
    args = parser.parse_args()
    for key in ('view', 'atlas', 'merlin', 'kernels', 'out', 'model_root', 'codec_model_root'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).absolute())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    source_view.test_modules()
    receipt = source_view.verify_view(args.view, args.atlas, args.merlin, args.kernels, args.mode)
    configure(args.view, args.merlin, args.model_root)
    if args._test:
        return run_file(args, receipt)
    if args.command == 'faithfulness':
        code = run_faithfulness(args, receipt)
        source_view.verify_view(args.view, args.atlas, args.merlin, args.kernels, args.mode)
        return code
    manifest = source_view.load_map()
    results = []
    for name in source_view.test_modules(manifest):
        output = args.out.parent / (Path(name).stem + '.json')
        cmd = [sys.executable, str(Path(__file__).absolute()), 'tests', '--_test', name]
        for key in ('view', 'atlas', 'merlin', 'kernels'):
            cmd.extend(['--' + key, str(getattr(args, key))])
        cmd.extend(['--out', str(output), '--mode', args.mode])
        model = args.codec_model_root if name == 'test/test_tiny_full_export.py' and args.codec_model_root else args.model_root
        if model:
            cmd.extend(['--model-root', str(model)])
        start = time.monotonic()
        try:
            limit = min(args.timeout, 180) if name == 'test/test_host_iss_gdb_buffered.py' else args.timeout
            result = subprocess.run(cmd, timeout=limit)
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
            record = json.loads(output.read_text()) if output.exists() else dict(source=name, ids=[])
            record.update(successful=False, status='TIMEOUT', timeout_seconds=limit)
            output.write_text(json.dumps(record, indent=2) + '\n')
        results.append(dict(source=name, returncode=returncode, receipt=str(output),
                            seconds=round(time.monotonic() - start, 3)))
    source_view.verify_view(args.view, args.atlas, args.merlin, args.kernels, args.mode)
    okay = all(row['returncode'] == 0 for row in results)
    args.out.write_text(json.dumps(dict(successful=okay, suites=len(results), results=results), indent=2) + '\n')
    return 0 if okay else 1


if __name__ == '__main__':
    sys.exit(main())
