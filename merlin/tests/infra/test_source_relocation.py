"""Metadata relocation and target-helper binding controls; no model execution."""

from merlin.common.digest import sha256_bytes
import ast
import copy
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from merlin.common.paths import repo_root

COMPARISONS = repo_root() / 'examples/atlas/comparisons'
sys.path.insert(0, str(COMPARISONS))
sys.path.insert(0, str(COMPARISONS / 'refresh'))
import source_relocation
from source_relocation import SourceRelocation
import capture_resnet
import tiny_workload_full


class RelocationTest(unittest.TestCase):
    def test_mapping_is_pure_and_preserves_digest_version_and_original(self):
        raw = {'path': '/old/tree/input.bin', 'sha256': 'abc', 'versions': {'torch': '2.10'},
               'libraries': {'/old/env/lib.so': {'path': '/old/env/lib.so', 'sha256': 'def'}},
               'nested': ['/old/tree/config.json', 'relative/data', 'https://example.invalid/a']}
        original = copy.deepcopy(raw)
        with patch.object(source_relocation.os, 'lstat', side_effect=AssertionError('filesystem access')):
            relocation = SourceRelocation(['/old=/new', '/old/env=/python'])
            observed = relocation.apply(raw)
        self.assertEqual(raw, original)
        self.assertEqual(observed['path'], '/new/tree/input.bin')
        self.assertEqual(observed['libraries'], {'/python/lib.so': {'path': '/python/lib.so', 'sha256': 'def'}})
        self.assertEqual(observed['sha256'], 'abc')
        self.assertEqual(observed['versions'], raw['versions'])
        self.assertEqual(observed['nested'][1:], raw['nested'][1:])
        self.assertEqual(len(relocation.report()['substitutions']), 3)

    def test_component_prefix_and_one_pass_longest_match(self):
        relocation = SourceRelocation(['/a=/b', '/a/nested=/special', '/b=/c'])
        self.assertEqual(relocation.path('/a/nested/x'), '/special/x')
        self.assertEqual(relocation.path('/a/x'), '/b/x')
        self.assertEqual(relocation.path('/ab/x'), '/ab/x')
        self.assertEqual(relocation.path('/a'), '/b')

    def test_duplicate_prefix_and_key_collision_refused(self):
        with self.assertRaises(ValueError):
            SourceRelocation(['/a=/x', '/a/=/y'])
        with self.assertRaises(ValueError):
            SourceRelocation(['/a=/x', '/b=/x']).apply({'/a/k': 1, '/b/k': 2})
        with self.assertRaises(ValueError):
            SourceRelocation(['/a=/x']).apply({'/a/k': 1, '/x/k': 2})

    def test_unsafe_paths_refused_without_filesystem(self):
        for bad in ('relative', '/old/../escape'):
            with self.subTest(bad=bad), patch.object(source_relocation.os, 'lstat', side_effect=AssertionError(
                'filesystem access')):
                with self.assertRaises(ValueError):
                    SourceRelocation([bad+'=/safe'])
                with self.assertRaises(ValueError):
                    SourceRelocation(['/safe='+bad])
                if bad.startswith('/'):
                    with self.assertRaises(ValueError):
                        SourceRelocation().apply({'path': bad})

    def test_metadata_read_binds_raw_bytes_before_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'receipt.json'
            raw = json.dumps({'path': '/old/input.bin', 'sha256': 'f'*64, 'version': 'frozen'}).encode()
            path.write_bytes(raw)
            digest = sha256_bytes(raw)
            relocation = SourceRelocation(['/old=/new'])
            observed, identity = relocation.read_json(path, digest)
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(identity, dict(path=str(path), sha256=digest, bytes=len(raw)))
            self.assertEqual(observed, {'path': '/new/input.bin', 'sha256': 'f'*64, 'version': 'frozen'})
            self.assertEqual(relocation.report()['raw_metadata'], [identity])
            path.write_bytes(raw+b' ')
            with self.assertRaises(ValueError):
                relocation.read_json(path, digest)

    def test_wrong_digest_prevents_parse_or_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'receipt.json'
            path.write_text('not JSON')
            relocation=SourceRelocation()
            with patch.object(relocation, 'apply', side_effect=AssertionError('mapped before admission')):
                with self.assertRaises(ValueError):
                    relocation.read_json(path, '0'*64)
            for digest in ('', 'x'*64, '0'*63):
                with patch.object(source_relocation.os, 'lstat', side_effect=AssertionError('filesystem access')):
                    with self.assertRaises(ValueError):
                        relocation.read_json(path, digest)

    def test_metadata_symlink_refused_before_target_descent(self):
        visited=[]
        def lstat(path):
            visited.append(str(path))
            return SimpleNamespace(st_mode=stat.S_IFLNK if path.name=='link' else stat.S_IFDIR)
        with patch.object(source_relocation.os,'lstat',side_effect=lstat), patch.object(Path,'read_bytes',
            side_effect=AssertionError('read link')):
            with self.assertRaises(ValueError):
                SourceRelocation().read_json('/not-opened/link/child.json','0'*64)
        self.assertEqual(visited,['/not-opened','/not-opened/link'])

    def test_parent_metadata_path_refused_before_stat(self):
        with patch.object(source_relocation.os,'lstat',side_effect=AssertionError('filesystem access')):
            for path in ('/not-opened/../x',):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    SourceRelocation().read_json(path,'0'*64)

    def test_sibling_reader_shares_context_and_keeps_raw_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'metadata.json'
            raw=b'{"path":"/old/data","sha256":"fixed"}'
            path.write_bytes(raw)
            args=SimpleNamespace(relocate=['/old=/new'])
            report={}
            observed=capture_resnet.read_source_metadata(args,report,path,sha256_bytes(raw))
            self.assertEqual(observed['path'],'/new/data')
            self.assertIs(capture_resnet.source_relocator(args),args._source_relocation)
            self.assertEqual(report['source_relocation']['raw_metadata'][0]['sha256'],sha256_bytes(raw))

    def test_frozen_resnet_hash_policy_not_weakened(self):
        args=SimpleNamespace(receipt_sha256='0'*64,preparation_sha256=capture_resnet.PREPARATION_SHA256)
        with patch.object(SourceRelocation,'read_json',side_effect=AssertionError('read unadmitted metadata')):
            with self.assertRaises(ValueError):
                capture_resnet.capture(args,{})


class ArithmeticImportTest(unittest.TestCase):
    def test_explicit_target_helper_loaded_and_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'external_arithmetic.py'
            path.write_text('def ordered_matmul(a,b): return (a,b)\ndef rescale_bf16_array(a,e): return (a,e)\n')
            digest=sha256_bytes(path.read_bytes())
            records=[]
            module=tiny_workload_full.load_arithmetic(path,digest,records)
            self.assertEqual(module.ordered_matmul('a','b'),('a','b'))
            self.assertEqual(records[0]['path'],str(path))
            self.assertEqual(records[0]['sha256'],digest)

    def test_wrong_helper_hash_refused_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'external_arithmetic.py'
            path.write_text('raise RuntimeError("must not execute")\n')
            with self.assertRaises(ValueError):
                tiny_workload_full.load_arithmetic(path,'0'*64,[])

    def test_missing_target_api_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'external_arithmetic.py'
            path.write_text('def ordered_matmul(a,b): return a\n')
            with self.assertRaises(ValueError):
                tiny_workload_full.load_arithmetic(path,sha256_bytes(path.read_bytes()),[])


class ReplayRelocationTest(unittest.TestCase):
    def test_generic_reader_shares_raw_identity_and_installed_library_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'
            raw = b'{"libraries":{"/old/env/lib.so":{"sha256":"fixed"}},"path":"/old/input"}'
            path.write_bytes(raw)
            args = SimpleNamespace(relocate=['/old=/new'])
            report = {}
            observed = source_relocation.read_metadata(args, report, path, sha256_bytes(raw))
            self.assertIn('/new/env/lib.so', observed['libraries'])
            self.assertEqual(report['source_relocation']['raw_metadata'][0]['sha256'], sha256_bytes(raw))
            self.assertIs(source_relocation.argument_relocator(args), args._source_relocation)

    def test_smol_validator_receives_mapper_before_admission(self):
        import evaluate_smolvla_attention as attention
        import evaluate_smolvla_retained_prefix as retained
        class BoundaryReached(Exception):
            pass
        args = SimpleNamespace(producer_root=Path('/selected/provider'), validator_sha256='v',
                               provider_receipt=Path('/selected/receipt.json'), provider_receipt_sha256='r',
                               native_bundle=Path('/selected/native'), native_manifest_sha256='n',
                               model_root=Path('/selected/model'), relocate=['/original=/relocated'])
        calls = []
        def admit(**kwargs):
            calls.append(kwargs)
            self.assertEqual(kwargs['metadata_path_mapper']('/original/model/npu_model/__init__.py'),
                             '/relocated/model/npu_model/__init__.py')
            raise BoundaryReached
        validator = SimpleNamespace(admit_contraction=admit)
        for module in (attention, retained):
            with self.subTest(module=module.__name__), patch.object(module, 'load_provider_validator', return_value=(validator, {
                })):
                with self.assertRaises(BoundaryReached):
                    module.evaluate(args, {})
        self.assertEqual(len(calls), 2)

    def test_all_delegated_provider_calls_use_read_boundary_mapper(self):
        names = ('refresh/evaluate_resnet.py',
                 'evaluate_smolvla_attention.py', 'evaluate_smolvla_retained_prefix.py')
        count = 0
        for name in names:
            for node in ast.walk(ast.parse((COMPARISONS / name).read_text())):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'admit_contraction':
                    self.assertIn('metadata_path_mapper', {keyword.arg for keyword in node.keywords}, name)
                    count += 1
        # ResNet conv2/conv3 share one admission boundary.
        self.assertEqual(count, 4)

    def test_selected_replay_accepts_candidate_plan_without_source_patching(self):
        import inspect
        import evaluate_smolvla_retained_prefix as retained
        self.assertIn('candidate_plan', inspect.signature(retained.evaluate).parameters)


class ReportLifecycleTest(unittest.TestCase):
    def test_invalid_relocation_keeps_failure_receipt_without_running_callback(self):
        import workload_cli
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed"
            args = SimpleNamespace(output_dir=output, relocate=["invalid"])
            with patch("capture_tinyllama_native.artifact_output", return_value=output), \
                    patch.object(sys, "argv", [str(COMPARISONS / "workload_cli.py")]), \
                    patch.object(workload_cli.sys, "dont_write_bytecode", False), \
                    patch("builtins.print"), \
                    patch.object(workload_cli, "argument_relocator", wraps=source_relocation.argument_relocator):
                callback = unittest.mock.Mock()
                self.assertEqual(workload_cli.run_report(args, callback, filename="receipt.json", claim="test"), 1)
                callback.assert_not_called()
                self.assertTrue(sys.dont_write_bytecode)
            receipt = json.loads((output / "receipt.json").read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["error"]["type"], "ValueError")

    def test_successful_report_records_shared_helper_bookends(self):
        import workload_cli
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "success"
            args = SimpleNamespace(output_dir=output, relocate=[])
            with patch("capture_tinyllama_native.artifact_output", return_value=output), \
                    patch.object(sys, "argv", [str(COMPARISONS / "workload_cli.py")]), \
                    patch.object(workload_cli.sys, "dont_write_bytecode", False), patch("builtins.print"):
                code = workload_cli.run_report(args, lambda args, report: report.update(status="complete"),
                                               filename="receipt.json", claim="test")
            self.assertEqual(code, 0)
            receipt = json.loads((output / "receipt.json").read_text())
            self.assertEqual(receipt["report_helper_bookends"][0]["sha256"],
                             sha256_bytes((COMPARISONS / "workload_cli.py").read_bytes()))


if __name__=='__main__':
    unittest.main()
