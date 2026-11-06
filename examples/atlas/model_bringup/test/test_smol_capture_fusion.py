"""Capture reply reuse and explicit development backend controls."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).absolute().parents[1]/'tools'))
from smol_capture import capture


class FusionCaptureControls(unittest.TestCase):
    def test_capture_binds_actual_package_init_source(self):
        expected=capture.PACKAGE/'__init__.py'
        self.assertIn(expected,capture.capture_source_paths())
        with patch.object(capture.smol_capture,'__file__','/synthetic/foreign/__init__.py'):
            with self.assertRaisesRegex(ValueError,'package imported outside selected source'):
                capture.capture_source_paths()

    def test_actual_reused_reply_persists_then_resumes_with_fp8_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'launch';folder.mkdir()
            initial,program,output=b'abcd',b'runtime',b'xy'
            source=root/'prior-output.bin';source.write_bytes(output)
            dma=dict(load_bytes=2,store_bytes=2)
            stats=dict(scale_exponent=-5,physical_values=2,preclip_saturation_count=1,
                preclip_nonfinite_count=0,preclip_nan_count=0,preclip_max_abs=16.0,emitted_nan_count=0)
            event=dict(index=0,sequence=1,kind='projection',output_offset=1,output_bytes=2,
                initial=dict(sha256=capture.digest(initial),bytes=4),
                runtime_program=dict(sha256=capture.digest(program),bytes=len(program)),
                output=dict(path=str(source),sha256=capture.digest(output),bytes=2),
                expected_arena_sha256=capture.digest(b'axyd'),observed_dma=dma,requested_dma=dma,
                provenance=dict(backend='old-original'),fp8_stats=stats)
            args=(event,folder,0,1,'projection',1,2,initial,program,dma)
            result=capture.reuse_execution(*args,output_dtype='fp8',output_scale_exponent=-5)
            persisted=json.loads((folder/'execution.json').read_text())
            self.assertEqual(persisted,result)
            self.assertEqual((folder/'arena-expected.bin').read_bytes(),b'axyd')
            self.assertEqual((folder/'output.bin').read_bytes(),output)
            # Capture consumes transient worker files before a durable-prefix resume.
            final=folder/'expected-output.bin';final.write_bytes(output)
            (folder/'output.bin').unlink();(folder/'arena-expected.bin').unlink()
            cached=dict(event,output=dict(event['output'],path=str(final)),provenance=result['provenance'])
            resumed=capture.resume_execution(cached,folder)
            self.assertEqual(resumed['fp8_stats'],stats)
            self.assertEqual(Path(resumed['output_path']).read_bytes(),output)
            self.assertEqual(resumed['expected_arena_sha256'],capture.digest(b'axyd'))
            cached['fp8_stats']=dict(stats,preclip_saturation_count=0)
            with self.assertRaisesRegex(ValueError,'FP8 observation'):
                capture.resume_execution(cached,folder)
            for change in ({'fp8_stats':None},{'fp8_stats':dict(stats,scale_exponent=-4)},
                           {'fp8_stats':dict(stats,physical_values=1)}):
                bad=dict(event,**change)
                if bad['fp8_stats'] is None:bad.pop('fp8_stats')
                with self.assertRaisesRegex(ValueError,'FP8 observation'):
                    capture.reuse_execution(bad,*args[1:],output_dtype='fp8',output_scale_exponent=-5)

    def test_worker_exit_failure_is_fatal_after_replies(self):
        worker=capture.Worker.__new__(capture.Worker)
        worker.log=Mock();worker.process=Mock()
        for code in (1,-9,False,None):
            worker.process.poll.return_value=code
            worker.process.wait.return_value=code
            with self.assertRaisesRegex(ValueError,'exit was not successful'):
                worker.close()
        worker.process.poll.return_value=0
        worker.close()

    def test_reply_reuse_requires_every_current_launch_binding(self):
        arena,program=b'complete staged arena',b'actual runtime stream'
        dma=dict(load_bytes=12,store_bytes=4)
        event=dict(index=7,sequence=8,kind='projection',output_offset=12,output_bytes=4,
            initial=dict(sha256=capture.digest(arena),bytes=len(arena)),
            runtime_program=dict(sha256=capture.digest(program),bytes=len(program)),
            observed_dma=dict(dma),requested_dma=dict(dma))
        args=(7,8,'projection',12,4,arena,program,dma)
        self.assertTrue(capture.reply_matches(event,*args))
        for field,value in [('index',6),('sequence',9),('kind','attention'),('output_offset',8),('output_bytes',8)]:
            changed=copy.deepcopy(event);changed[field]=value
            self.assertFalse(capture.reply_matches(changed,*args),field)
        for field in ('initial','runtime_program'):
            for member,value in [('sha256','0'*64),('bytes',0)]:
                changed=copy.deepcopy(event);changed[field][member]=value
                self.assertFalse(capture.reply_matches(changed,*args),(field,member))
        for field in ('observed_dma','requested_dma'):
            changed=copy.deepcopy(event);changed[field]['store_bytes']=8
            self.assertFalse(capture.reply_matches(changed,*args),field)
        self.assertFalse(capture.reply_matches(event,*args[:-1],dict(load_bytes=16,store_bytes=4)))

    def test_default_worker_removes_inherited_math_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            process=Mock();process.poll.return_value=0
            with patch.object(capture,'EXECUTION_OPTIONS',{}), \
                 patch.object(capture,'MODEL_ROOT',Path('/synthetic/original')), \
                 patch.object(capture,'MODEL_PIN','original-pin'), \
                 patch.dict(capture.os.environ,{'NPU_MODEL_RTL_NATIVE_MXU0':'1','PYTHONPATH':'bad'}), \
                 patch.object(capture.subprocess,'Popen',return_value=process) as launch:
                worker=capture.Worker('projection',Path(directory));worker.close()
            command=launch.call_args.args[0];env=launch.call_args.kwargs['env']
            self.assertEqual(env['ATLAS_NPU_MODEL_ROOT'],'/synthetic/original')
            self.assertEqual(env['ATLAS_NPU_MODEL_PIN'],'original-pin')
            self.assertNotIn('NPU_MODEL_RTL_NATIVE_MXU0',env)
            self.assertNotIn('PYTHONPATH',env)
            self.assertNotIn('--native-admission-root',command)

    def test_optional_backends_are_isolated_by_worker(self):
        options=dict(native_projection_model=Path('/synthetic/native'),native_build_dir=Path('/synthetic/build'),
            native_admission_root=Path('/synthetic/catalog'),native_admission_sha256='a'*64,
            fast_attention_admission=Path('/synthetic/attention'))
        with tempfile.TemporaryDirectory() as directory:
            process=Mock();process.poll.return_value=0
            with patch.object(capture,'EXECUTION_OPTIONS',options), \
                 patch.object(capture,'MODEL_ROOT',Path('/synthetic/original')), \
                 patch.object(capture,'MODEL_PIN','original-pin'), \
                 patch.object(capture.subprocess,'Popen',return_value=process) as launch:
                for backend in ('projection','attention'):
                    worker=capture.Worker(backend,Path(directory));worker.close()
                    command=launch.call_args.args[0];env=launch.call_args.kwargs['env']
                    if backend=='projection':
                        self.assertEqual(env['ATLAS_NPU_MODEL_ROOT'],'/synthetic/native')
                        self.assertEqual(env['NPU_MODEL_RTL_NATIVE_MXU0'],'1')
                        self.assertIn('--native-admission-root',command)
                        self.assertNotIn('--fast-attention-admission',command)
                    else:
                        self.assertEqual(env['ATLAS_NPU_MODEL_ROOT'],'/synthetic/original')
                        self.assertNotIn('NPU_MODEL_RTL_NATIVE_MXU0',env)
                        self.assertIn('--fast-attention-admission',command)
                        self.assertNotIn('--native-admission-root',command)


if __name__=='__main__':unittest.main()
