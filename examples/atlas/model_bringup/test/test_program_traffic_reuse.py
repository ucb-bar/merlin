"""Admission, source closure, and default compatibility for opt-in traffic reuse."""
import hashlib
import importlib
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import kernel_library  # noqa: E402,F401
import atlas_program as program

# Captured from the unmodified generator before introducing the opt-in path.
GOLDEN = [{'mkn': [32, 32, 32],
  'exponent': None,
  'flags': {},
  'sha256': '417148a962e510c0ffd519bc7b180b1384c61490b91d1c09708fc3e11d472e1c'},
 {'mkn': [64, 64, 64],
  'exponent': -19,
  'flags': {'resident_b': True},
  'sha256': '00adb37f615953037d555af81e50f474dd194c4b1d846be6c3d0f90725f88a32'},
 {'mkn': [64, 64, 64],
  'exponent': -19,
  'flags': {'resident_b': True, 'pretranspose_resident_b': True},
  'sha256': 'eb3b4604b30b312b89144fd40d13aeea1aebe5cba7371f47a2bf2afa6c2a78c4'},
 {'mkn': [128, 768, 768],
  'exponent': -19,
  'flags': {'resident_a': True},
  'sha256': '18adfb148517cf06d814d279abe5ecebd69581ac8f5ea4af4cddb07fed01eb9b'},
 {'mkn': [128, 2048, 2048],
  'exponent': -14,
  'flags': {'resident_a': True},
  'sha256': '295cf8b87a435c281671fdc066c89289c4eb2ec5452632aaf1e2157c91d848f0'},
 {'mkn': [128, 3072, 768],
  'exponent': -15,
  'flags': {},
  'sha256': 'dce0cda320332da92c77f6bfed92affa6fd58c79d3948bc7dd3caad4fce5f6e2'}]

class TrafficReuseTests(unittest.TestCase):
    def test_default_assembly_is_unchanged(self):
        for case in GOLDEN:
            m, k, n = case["mkn"]
            args = (m, k, n, m*k+k*n+256, case["exponent"])
            with self.subTest(case=case):
                for option in ({}, {"traffic_reuse": False}):
                    text = program.assembly(*args, **case["flags"], **option)
                    self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), case["sha256"])
                flags = {key: value for key, value in case["flags"].items() if key != "pretranspose_resident_b"}
                old_loads = (m*k*(n//32)+k*n if flags.get("resident_b") else
                             m*k+k*n*(m//32) if flags.get("resident_a") else
                             2*m*k*(n//32))
                self.assertEqual(program.dma_load_bytes(m, k, n, **flags), old_loads)
                self.assertEqual(program.dma_load_bytes(m, k, n, **flags, traffic_reuse=False), old_loads)

    def test_exact_dispatch_and_recipe_loads(self):
        for dims, name in program._TRAFFIC_REUSE_MODULES.items():
            m, k, n = dims
            helper = importlib.import_module(name)
            for exponent in (-126, -19, 0, 127):
                with self.subTest(dims=dims, exponent=exponent):
                    args = (*dims, m*k+k*n+256, exponent)
                    self.assertEqual(program.assembly(*args, traffic_reuse=True), helper.assembly(*args))
            expected = {(64,12288,960): 24379392, (128,5632,1024): 12255232, (32,2048,2048): 4259840,
                        (224,2304,256): 1695744, (64,4608,512): 2654208,
                        (800,1152,128): 1216512,
                        (224, 256, 1024): 319488,
                        (3136, 576, 64): 1953792,
                        (3136, 64, 256): 217088,
                        (224, 1024, 256): 491520,
                        (800, 128, 512): 167936,
                        (64, 512, 2048): 1081344,
                        (800, 512, 128): 475136,
                        (12544, 160, 64): 2048000,
                        (64, 2048, 512): 1179648,
                        (64, 1024, 2048): 2162688,
                        (224, 512, 1024): 638976,
                        (224, 1024, 512): 753664,
                        (800, 256, 512): 335872,
                        (3136, 256, 64): 835584,
                        (3136, 256, 128): 868352,
                        (800, 512, 256): 540672,
                        (3136, 64, 64): 204800,
                        (32, 2048, 1024): 2162688,
                        (192, 2560, 960): 5406720}.get(dims, m*k+k*n)
            self.assertEqual(program.dma_load_bytes(*dims, traffic_reuse=True), expected)

    def test_reject_unsupported_dimensions_and_legacy_flags(self):
        for dims in ((32,32,32), (0,768,768), (True,768,768), (128.0,768,768), (128,769,768), (128,768,-768)):
            with self.subTest(dims=dims):
                with self.assertRaises(ValueError):
                    program.assembly(*dims, 688384, -19, traffic_reuse=True)
                with self.assertRaises(ValueError):
                    program.dma_load_bytes(*dims, traffic_reuse=True)
        for flag in ("resident_a", "resident_b", "pretranspose_resident_b"):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                program.assembly(128,768,768,688384,-19,traffic_reuse=True,**{flag:True})
            if flag != "pretranspose_resident_b":
                with self.assertRaises(ValueError):
                    program.dma_load_bytes(128,768,768,traffic_reuse=True,**{flag:True})

    def test_reject_abi_or_rescale_changes(self):
        for offset in (688383,688385,688384.0,True):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                program.assembly(128,768,768,offset,-19,traffic_reuse=True)
        for exponent in (None,True,0.0,float("inf"),float("nan"),-127,128):
            with self.subTest(exponent=exponent), self.assertRaises(ValueError):
                program.assembly(128,768,768,688384,exponent,traffic_reuse=True)

    def test_flag_is_boolean(self):
        for flag in (0,1,None,"",[],{}):
            with self.subTest(flag=flag):
                with self.assertRaises(ValueError):
                    program.assembly(128,768,768,688384,-19,traffic_reuse=flag)
                with self.assertRaises(ValueError):
                    program.dma_load_bytes(128,768,768,traffic_reuse=flag)
                with self.assertRaises(ValueError):
                    program.code_sources(traffic_reuse=flag)

    def test_source_closure_tracks_all_helpers_only_when_enabled(self):
        legacy = {"atlas_program.py", "run_model_tile.py", "atlas_tile_arithmetic.py", "atlas_epilogue.py", "atlas_rne_add.py",
                  "atlas_resnet_input.py", "atlas_program_checks.py", "atlas_program_reuse_common.py", "atlas_weight_source.py"}
        helpers = {name+".py" for name in program._TRAFFIC_REUSE_MODULES.values()}
        expanded = legacy | helpers
        self.assertEqual(set(kernel_library.code_sources()), legacy)
        self.assertEqual(set(kernel_library.code_sources(traffic_reuse=False)), legacy)
        self.assertEqual(set(kernel_library.code_sources(traffic_reuse=True)), expanded)
        with tempfile.TemporaryDirectory(prefix="atlas-reuse-sources-") as tmp:
            directory = Path(tmp)
            for name in expanded:
                shutil.copyfile(kernel_library.source_path(name), directory / name)
            spec = importlib.util.spec_from_file_location("copied_atlas_program", directory/"atlas_program.py")
            copied = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(copied)
            before = copied.code_sources(traffic_reuse=True)
            for name, digest in before.items():
                self.assertEqual(digest, hashlib.sha256((directory/name).read_bytes()).hexdigest())
            for name in sorted(helpers):
                path = directory/name
                original = path.read_bytes()
                path.write_bytes(original+b"\n# source closure probe\n")
                after = copied.code_sources(traffic_reuse=True)
                self.assertEqual({key for key in before if before[key] != after[key]}, {name})
                self.assertEqual(copied.code_sources(), kernel_library.code_sources())
                path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
