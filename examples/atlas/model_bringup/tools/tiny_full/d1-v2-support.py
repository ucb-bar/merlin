"""Pinned reuse of frozen D1 numerical helpers; no source edits or execution on import."""
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
ATLAS_ROOT = HERE.parents[1]
MERLIN_ROOT = Path(os.environ['MERLIN_ROOT']).absolute()
ROOT = Path(os.environ['TINY_WORK_ROOT']).absolute()
ATLAS_HELPERS = ATLAS_ROOT/'tools/tiny_full'
MERLIN_HELPERS = MERLIN_ROOT/'examples/atlas/comparisons'
MODEL_ROOT = Path(os.environ['ATLAS_NPU_MODEL_ROOT']).absolute()



def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe(value):
    path = Path(value).absolute()
    if (not any(path.is_relative_to(root) for root in (ROOT, ATLAS_ROOT, MERLIN_ROOT, MODEL_ROOT)) or '..' in path.parts or '\\' in str(path)
            or any(ord(c) < 32 or ord(c) == 127 for c in str(path))):
        raise ValueError('only physical D paths are allowed')
    for prefix in (*reversed(path.parents), path):
        try:
            mode = prefix.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ValueError('symbolic links are forbidden')
    return path


def frozen():
    paths = [ATLAS_HELPERS/'d1-qualify.py', ATLAS_HELPERS/'d1-model.py',
             MERLIN_HELPERS/'d1-worker.py', MERLIN_HELPERS/'d1-stream-worker.py', MERLIN_HELPERS/'d1-stream-original.py']
    return {str(safe(path)): sha(safe(path).read_bytes()) for path in paths}


def verify(values):
    if not isinstance(values, dict) or not values:
        raise ValueError('missing nonempty source closure')
    for path, digest in values.items():
        if sha(safe(path).read_bytes()) != digest:
            raise ValueError('source or payload pin changed: '+str(path))


def load(name):
    frozen()
    spec = importlib.util.spec_from_file_location('d1_v2_'+name.replace('-', '_'), safe((MERLIN_HELPERS if 'stream' in name or name == 'd1-worker' else ATLAS_HELPERS)/(name+'.py')))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    frozen()
    return module


def adapter_sources():
    names = ('d1-v2-support.py', 'd1-v2-stream-original.py', 'd1-v2-stream-worker.py', 'd1-v2-qualify.py')
    return {**frozen(), **{str(safe((MERLIN_HELPERS if 'stream' in name else ATLAS_HELPERS)/name)): sha(safe((MERLIN_HELPERS if 'stream' in name else ATLAS_HELPERS)/name).read_bytes()) for name in names}}
