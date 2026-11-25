"""Fixed host-math ABI and shared supplier policy for emitters and evaluators.

Import this module directly by file path if Merlin's optional dependencies are
not installed. No compiler lowering or framework kernels are used here.
"""
import ctypes
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POLICY_PATH = ROOT / 'host-unary-math-policy.json'
HEADER_PATH = ROOT / 'host_math.h'
FUNCTIONS = ('expf', 'sinf', 'cosf', 'sqrtf', 'tanhf', 'erff', 'sin', 'cos', 'powf', 'pow')
FUNCTIONS64 = ('sin', 'cos', 'pow')
BINARY_FUNCTIONS = ('powf', 'pow')


def arity(name):
    return 2 if name in BINARY_FUNCTIONS else 1


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policy(path=POLICY_PATH, required=()):
    """Validate policy and pinned local sources; reject unavailable operations."""
    value = json.loads(Path(path).read_text())
    fixed = {'schema': 'merlin.host_math.v1', 'interface': 'merlin-host-math-v1',
             'library': 'core-math', 'rounding': 'rne', 'subnormals': 'preserve',
             'nan': 'canonical-qnan-f32-0x7fc00000-f64-0x7ff8000000000000', 'effects': 'errno-and-fp-flags-unobserved'}
    if set(value) != set(fixed) | {'revision', 'required_functions', 'functions', 'support_sources'}:
        raise ValueError('unknown or missing host-math policy fields')
    if any(value.get(k) != v for k, v in fixed.items()):
        raise ValueError('unsupported host-math policy')
    if value['required_functions'] != list(FUNCTIONS):
        raise ValueError('host-math interface function set differs')
    vendor = ROOT / 'third_party/core-math'
    if value['revision'] != (vendor / 'REVISION').read_text().strip():
        raise ValueError('CORE-MATH revision differs')
    if not isinstance(value['functions'], dict) or not value['functions']:
        raise ValueError('host-math supplier has no functions')
    for name, spec in value['functions'].items():
        if name not in FUNCTIONS or not isinstance(spec, dict) or set(spec) != {'symbol', 'implementation', 'source', 'sha256'}:
            raise ValueError('invalid host-math function entry')
        implementation, source = ('merlin_ieee_sqrtf', 'hardware_sqrt.c') if name == 'sqrtf' else ('cr_'+name, 'pow/pow.c' if name == 'pow' else name+'.c')
        if (spec['symbol'], spec['implementation'], spec['source']) != ('merlin_host_'+name, implementation, source):
            raise ValueError('invalid host-math supplier mapping')
        source_root = ROOT if name == 'sqrtf' else vendor
        if sha256(source_root / spec['source']) != spec['sha256']:
            raise ValueError('CORE-MATH source hash differs: '+name)
    if value['support_sources'] != {name: sha256(vendor / name) for name in ('pow/pow.h', 'pow/dint.h', 'pow/qint.h')}:
        raise ValueError('CORE-MATH support source hash differs')
    missing = set(required) - value['functions'].keys()
    if missing:
        raise ValueError('unavailable host-math functions: '+', '.join(sorted(missing)))
    return value


def symbol(name, policy_path=POLICY_PATH):
    return load_policy(policy_path, [name])['functions'][name]['symbol']


class HostMath:
    """Independent evaluator binding. Archive receipt authenticates shared object.

    Caller must retain default RNE and gradual underflow; no flags/errno observed.
    Inputs/outputs to call_bits use raw binary32/binary64 words, avoiding Python NaN casts.
    """
    def __init__(self, build_dir, policy_path=POLICY_PATH, required=FUNCTIONS):
        self.policy = load_policy(policy_path, required)
        build_dir = Path(build_dir)
        receipt = json.loads((build_dir / 'build-receipt.json').read_text())
        if receipt['target'] != 'x86' or receipt['policy_sha256'] != sha256(policy_path):
            raise ValueError('evaluator build policy/target differs')
        if receipt.get('support_sources_sha256') != self.policy['support_sources']:
            raise ValueError('evaluator support source hash differs')
        if receipt.get('support_header_sha256') != sha256(ROOT / 'supplier_support.h'):
            raise ValueError('evaluator support header hash differs')
        library = build_dir / 'libmerlin-host-math.so'
        if sha256(library) != receipt['shared_sha256'] or sha256(HEADER_PATH) != receipt['header_sha256']:
            raise ValueError('evaluator library/header hash differs')
        self.library = ctypes.CDLL(str(library.resolve()))
        self.check_env = self.library.merlin_host_check_fp_env
        self.check_env.argtypes, self.check_env.restype = [], ctypes.c_int
        self._check_fp_env()
        self.calls = {}
        self.bits = {}
        for name in self.policy['functions']:
            fn = getattr(self.library, symbol(name, policy_path))
            scalar = ctypes.c_double if name in FUNCTIONS64 else ctypes.c_float
            word = ctypes.c_uint64 if name in FUNCTIONS64 else ctypes.c_uint32
            fn.argtypes, fn.restype = [scalar] * arity(name), scalar
            self.calls[name] = fn
            fn_bits = getattr(self.library, 'merlin_host_bits_'+name)
            fn_bits.argtypes, fn_bits.restype = [word] * arity(name), word
            self.bits[name] = fn_bits

    def _check_fp_env(self):
        if self.check_env() != 1:
            raise RuntimeError('host math requires RNE and gradual underflow (FTZ/DAZ off)')

    def call(self, name, *args):
        self._check_fp_env()
        if name not in self.calls or len(args) != arity(name):
            raise ValueError('unavailable host-math function: '+name)
        return self.calls[name](*args)

    def call_bits(self, name, *words):
        self._check_fp_env()
        limit = (1 << (64 if name in FUNCTIONS64 else 32)) - 1
        if (name not in self.bits or len(words) != arity(name)
                or any(not isinstance(word, int) or not 0 <= word <= limit for word in words)):
            raise ValueError('unavailable function or invalid floating-point word')
        return self.bits[name](*words)
