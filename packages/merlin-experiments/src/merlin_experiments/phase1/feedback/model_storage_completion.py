"""Completed storage custody at the actual trusted model-worker return only."""

import contextlib
import contextvars
import hashlib
import hmac
import json
import subprocess
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.llvmlower.completed_host_storage import _from_trusted_worker, digest, export_completed_storage

FIELD = '_completed_host_storage_return_v1'
_SIGNING_KEY = contextvars.ContextVar('trusted_model_storage_return_key', default=None)


@contextlib.contextmanager
def signing_scope(key):
    if key is not None and (type(key) is not str or len(key) != 64 or len(bytes.fromhex(key)) != 32):
        raise ValueError('completed storage requires the original sealed child key')
    token = _SIGNING_KEY.set(key)
    try:
        yield
    finally:
        _SIGNING_KEY.reset(token)


def _signature(key, payload):
    return hmac.new(bytes.fromhex(key), json.dumps(payload, sort_keys=True, allow_nan=False).encode(),
                    hashlib.sha256).hexdigest()


def implementation_pins():
    import sys

    from merlin.llvmlower import borrowed_arena, completed_host_storage, host_storage_policy
    from merlin.targetgen import owned_host_model_execution, source_composition_preferences

    modules = (borrowed_arena, completed_host_storage, host_storage_policy,
               owned_host_model_execution, source_composition_preferences, sys.modules[__name__])
    return [dict(path=str(path), sha256=sha256_file(path)) for module in modules
            if (path := Path(module.__file__).resolve(strict=True))]


def export_return(result):
    record = result.get('owned_host_native_execution') or {}
    build = record.get('build')
    if not isinstance(build, dict):
        return
    payload = export_completed_storage(build)
    if payload is not None:
        key = _SIGNING_KEY.get()
        if FIELD in result or key is None:
            raise ValueError('completed storage needs the original trusted child signing scope')
        record = dict(schema=FIELD, result_sha256=digest(result), preparation=payload)
        result[FIELD] = {**record, 'signature': _signature(key, record)}


class ModelStorageReturn:
    """Actual Popen owner plus immutable selected worker source closure."""

    def __init__(self, worker_pins, key):
        self.pins = json.loads(json.dumps(worker_pins))
        self.process = self.command = None
        self.key, self.consumed = key, False

    def launched(self, process, command):
        if type(process) is not subprocess.Popen or self.process is not None or list(process.args) != list(command):
            raise ValueError('completed storage needs the original launched subprocess')
        self.process, self.command = process, tuple(command)

    def verify_result(self, result, process):
        payload = result.get(FIELD)
        if payload is None:
            return result
        if (type(process) is not subprocess.Popen or process is not self.process
                or process.returncode != 0 or process.poll() != 0 or tuple(process.args) != self.command
                or self.key is None or self.consumed):
            raise ValueError('completed storage requires the actual successful child return')
        for pin in self.pins:
            path = Path(pin['path'])
            if path.is_symlink() or not path.is_file() or sha256_file(path) != pin['sha256']:
                raise ValueError('completed storage selected worker changed')
        ordinary = {key: value for key, value in result.items() if key != FIELD}
        if (type(payload) is not dict or set(payload) != {'schema', 'result_sha256', 'preparation', 'signature'}
                or payload['schema'] != FIELD or payload['result_sha256'] != digest(ordinary)):
            raise ValueError('completed storage returned result changed')
        unsigned = {key: value for key, value in payload.items() if key != 'signature'}
        if not hmac.compare_digest(payload['signature'], _signature(self.key, unsigned)):
            raise ValueError('completed storage lacks the original trusted worker completion')
        build = ordinary['owned_host_native_execution']['build']
        observed = _from_trusted_worker(build, payload['preparation'], self.pins)
        result['owned_host_native_execution']['build'] = observed
        result.pop(FIELD)
        self.consumed = True
        return result
