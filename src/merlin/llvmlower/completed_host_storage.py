"""Custody of an already completed storage preparation, never allocator effects.

Only the ordinary live recheck can create a child completion. A trusted worker
return may consume it in the parent for those exact products. Saved JSON cannot
restore either kind of observation or authorize another allocation/rewrite.
"""

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from merlin.common.digest import sha256_file

from .borrowed_arena import receipt_borrowed_lifetime

_SEAL = object()


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _check_pins(pins):
    for pin in pins:
        path = Path(pin['path'])
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or sha256_file(path) != pin['sha256']):
            raise ValueError('completed host storage source or product changed')


@dataclass(frozen=True, init=False, eq=False, repr=False)
class _ChildCompletion:
    def __init__(self, seal, receipt, policy, artifacts, native):
        if seal is not _SEAL:
            raise TypeError('completed storage requires its ordinary live recheck')
        from .host_storage_policy import _sources

        object.__setattr__(self, 'native', native)
        object.__setattr__(self, 'policy', policy)
        source = native._selection
        pins = ((source.source_path, source.source_sha256), *source.pins,
                *(source._plural_sources or ()), *native._pins)
        products = {pin['path']: pin for pin in artifacts.values()}
        for path, sha in pins:
            pin = dict(path=str(path.resolve()), sha256=sha)
            if pin['path'] in products and products[pin['path']]['sha256'] != sha:
                raise ValueError('completed host storage has conflicting original pins')
            products[pin['path']] = pin
        object.__setattr__(self, 'payload', _json(dict(
            schema='merlin.completed_host_storage.v1', build_sha256=digest(receipt),
            policy=policy.record(), artifacts=artifacts, pins=list(products.values()),
            implementation_sources=_sources(),
        )))
        object.__setattr__(self, '_state', (self.native, self.policy, self.payload))

    def export(self, receipt):
        if any(actual is not original for actual, original in zip(
                (self.native, self.policy, self.payload), self._state, strict=True)):
            raise ValueError('original completed storage observation changed')
        self.native._check()
        self.policy.revalidate()
        data = json.loads(self.payload)
        if digest(receipt) != data['build_sha256'] or json.loads(_json(self.policy.record())) != data['policy']:
            raise ValueError('completed host storage receipt or original policy changed')
        _check_pins(data['pins'])
        return data


def remember_completed_storage(receipt, policy, artifacts):
    """Called only after the normal original-owner storage recheck succeeded."""
    native = receipt_borrowed_lifetime(receipt)
    if native is not None:
        receipt._completed_host_storage = _ChildCompletion(_SEAL, receipt, policy, artifacts, native)


def export_completed_storage(receipt):
    completion = getattr(receipt, '_completed_host_storage', None)
    if type(completion) is not _ChildCompletion:
        return None
    return completion.export(receipt)


class _ReturnedReceipt(dict):
    def __init__(self, record=(), *, observation=None):
        super().__init__(record)
        self._observation = observation

    def __copy__(self):
        return dict(self)

    def __deepcopy__(self, memo):
        return deepcopy(dict(self), memo)

    def __reduce_ex__(self, protocol):
        return dict, (dict(self),)


@dataclass(frozen=True, init=False, eq=False, repr=False)
class CompletedHostStorage:
    """Parent-only proof of one trusted child's completed exact preparation.

    There is deliberately no borrowed-call/allocator permission API here.
    """

    def __init__(self, seal, payload, worker_pins):
        if seal is not _SEAL:
            raise TypeError('completed storage needs an actual trusted worker return')
        object.__setattr__(self, '_payload', _json(payload))
        object.__setattr__(self, '_worker_pins', _json(worker_pins))
        object.__setattr__(self, '_state', (self._payload, self._worker_pins))

    def recheck(self, directory, policy, receipt):
        from .host_storage_policy import _sources

        if self._payload is not self._state[0] or self._worker_pins is not self._state[1]:
            raise ValueError('original completed storage observation changed')
        data = json.loads(self._payload)
        policy.revalidate()
        _check_pins(json.loads(self._worker_pins))
        _check_pins(data['pins'])
        if (data.get('schema') != 'merlin.completed_host_storage.v1'
                or digest(receipt) != data['build_sha256']
                or json.loads(_json(policy.record())) != data['policy']
                or data['implementation_sources'] != _sources()
                or Path(data['artifacts']['report']['path']).parent != Path(directory).resolve()):
            raise ValueError('completed host storage differs from the original exact preparation')
        return data['artifacts']


def receipt_completed_storage(receipt):
    return receipt._observation if type(receipt) is _ReturnedReceipt else None


def _from_trusted_worker(receipt, payload, worker_pins):
    """Private consumer of the original subprocess transport's verified return."""
    if payload.get('schema') != 'merlin.completed_host_storage.v1' or digest(receipt) != payload['build_sha256']:
        raise ValueError('trusted child returned another completed storage receipt')
    _check_pins(worker_pins)
    _check_pins(payload['pins'])
    return _ReturnedReceipt(receipt, observation=CompletedHostStorage(_SEAL, payload, worker_pins))
