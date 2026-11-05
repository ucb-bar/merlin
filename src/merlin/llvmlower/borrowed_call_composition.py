"""Join distinct live borrowed owners for one exact original source.

No owner is recovered from a record. Each original callback, effect object,
ordered ABI and native observer remains independently required.
"""

import json
from pathlib import Path

from merlin.common.digest import sha256_file

from . import borrowed_buffer_calls as K


def compose_borrowed_calls(owners):
    """Combine disjoint original rosters without changing either source epoch."""
    if (type(owners) is not tuple or not 1 <= len(owners) <= 32
            or any(type(owner) is not K.BorrowedBufferCalls for owner in owners)
            or len({id(owner) for owner in owners}) != len(owners)):
        raise ValueError('borrowed composition requires distinct live original owners')
    for owner in owners:
        owner.revalidate()
    first = owners[0]
    epoch = first.source_path, first.source_sha256, first.input_sha256
    if any((owner.source_path, owner.source_sha256, owner.input_sha256) != epoch for owner in owners):
        raise ValueError('borrowed owners do not share the exact original source epoch')
    calls = tuple(call for owner in owners for call in owner.calls)
    K.validate_calls(calls)
    records = tuple(json.dumps(owner.record(), sort_keys=True) for owner in owners)
    callbacks = tuple((K._callback_binding(owner.validate_original), K._callback_binding(owner.observe_native))
                      for owner in owners)
    original_calls = tuple(tuple(owner.calls) for owner in owners)
    products = {}
    for path, digest in (*(pin for owner in owners for pin in owner.pins),
                         (Path(__file__).resolve(), sha256_file(__file__))):
        if path in products and products[path] != digest:
            raise ValueError('borrowed owners have conflicting original product identities')
        products[path] = digest

    def validate():
        for index, owner in enumerate(owners):
            def check():
                if ((owner.source_path, owner.source_sha256, owner.input_sha256) != epoch
                        or json.dumps(owner.record(), sort_keys=True) != records[index]
                        or (K._callback_binding(owner.validate_original), K._callback_binding(owner.observe_native))
                        != callbacks[index] or len(owner.calls) != len(original_calls[index])
                        or any(a is not b for a, b in zip(owner.calls, original_calls[index], strict=True))):
                    raise ValueError('original borrowed composition owner or ABI changed')
            check()
            owner.revalidate()
            check()

    def observe(paths, initial):
        validate()
        for owner in owners:
            if owner.observe_native is not None:
                owner.observe_native(paths, initial)
        validate()

    result = K.BorrowedBufferCalls(*epoch, calls, tuple(products.items()), validate,
                                  observe if any(owner.observe_native is not None for owner in owners) else None)
    result.revalidate()
    return result
