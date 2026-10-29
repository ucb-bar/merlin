"""Bounded assignment of complete persistent-state requests to supplied contexts.

This is a source planning utility, not a hardware or asynchronous-execution
qualification. Callers supply the supported context identities and complete
state requests, encode the actual setters/consumers, and use ordinary
configuration-effect verification on the resulting program.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfigurationRequest:
    consumer: str
    state: tuple[tuple[str, int], ...]

    def __post_init__(self):
        if type(self.consumer) is not str or not self.consumer:
            raise ValueError("configuration request requires an explicit consumer identity")
        if (
            type(self.state) is not tuple
            or not self.state
            or any(type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not str
                   or not pair[0] or type(pair[1]) is not int for pair in self.state)
            or len({key for key, _ in self.state}) != len(self.state)
        ):
            raise ValueError("configuration request requires complete unique integer state fields")


@dataclass(frozen=True)
class ConfigurationContexts:
    bindings: tuple[tuple[str, int], ...]
    states: tuple[tuple[int, tuple[tuple[str, int], ...]], ...]

    def context(self, consumer: str) -> int:
        selected = [context for name, context in self.bindings if name == consumer]
        if len(selected) != 1:
            raise ValueError("consumer lacks a unique selected configuration context")
        return selected[0]

    def record(self):
        return {
            "schema": "merlin.configuration_contexts.v1",
            "bindings": dict(self.bindings),
            "states": [{"context": context, "fields": dict(state)} for context, state in self.states],
            "scope": "complete request sharing; caller must encode and verify actual persistent state effects",
        }


def assign_configuration_contexts(
    requests: tuple[ConfigurationRequest, ...],
    contexts: tuple[int, ...],
    *,
    max_requests: int = 256,
    max_fields: int = 64,
) -> ConfigurationContexts | None:
    """Share equal complete states; refuse insufficient supplied contexts.

    No replacement or scheduling is inferred. A consumer with several distinct
    states cannot have one static context selector; callers retain their prior
    legal route instead. Assignment is deterministic in source-request order.
    """
    if any(type(n) is not int or n <= 0 for n in (max_requests, max_fields)):
        raise ValueError("configuration assignment requires explicit positive budgets")
    if (type(requests) is not tuple or len(requests) > max_requests
            or any(type(request) is not ConfigurationRequest for request in requests)):
        raise ValueError("configuration requests exceed the typed bounded source roster")
    if (type(contexts) is not tuple or not contexts or len(contexts) > max_requests
            or any(type(context) is not int or context < 0 for context in contexts)
            or len(set(contexts)) != len(contexts)):
        raise ValueError("supported configuration contexts must be explicit unique nonnegative identities")
    states, bindings, by_consumer = [], [], {}
    for request in requests:
        request.__post_init__()
        if len(request.state) > max_fields:
            raise ValueError("configuration request exceeds the explicit field budget")
        state = tuple(sorted(request.state))
        if request.consumer in by_consumer:
            if by_consumer[request.consumer] != state:
                return None
            continue
        if state not in states:
            if len(states) == len(contexts):
                return None
            states.append(state)
        by_consumer[request.consumer] = state
        bindings.append((request.consumer, contexts[states.index(state)]))
    return ConfigurationContexts(tuple(bindings), tuple(zip(contexts, states)))
