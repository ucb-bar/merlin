"""Transitional provider imports for existing target-generation and OOT consumers.

Identity and metadata parsing live in ``merlin.compiler.backend.load``. Remove
this surface when current support plugins and target-generation callers migrate;
the aliases preserve class identity and contain no second implementation.
"""

from merlin.compiler.backend.load import (
    METADATA_FILE,
    SCHEMA,
    Provider,
    ProviderError,
    ProviderRole,
    contained_resource,
    read_provider,
)

__all__ = [
    "METADATA_FILE",
    "SCHEMA",
    "Provider",
    "ProviderError",
    "ProviderRole",
    "contained_resource",
    "read_provider",
]
