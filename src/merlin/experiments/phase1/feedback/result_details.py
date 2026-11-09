"""Host import of the standalone public result index/detail transport."""

from ..tools.result_details import (
    MAX_DETAIL_BYTES,
    SCHEMA,
    _CACHE,
    append_rows,
    compact_document as _compact_document,
    compact_row as _compact_row,
    hydrate,
    load_row,
)


def compact_document(document, *, workspace, namespace="grade"):
    from .author_disclosure import project

    return _compact_document(project(document, workspace=workspace), workspace=workspace, namespace=namespace)


def compact_row(row, *, workspace):
    from .author_disclosure import project, restricted

    if restricted():
        row = project(load_row(row, workspace=workspace))
    return _compact_row(row, workspace=workspace)

__all__ = ["MAX_DETAIL_BYTES", "SCHEMA", "append_rows", "compact_document", "compact_row", "hydrate", "load_row"]
