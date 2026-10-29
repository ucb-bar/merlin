"""Known native source refusal versus tool failure; no kernel acceptance."""

import hashlib
from types import SimpleNamespace

import pytest

from merlin.verify.lean.source_profile_export import SCHEMA, export_source


class NativeSyntaxError(Exception):
    pass


class Context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def native(*, parser_error=None, exporter_error=None):
    calls = []

    def parse(text):
        calls.append(("parse", text))
        if parser_error is not None:
            raise parser_error

    def export(text):
        calls.append(("export", text))
        if exporter_error is not None:
            raise exporter_error
        return {
            "schema": "merlin.lean.ir.v1",
            "source": {"sha256": hashlib.sha256(text.encode()).hexdigest(), "bytes": len(text.encode())},
        }

    return SimpleNamespace(
        ir=SimpleNamespace(Context=Context, Module=SimpleNamespace(parse=parse), MLIRError=NativeSyntaxError),
        export=export,
    ), calls


def test_native_syntax_refusal_has_bound_source_and_never_calls_exporter():
    tool, calls = native(parser_error=NativeSyntaxError("unsupported source assembly"))
    raw = b"source\r\n"
    result = export_source(raw, tool)
    assert result == {
        "schema": SCHEMA,
        "source": {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)},
        "supported": False,
        "reason": "unsupported source assembly",
    }
    assert calls == [("parse", raw.decode())]


@pytest.mark.parametrize("error", [RuntimeError("parser failed"), MemoryError("resource failure")])
def test_unexpected_parser_failures_are_not_source_exclusions(error):
    tool, calls = native(parser_error=error)
    with pytest.raises(type(error)) as caught:
        export_source(b"source", tool)
    assert caught.value is error and len(calls) == 1


def test_export_phase_native_error_is_still_a_tool_failure():
    error = NativeSyntaxError("export failed after successful source parsing")
    tool, _ = native(exporter_error=error)
    with pytest.raises(NativeSyntaxError) as caught:
        export_source(b"source", tool)
    assert caught.value is error


def test_success_preserves_original_crlf_bytes_for_the_ordinary_exporter():
    tool, calls = native()
    raw = b"source\r\n"
    result = export_source(raw, tool)
    assert result["schema"] == "merlin.lean.ir.v1"
    assert result["source"] == {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    assert calls == [("parse", raw.decode()), ("export", raw.decode())]
