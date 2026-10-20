"""A call-local source parse cannot admit changed bytes, paths or IR."""

import pytest

from merlin.common import mlir_query as mq
from merlin.frontends.capture_normalization import normalize_capture_mlir
from merlin.targetgen import application_inventory as A


def source(tmp_path):
    path = tmp_path / "model.mlir"
    path.write_text("module { func.func @f() { %x = arith.constant 2 : i32 func.return } }")
    normalized, _ = normalize_capture_mlir(path.read_text())
    return path, normalized


def test_prepared_inventory_is_identical_and_avoids_second_parse(tmp_path, monkeypatch):
    path, text = source(tmp_path)
    ordinary = A._application_operation_inventory(path, "fixture", {})
    prepared = A.PreparedApplicationSource(path, text)
    calls = []
    parse = mq.parse

    def observed(*args, **kwargs):
        calls.append(1)
        return parse(*args, **kwargs)

    monkeypatch.setattr(mq, "parse", observed)
    actual = A._application_operation_inventory(path, "fixture", {}, prepared_source=prepared)
    assert actual == ordinary
    retained_checks = len(calls)
    calls.clear()
    A._application_operation_inventory(path, "fixture", {})
    assert len(calls) == retained_checks + 1


@pytest.mark.parametrize("mutation", ["bytes", "module", "path"])
def test_prepared_source_refuses_drift_before_inventory(tmp_path, mutation):
    from xdsl.dialects.builtin import StringAttr

    path, text = source(tmp_path)
    prepared = A.PreparedApplicationSource(path, text)
    if mutation == "bytes":
        path.write_text(path.read_text().replace("constant 2", "constant 3"))
    elif mutation == "module":
        prepared.module.attributes["changed"] = StringAttr("yes")
    else:
        other = tmp_path / "same.mlir"
        other.write_text(path.read_text())
        path = other
    with pytest.raises(ValueError, match="changed"):
        A._application_operation_inventory(path, "fixture", {}, prepared_source=prepared)


def test_source_mutated_during_inventory_is_not_published(tmp_path, monkeypatch):
    path, text = source(tmp_path)
    prepared = A.PreparedApplicationSource(path, text)
    verify = A.verify_capture_receipt

    def change(selected):
        result = verify(selected)
        path.write_text(path.read_text().replace("constant 2", "constant 3"))
        return result

    monkeypatch.setattr(A, "verify_capture_receipt", change)
    with pytest.raises(ValueError, match="changed"):
        A._application_operation_inventory(path, "fixture", {}, prepared_source=prepared)


def test_wrong_normalization_and_foreign_preparation_refuse(tmp_path):
    path, text = source(tmp_path)
    with pytest.raises(ValueError, match="normalized capture"):
        A.PreparedApplicationSource(path, text + "\n")
    with pytest.raises(ValueError, match="original prepared source"):
        A._application_operation_inventory(path, "fixture", {}, prepared_source=object())


def test_exact_but_unissued_carrier_cannot_substitute_another_module(tmp_path):
    path, text = source(tmp_path)
    forged = object.__new__(A.PreparedApplicationSource)
    forged.module = mq.parse(text.replace("constant 2", "constant 3"))
    with pytest.raises(ValueError, match="original issued"):
        A._application_operation_inventory(path, "fixture", {}, prepared_source=forged)


def test_mutating_carrier_metadata_cannot_rebind_issued_source(tmp_path):
    path, text = source(tmp_path)
    prepared = A.PreparedApplicationSource(path, text)
    prepared.module = mq.parse(text.replace("constant 2", "constant 3"))
    prepared._rendered = str(prepared.module)
    prepared._raw = path.read_bytes()
    with pytest.raises(ValueError, match="changed"):
        A._application_operation_inventory(path, "fixture", {}, prepared_source=prepared)
