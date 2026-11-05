"""Fresh shared-file custody across actual completed provider records."""

from __future__ import annotations

import copy
import json
import os
from collections import Counter
from pathlib import Path

import pytest
from test_host_provider import _fixture, _link

from merlin.common.digest import sha256_file
from merlin.runtime import host_provider as HP


def completed(directory):
    directory.mkdir()
    context, build, nm = _fixture(directory)
    objects, record = HP.prepare_host_provider(lambda ctx: build, context, inspector=nm)
    executable = directory / "linked.so"
    _link(context, objects, executable)
    HP.close_host_provider(
        record,
        [context.model_object_path, *objects],
        executable,
        inspector=nm,
        native_dynamic_imports=(),
        native_link_inputs=tuple(
            (path, sha256_file(path)) for path in (context.compiler.resolve(), directory / "model.c")
        ),
    )
    return record


def persist(record):
    Path(record["receipt_path"]).write_text(json.dumps(record))


def counter(monkeypatch):
    calls = Counter()
    original = HP.sha256_file

    def measured(path):
        calls[str(Path(path))] += 1
        return original(path)

    monkeypatch.setattr(HP, "sha256_file", measured)
    return calls


def test_batch_hashes_shared_exact_paths_once_and_reopens_next_boundary(tmp_path, monkeypatch):
    first, second = completed(tmp_path / "first"), completed(tmp_path / "second")
    shared = set(first["compilation_pins"]) & set(second["compilation_pins"])
    assert len(shared) == 1
    calls = counter(monkeypatch)
    HP.revalidate_host_providers((first, None, second, first))
    assert shared <= calls.keys()
    assert set(calls.values()) == {1}
    previous = calls.copy()
    HP.revalidate_host_providers([second, first])
    assert calls == Counter({path: count * 2 for path, count in previous.items()})
    calls.clear()
    HP.revalidate_host_provider(first)
    assert set(calls.values()) == {1}
    assert all(calls[path] == 1 for path in shared)


def test_next_batch_detects_same_size_restored_timestamp_change(tmp_path):
    record = completed(tmp_path / "model")
    HP.revalidate_host_providers((record,))
    source = next(Path(path) for path in record["compilation_pins"] if path.endswith("provider.c"))
    timestamp, contents = source.stat(), source.read_bytes()
    changed = contents.replace(b"source_fallback(in, out)", b"source_fallback(out, in)")
    assert changed != contents and len(changed) == len(contents)
    source.write_bytes(changed)
    os.utime(source, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
    assert source.stat().st_size == timestamp.st_size
    assert source.stat().st_mtime_ns == timestamp.st_mtime_ns
    with pytest.raises(ValueError, match="compilation dependency changed"):
        HP.revalidate_host_providers((record,))


@pytest.mark.parametrize("kind", ["compilation", "native", "receipt"])
def test_conflicting_original_expectations_refuse(tmp_path, kind):
    first, second = completed(tmp_path / "first"), completed(tmp_path / "second")
    shared = next(iter(set(first["compilation_pins"]) & set(second["compilation_pins"])))
    if kind == "compilation":
        second["compilation_pins"][shared] = "f" * 64
        persist(second)
    elif kind == "native":
        second["native_dynamic_link"]["original_inputs"][0]["sha256"] = "f" * 64
        persist(second)
    else:
        # Two originals cannot assign different logical contents to one receipt.
        second["receipt_path"] = first["receipt_path"]
    with pytest.raises(ValueError, match="changed"):
        HP.revalidate_host_providers((first, second))


@pytest.mark.parametrize("kind", ["symlink", "missing", "symlink_alias"])
def test_raw_paths_remain_regular_files_even_after_shared_cache_hit(tmp_path, kind):
    first, second = completed(tmp_path / "first"), completed(tmp_path / "second")
    source = next(Path(path) for path in second["compilation_pins"] if path.endswith("provider.c"))
    if kind == "missing":
        source.unlink()
    elif kind == "symlink":
        original = source.with_suffix(".original")
        source.rename(original)
        source.symlink_to(original)
    else:
        known = next(Path(path) for path in first["compilation_pins"] if path.endswith("provider.c"))
        alias = tmp_path / "alias.c"
        alias.symlink_to(known)
        second["compilation_pins"][str(alias)] = sha256_file(known)
        persist(second)
    with pytest.raises(ValueError, match="regular files"):
        HP.revalidate_host_providers((first, second))


@pytest.mark.parametrize("kind", ["record", "sequence", "native_input"])
def test_original_record_mutation_during_check_cannot_publish_success(tmp_path, monkeypatch, kind):
    record = completed(tmp_path / "model")
    originals = [record]
    original = HP.sha256_file
    triggered = False

    def mutate(path):
        nonlocal triggered
        result = original(path)
        if not triggered:
            triggered = True
            if kind == "record":
                record["effect_witness_sha256"] = "f" * 64
            elif kind == "sequence":
                originals[0] = copy.deepcopy(record)
            else:
                record["native_dynamic_link"]["original_inputs"].clear()
        return result

    monkeypatch.setattr(HP, "sha256_file", mutate)
    with pytest.raises(ValueError, match="original custody records changed"):
        HP.revalidate_host_providers(originals)


def test_explicit_finite_sequences_and_none_leave_no_artifacts(tmp_path):
    HP.revalidate_host_providers([])
    HP.revalidate_host_providers((None, None))
    HP.revalidate_host_provider(None)
    with pytest.raises(ValueError, match="explicit finite"):
        HP.revalidate_host_providers(iter((None,)))
    with pytest.raises(ValueError, match="completed link"):
        HP.revalidate_host_providers((None, {"status": "prepared"}))
    assert list(tmp_path.iterdir()) == []


def test_later_record_check_cannot_mutate_an_earlier_original(tmp_path, monkeypatch):
    first, second = completed(tmp_path / "first"), completed(tmp_path / "second")
    original = HP.sha256_file
    changed = []

    def mutate(path):
        result = original(path)
        if str(path) == second["receipt_path"]:
            first["effect_witness_sha256"] = "f" * 64
            changed.append(True)
        return result

    monkeypatch.setattr(HP, "sha256_file", mutate)
    with pytest.raises(ValueError, match="original custody records changed"):
        HP.revalidate_host_providers((first, second))
    assert changed == [True]


def test_all_originals_are_frozen_before_any_file_read(tmp_path, monkeypatch):
    first, second = completed(tmp_path / "first"), completed(tmp_path / "second")
    original = HP.sha256_file

    def mutate(path):
        result = original(path)
        if str(path) == first["receipt_path"]:
            second["effect_witness_sha256"] = "f" * 64
            persist(second)
        return result

    monkeypatch.setattr(HP, "sha256_file", mutate)
    with pytest.raises(ValueError, match="completed receipt changed"):
        HP.revalidate_host_providers((first, second))


@pytest.mark.parametrize("role", ["source_model", "ordered_link_objects", "executable", "native"])
@pytest.mark.parametrize("mutation", ["extra_field", "missing_field", "noncanonical_path"])
def test_full_identity_roles_preserve_exact_record_equality(tmp_path, role, mutation):
    record = completed(tmp_path / "model")
    if role == "native":
        expected = record["native_dynamic_link"]["original_inputs"][0]
    elif role == "executable":
        expected = record[role]
    else:
        expected = record[role][0]
    if mutation == "extra_field":
        expected["extra"] = "not an original identity field"
    elif mutation == "missing_field":
        del expected["sha256"]
    else:
        path = Path(expected["path"])
        expected["path"] = str(path.parent) + "/../" + path.parent.name + "/" + path.name
    persist(record)
    with pytest.raises(ValueError, match="changed after link closure"):
        HP.revalidate_host_providers((record,))


def test_actual_native_only_original_input_changes_still_refuse(tmp_path):
    record = completed(tmp_path / "model")
    HP.revalidate_host_providers((record,))
    model_source = tmp_path / "model" / "model.c"
    assert str(model_source) not in record["compilation_pins"]
    model_source.write_bytes(model_source.read_bytes() + b"\n/* changed native original */\n")
    with pytest.raises(ValueError, match="native provider original model/runtime input changed"):
        HP.revalidate_host_providers((record,))


def test_record_count_budget_refuses_before_snapshot_or_file_work(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("over-budget records must not be serialized or hashed")

    monkeypatch.setattr(HP.json, "dumps", forbidden)
    monkeypatch.setattr(HP, "sha256_file", forbidden)
    with pytest.raises(ValueError, match="exceeds its explicit budget"):
        HP.revalidate_host_providers((None, None), max_records=1)
    with pytest.raises(ValueError, match="exceeds its explicit budget"):
        HP.revalidate_host_providers([None] * 4097)


@pytest.mark.parametrize("budget", [True, 0, -1, 1.5, None])
def test_record_count_budget_is_an_original_positive_integer(budget):
    with pytest.raises(ValueError, match="record budget"):
        HP.revalidate_host_providers((), max_records=budget)


def test_record_count_budget_accepts_exact_boundary():
    HP.revalidate_host_providers((None, None), max_records=2)


def test_distinct_raw_path_spellings_are_not_coalesced(tmp_path, monkeypatch):
    record = completed(tmp_path / "model")
    source = next(path for path in record["compilation_pins"] if path.endswith("provider.c"))
    path = Path(source)
    alias = str(path.parent) + "/./" + path.name
    assert alias != source
    record["compilation_pins"][alias] = record["compilation_pins"][source]
    persist(record)
    calls = counter(monkeypatch)
    HP.revalidate_host_providers((record,))
    # The actual file is read twice because the caller supplied two raw paths.
    assert calls[source] == 2


def test_receipt_raw_path_is_not_coalesced_with_a_differently_spelled_pin(tmp_path, monkeypatch):
    record = completed(tmp_path / "model")
    receipt = Path(record["receipt_path"])
    record["receipt_path"] = str(receipt.parent) + "/./" + receipt.name
    persist(record)
    # Use a second completed record to pin the first receipt, avoiding a
    # self-referential digest while exercising this distinct raw-path role.
    other = completed(tmp_path / "other")
    other["compilation_pins"][str(receipt)] = sha256_file(receipt)
    persist(other)
    calls = counter(monkeypatch)
    HP.revalidate_host_providers((record, other))
    assert calls[str(receipt)] == 2
