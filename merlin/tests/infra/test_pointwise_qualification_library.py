"""Independent execution must load the newly built qualification library."""

from types import SimpleNamespace

from test_pointwise_source_provider import qualify


def test_recycled_work_path_does_not_reuse_previously_loaded_native_code(tmp_path):
    source = tmp_path / "provider.c"
    provider = SimpleNamespace(work=tmp_path, c_source=source, records=(), observation_count=1)
    records = []
    # pytest normally removes passed temporary trees and may reuse their names.
    # The dynamic loader can retain a previous CDLL at the same absolute path.
    # A sentinel exposes that mismatch without trapping the pytest process.
    for sentinel in (17, 29):
        source.write_text(
            "#include <stdint.h>\n"
            "void pointwise_reset(void){}\n"
            f"uint64_t pointwise_count(int index){{return {sentinel};}}\n"
        )
        _, record = qualify(provider, ())
        assert record["counters"] == [sentinel], "qualification executed a previously loaded library"
        records.append(record)
    assert records[0]["source_sha256"] != records[1]["source_sha256"]
    assert records[0]["library_sha256"] != records[1]["library_sha256"]
