"""Compile the actual target hasher and compare canonical bytes to hashlib."""

import ctypes
import hashlib
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root


def test_installed_resource_inventory_includes_the_digest_header():
    files = json.loads((repo_root() / "build_tools/package_resources.json").read_text())["files"]
    assert "merlin/runtime/baremetal/spike/output_sha256.h" in files


def test_build_refuses_a_digest_for_unsupported_output(tmp_path, monkeypatch):
    from merlin.common.digest import sha256_file
    from merlin.llvmlower import compilation_recipe, qinner, weight_prepack
    from merlin.runtime.backends import spike_model

    compiler = tmp_path / "compiler"
    compiler.write_bytes(b"test compiler identity")
    source = tmp_path / "model.ll"
    source.write_text("define void @forward() { ret void }\n")
    observation = {
        "compiler_resolved": str(compiler),
        "compiler_sha256": sha256_file(compiler),
        "data_layout": "e-p:64:64",
        "index_bits": 64,
    }
    flags = ["-march=rv64gc", "-mabi=lp64d"]
    monkeypatch.setattr(weight_prepack, "prepare_build_bundle", lambda model, *_: model)
    monkeypatch.setattr(qinner, "plan_for_bundle", lambda *_: False)
    monkeypatch.setattr(spike_model._spike, "gcc_path", lambda: compiler)
    monkeypatch.setattr(spike_model, "_mlir_runtime_compiler", lambda *_: [str(compiler)])
    monkeypatch.setattr(
        spike_model,
        "selected_model_compiler_plan",
        lambda **_: {
            "observation": observation,
            "gcc_cflags": flags,
            "clang_cflags": flags,
            "model_cflags": flags,
        },
    )
    monkeypatch.setattr(
        spike_model,
        "lower_model_file",
        lambda *_a, **_k: SimpleNamespace(
            ll_path=source,
            stats={"index_lowering": {"data_layout": "e-p:64:64", "index_bits": 64, "effective_pipeline": "test"}},
        ),
    )
    monkeypatch.setattr(compilation_recipe.CompilationRecipe, "run", lambda *_a, **_k: None)
    monkeypatch.setattr(spike_model.c_runtime, "generate", lambda *_a, **_k: {"out_dt": "bf16"})

    with pytest.raises(spike_model.SpikeModelError, match="SHA256.*f32, i64 or i1"):
        spike_model.build(tmp_path / "capture", tmp_path / "build", backend="scalar", output_sha256=True)
    assert not (tmp_path / "build/model.elf").exists()


@pytest.fixture(scope="module")
def hasher(tmp_path_factory):
    if not shutil.which("cc"):
        pytest.skip("host C compiler unavailable")
    root = repo_root()
    work = tmp_path_factory.mktemp("sha256")
    source = work / "probe.c"
    source.write_text("""#include "output_sha256.h"
void hash_bytes(const unsigned char *p, size_t n, unsigned char *out) {
 merlin_sha256 s; merlin_sha256_init(&s);
 /* Exercise streaming across every block/padding boundary. */
 for(size_t i=0;i<n;i+=7) merlin_sha256_update(&s,p+i,n-i<7?n-i:7);
 merlin_sha256_final(&s,out);
}
void hash_floats(const float *p,size_t n,unsigned char *out) {
 merlin_output_sha256(p,n,out);
}
void hash_i64(const void *p,size_t n,unsigned char *out) {
 merlin_output_sha256_i64(p,n,out);
}
void hash_i1(const void *p,size_t n,unsigned char *out) {
 merlin_output_sha256_i1(p,n,out);
}
void hash_raw(const void *p,size_t n,unsigned char *out) {
 merlin_output_sha256_raw(p,n,out);
}
""")
    lib = work / "probe.so"
    subprocess.run(
        [
            "cc",
            "-std=c99",
            "-O2",
            "-Wall",
            "-Werror",
            "-shared",
            "-fPIC",
            "-I",
            str(root / "merlin/runtime/baremetal/spike"),
            str(source),
            "-o",
            str(lib),
        ],
        check=True,
    )
    dll = ctypes.CDLL(str(lib))
    for name in ("hash_bytes", "hash_floats", "hash_i64", "hash_i1", "hash_raw"):
        getattr(dll, name).argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    return dll


@pytest.mark.parametrize(
    "data",
    [b"", b"abc", b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", b"a" * 1_000_000]
    + [bytes(range(n)) for n in (55, 56, 63, 64, 65, 119, 120, 127, 128)],
)
def test_known_vectors_and_padding(hasher, data):
    output = ctypes.create_string_buffer(32)
    hasher.hash_bytes(data, len(data), output)
    assert output.raw.hex() == hashlib.sha256(data).hexdigest()


def test_f32_encoding_preserves_every_bit(hasher):
    # Signed zero, infinities, NaN payloads, subnormal and ordinary finite bits.
    words = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC01234, 1, 0x3F800000]
    floats = (ctypes.c_uint32 * len(words))(*words)
    output = ctypes.create_string_buffer(32)
    hasher.hash_floats(floats, len(words), output)
    canonical = b"".join(w.to_bytes(4, "little") for w in words)
    assert len(canonical) == 4 * len(words)
    assert output.raw.hex() == hashlib.sha256(canonical).hexdigest()


def test_integer_and_boolean_output_encoding(hasher):
    words = [0, 1, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF]
    integers = (ctypes.c_uint64 * len(words))(*words)
    output = ctypes.create_string_buffer(32)
    hasher.hash_i64(integers, len(words), output)
    assert output.raw.hex() == hashlib.sha256(
        b"".join(word.to_bytes(8, "little") for word in words)).hexdigest()

    booleans = bytes([0, 1, 1, 0, 1])
    hasher.hash_i1(booleans, len(booleans), output)
    assert output.raw.hex() == hashlib.sha256(booleans).hexdigest()
    hasher.hash_raw(booleans, len(booleans), output)
    assert output.raw.hex() == hashlib.sha256(booleans).hexdigest()


def test_record_validation():
    import numpy as np

    from merlin.runtime.output_digest import verify_output_sha256

    reference = np.array([[1, -0.0, 3]], dtype="<f4")
    digest = hashlib.sha256(reference.tobytes()).hexdigest()
    record = f"OUT_SHA256 f32le 3 12 {digest}\n"
    assert verify_output_sha256(record, reference)["bytes"] == 12
    assert verify_output_sha256(record, reference.astype(">f4"))["sha256"] == digest
    for bad in (
        "",
        record + record,
        record.replace("3 12", "2 12"),
        record.replace("3 12", "3 8"),
        record.replace(digest, "0" * 64),
    ):
        with pytest.raises(ValueError):
            verify_output_sha256(bad, reference)
    with pytest.raises(ValueError):
        verify_output_sha256(record, reference.astype("float64"))


@pytest.mark.parametrize("reference,encoding", [
    ("i64", "i64le"), ("bool", "i1byte"),
])
def test_typed_record_validation(reference, encoding):
    import numpy as np

    from merlin.runtime.output_digest import verify_output_sha256

    array = (np.array([0, -1, 2**62], dtype=np.int64) if reference == "i64"
             else np.array([False, True, False, True], dtype=np.bool_))
    raw = array.astype("<i8", copy=False).tobytes() if reference == "i64" else array.tobytes()
    digest = hashlib.sha256(raw).hexdigest()
    record = f"OUT_SHA256 {encoding} {array.size} {len(raw)} {digest}\n"
    assert verify_output_sha256(record, array)["encoding"] == encoding
    with pytest.raises(ValueError):
        verify_output_sha256(record, np.zeros(array.shape, dtype=np.float32))
    with pytest.raises(ValueError):
        verify_output_sha256(record.replace(digest, "0" * 64), array)


def test_indexed_output_digests_cover_boolean_and_state_storage():
    import numpy as np

    from merlin.runtime.output_digest import verify_all_output_sha256, verify_output_sha256

    outputs = [np.array([False, True, True], dtype=np.bool_),
               np.array([[0x3F80, 0xBF80]], dtype=np.uint16)]
    records = [f"OUT_SHA256_INDEX {index} rawbytes {array.size} {array.nbytes} "
               f"{hashlib.sha256(array.tobytes()).hexdigest()}"
               for index, array in enumerate(outputs)]
    console = "\n".join(records) + "\n"
    assert [row["index"] for row in verify_all_output_sha256(console, outputs)] == [0, 1]
    first = f"OUT_SHA256 i1byte 3 3 {hashlib.sha256(outputs[0].tobytes()).hexdigest()}\n"
    assert verify_output_sha256(first + console, outputs[0])["encoding"] == "i1byte"
    with pytest.raises(ValueError, match="mismatch"):
        verify_all_output_sha256(console, [outputs[0], np.zeros_like(outputs[1])])
    with pytest.raises(ValueError, match="count"):
        verify_all_output_sha256(console, outputs[:1])


def test_observed_digests_bind_every_result_to_the_capture_abi():
    from merlin.runtime.output_digest import observe_output_digests

    first = hashlib.sha256(b"\x00\x00\x80?").hexdigest()
    state = hashlib.sha256(b"\x00\x00\x00\x00").hexdigest()
    console = (f"OUT_SHA256 f32le 1 4 {first}\n"
               f"OUT_SHA256_INDEX 0 rawbytes 1 4 {first}\n"
               f"OUT_SHA256_INDEX 1 rawbytes 2 4 {state}\n")
    abi = [([1], "f32"), ([2], "bf16")]
    report = observe_output_digests(console, abi, require_first_typed=True)
    assert report["indexed"][1]["dtype"] == "bf16"
    assert report["first_typed"]["sha256"] == first
    assert report["numerical_qualification"] == "not_established"
    with pytest.raises(ValueError, match="count"):
        observe_output_digests(console, abi[:1], require_first_typed=True)
    with pytest.raises(ValueError, match="disagrees"):
        observe_output_digests(console.replace("2 4", "2 8"), abi, require_first_typed=True)
    with pytest.raises(ValueError, match="typed first-output"):
        observe_output_digests(console.replace(f"OUT_SHA256 f32le 1 4 {first}\n", ""),
                               abi, require_first_typed=True)
    with pytest.raises(ValueError, match="typed first-output"):
        observe_output_digests(console.replace(f"OUT_SHA256 f32le 1 4 {first}",
                                               f"OUT_SHA256 f32le 1 4 {state}"),
                               abi, require_first_typed=True)
