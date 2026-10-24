"""Lossless multi-output protocol used by batched conformance images."""

import pytest

from merlin.runtime.backends.spike_model import SpikeModelError, parse_all_output_bytes


def test_parse_all_output_bytes_preserves_result_boundaries_and_raw_values():
    console = "OUT_BYTES 0 3 0 127 255\nOUT_BYTES 1 2 52 18\nDONE\n"
    assert parse_all_output_bytes(console) == [b"\x00\x7f\xff", b"\x34\x12"]


@pytest.mark.parametrize(
    "line",
    [
        "OUT_BYTES 0 2 1",
        "OUT_BYTES 0 1 256",
        "OUT_BYTES 1 1 0",
        "OUT_BYTES 0 0\nOUT_BYTES 0 0",
    ],
)
def test_parse_all_output_bytes_fails_closed_on_malformed_protocol(line):
    with pytest.raises(SpikeModelError):
        parse_all_output_bytes(line)


@pytest.mark.parametrize(
    "console",
    [
        "OUT_BYTES 0 1 0\n",
        "DONE\nOUT_BYTES 0 1 0\n",
        "OUT_BYTES 0 1 0\nDONE\nDONE\n",
        "OUT_BYTES 0 1 -1\nDONE\n",
        "OUT_BYTES 0 1 0\nOUT 1 0\nDONE\n",
    ],
)
def test_execution_parser_requires_complete_unmixed_protocol(console):
    from merlin.runtime.backends.spike_model import parse_console

    with pytest.raises(SpikeModelError):
        parse_console(console)


def test_run_returns_all_storage_bytes(monkeypatch):
    from types import SimpleNamespace

    from merlin.runtime.backends import spike_model

    console = b"OUT_BYTES 0 0\nOUT_BYTES 1 3 0 128 255\nMETRIC cycles 42\nDONE\n"
    monkeypatch.setattr(spike_model._spike, "spike_path", lambda: "unused")
    monkeypatch.setattr(
        spike_model.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=0, stdout=console, stderr=b"")
    )
    result = spike_model.run("unused")
    assert result["output_bytes"] == [b"", b"\x00\x80\xff"]
    assert result["metrics"] == {"cycles": 42}


@pytest.mark.parametrize("dump_all", [False, True])
def test_generator_sizes_every_dtype_and_accepts_zero_inputs(tmp_path, monkeypatch, dump_all):
    import json
    import subprocess

    import numpy as np

    from merlin.common.paths import repo_root
    from merlin.llvmlower import c_runtime

    model = tmp_path / "capture"
    model.mkdir()
    (model / "model.mlir").write_text("module {}")
    (model / "weights.safetensors.manifest.json").write_text(json.dumps({}))
    np.savez(model / "inputs.npz")
    specs = [([3], dt) for dt in c_runtime.DT_BYTES] + [([0], "f32"), ([], "i64"), ([1025], "f32")]
    monkeypatch.setattr(c_runtime, "parse_forward_signature", lambda _: [])
    monkeypatch.setattr(c_runtime, "_out_specs", lambda _: specs)
    generated = tmp_path / "generated"
    c_runtime.generate(model, generated, model / "inputs.npz", dump_all_outputs=dump_all)
    header = (generated / "model_gen.h").read_text()
    assert ("MERLIN_DUMP_ALL_OUTPUTS" in header) == dump_all
    if dump_all:
        sizes = ",".join(str(3 * width) for width in c_runtime.DT_BYTES.values()) + ",0,8,4100"
        assert "= {" + sizes + "};" in header
    io = (generated / "model_io.h").read_text()
    assert "MERLIN_INPUT_PTR[MERLIN_N_ARGS] = {" + ",".join("0" for _ in specs) + "};" in io
    probe = generated / "probe.c"
    probe.write_text('#include "model_gen.h"\n#include "model_io.h"\n')
    subprocess.run(
        [
            "cc",
            "-std=c11",
            "-fsyntax-only",
            "-I",
            str(repo_root() / "merlin/runtime/c"),
            "-I",
            str(generated),
            str(probe),
        ],
        check=True,
        capture_output=True,
    )
    if dump_all:
        # Execute the runtime's actual byte loop on host storage with every dtype.
        runtime = (repo_root() / "merlin/runtime/baremetal/spike/model_main.c").read_text()
        emission = runtime.split("#ifdef MERLIN_DUMP_ALL_OUTPUTS\n", 1)[1].split("#else", 1)[0]
        probe.write_text(
            '#include "model_gen.h"\n#include "model_io.h"\n#include <stdio.h>\n'
            "void htif_puts(const char *s) { fputs(s, stdout); }\n"
            'void htif_putd(long v) { printf("%ld", v); }\n'
            "void htif_putc(char c) { putchar(c); }\n"
            "int main(void) {\n"
            "for (int o=0; o<MERLIN_N_OUTPUTS; o++)\n"
            " for (size_t b=0; b<MERLIN_OUTPUT_NBYTES[o]; b++)\n"
            "  ((unsigned char *)MERLIN_OUTPUT_PTR[o])[b]=(unsigned char)(b+o);\n"
            + emission
            + 'htif_puts("DONE\\n"); return 0; }\n'
        )
        executable = generated / "probe"
        subprocess.run(
            [
                "cc",
                "-std=c11",
                "-I",
                str(repo_root() / "merlin/runtime/c"),
                "-I",
                str(generated),
                str(probe),
                "-lm",
                "-o",
                str(executable),
            ],
            check=True,
            capture_output=True,
        )
        console = subprocess.run([str(executable)], check=True, capture_output=True, text=True).stdout
        expected = [
            bytes((b + i) % 256 for b in range(int(np.prod(shape)) * c_runtime.DT_BYTES[dt]))
            for i, (shape, dt) in enumerate(specs)
        ]
        assert parse_all_output_bytes(console) == expected


@pytest.mark.parametrize("dump_all", [False, True])
def test_build_threads_opt_in_to_generator(tmp_path, monkeypatch, dump_all):
    from types import SimpleNamespace

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

    def generate(*args, **kwargs):
        assert kwargs.get("dump_all_outputs", False) == dump_all
        raise RuntimeError("generator plumbing observed")

    monkeypatch.setattr(spike_model.c_runtime, "generate", generate)
    with pytest.raises(RuntimeError, match="generator plumbing observed"):
        spike_model.build(tmp_path / "capture", tmp_path / "build", backend="scalar", dump_all_outputs=dump_all)
