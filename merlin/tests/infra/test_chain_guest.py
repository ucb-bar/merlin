"""Actually link retained tables/objects from two programs into one loop guest."""

import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from merlin.model_check.chain_image_data import render
from merlin.model_check.chain_image_objects import namespace, pin

ROOT = Path(__file__).resolve().parents[3]


def run(argv):
    result = subprocess.run([str(x) for x in argv], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode()
    return result


@pytest.fixture
def guest(tmp_path, request):
    mode = getattr(request, "param", {})
    gcc = Path(shutil.which("gcc")).resolve()
    nm = Path(shutil.which("nm")).resolve()
    objcopy = Path(shutil.which("objcopy")).resolve()
    headers = ROOT / "merlin/runtime/c"
    names, programs = [], []
    for index in range(2):
        directory = tmp_path / f"original{index}"
        directory.mkdir()
        code = {
            "model.c": """#include <stdint.h>
void forward(void **d) {
 float *in=((void **)d[0])[1], *w=((void **)d[1])[1], *out=((void **)d[2])[1];
 int64_t *count=((void **)d[3])[1]; unsigned char *flag=((void **)d[4])[1];
 out[0]=in[0]+w[0]; out[1]=in[1]+w[1]; count[0]=(int64_t)out[0]; flag[0]=(out[1]>0);
}
uint64_t fixture_counter(void) { static uint64_t counter; return ++counter; }
""",
            "model_call.c": "void forward(void **); void merlin_invoke(void **d) { forward(d); }\n",
            "model_main.c": """#include "merlin_model.h"
static float input[2], output[2]; static int64_t count[1]; static unsigned char flag[1];
static const merlin_arg_t MERLIN_ARGS[] = {
 {MERLIN_INPUT,0,1,{2},4},{MERLIN_WEIGHT,0,1,{2},4},{MERLIN_OUTPUT,0,1,{2},4},
 {MERLIN_OUTPUT,1,1,{1},8},{MERLIN_OUTPUT,2,1,{1},1}};
static void *const MERLIN_INPUT_PTR[]={input,0,0,0,0};
static void *const MERLIN_OUTPUT_PTR[]={output,count,flag};
int main(void) { return (int)(MERLIN_ARGS[0].dims[0] + (char *)MERLIN_INPUT_PTR[0] - (char *)MERLIN_OUTPUT_PTR[0]); }
""",
        }
        if mode.get("assert_no_transport"):
            code["model.c"] = code["model.c"].replace(
                "void forward(void **d) {",
                "extern int fixture_transport_started;\nvoid forward(void **d) {\n"
                " if (fixture_transport_started) __builtin_trap();",
            )
        substitutions = {
            "nan": "out[0]=0.0f/0.0f; count[0]=0;",
            "inf": "out[0]=1.0f/0.0f; count[0]=0;",
            "bool": "flag[0]=2;",
            "zero": "out[0]=-0.0f; count[0]=0;",
            "near": "out[0] += 0.000002f;",
            "wide": "count[0]=9007199254740993LL;",
        }
        if mode.get("profile"):
            code["model.c"] += """
void htif_puts(const char *);
void merlin_group_profile_reset(void) {}
void merlin_group_profile_dump(uint64_t start, uint64_t end) {
  (void)start; (void)end;
  htif_puts("GAP 0 1\\nMETRIC group_calls 0\\nMETRIC group_calls_dropped 0\\n");
}
"""
        weightless = index == 1 and mode.get("weightless_second")
        if weightless:
            code["model.c"] = (
                code["model.c"]
                .replace(
                    "float *in=((void **)d[0])[1], *w=((void **)d[1])[1], *out=((void **)d[2])[1];",
                    "float *in=((void **)d[0])[1], *out=((void **)d[1])[1];",
                )
                .replace("d[3]", "d[2]")
                .replace("d[4]", "d[3]")
                .replace("w[0]", "10.0f")
                .replace("w[1]", "20.0f")
            )
            code["model_main.c"] = (
                code["model_main.c"]
                .replace("{MERLIN_WEIGHT,0,1,{2},4},", "")
                .replace("MERLIN_INPUT_PTR[]={input,0,0,0,0}", "MERLIN_INPUT_PTR[]={input,0,0,0}")
            )
        if mode.get("actual"):
            # Override after the ordinary outputs without introducing invalid float-to-int casts.
            code["model.c"] = code["model.c"].replace(
                "\n}\nuint64_t", "\n" + substitutions[mode["actual"]] + "\n}\nuint64_t"
            )
            if mode["actual"] in {"nan", "inf"}:
                code["model.c"] = code["model.c"].replace("count[0]=(int64_t)out[0]", "count[0]=0")
        objects = []
        for name, text in code.items():
            source, obj = directory / name, directory / Path(name).with_suffix(".o")
            source.write_text(text)
            run([gcc, "-O0", "-fno-pie", "-I" + str(headers), "-c", source, "-o", obj])
            objects.append(pin(obj))
        weights = directory / "weights.bin"
        weights.write_bytes(
            b"" if weightless else struct.pack("<ff", 1.0 if not index else 10.0, 2.0 if not index else 20.0)
        )
        weight_obj = directory / "weights_blob.o"
        if weightless:
            run([Path(shutil.which("ld")).resolve(), "-r", "-b", "binary", "-o", weight_obj, weights])
        else:
            run([objcopy, "-I", "binary", "-O", "elf64-x86-64", "-B", "i386:x86-64", weights, weight_obj])
        objects.append(pin(weight_obj))
        program = {"objects": objects, "by_name": {Path(r["path"]).name: r for r in objects}, "link": {"argv": []}}
        names.append(namespace(program, index, tmp_path / f"namespace{index}", nm=nm, objcopy=objcopy, timeout=15))
        programs.append(
            {
                "index": index,
                "name": f"p{index}",
                "steps": (1, 3)[index],
                "n_args": 4 if weightless else 5,
                "n_outputs": 3,
                "outputs": [([2], "f32"), ([1], "i64"), ([1], "i1")],
            }
        )
    schedule = [
        {
            "program": p,
            "step": s,
            "transfers": [
                {"argument": 0, "source_program": p if s else p - 1, "source_output": 0, "source_offset": 0, "bytes": 8}
            ],
        }
        for p, n in enumerate((1, 3))
        for s in range(n)
    ]
    inputs = tmp_path / "inputs.bin"
    inputs.write_bytes(struct.pack("<ff", 2.0, 4.0))
    references = tmp_path / "references.bin"
    locations = []
    with references.open("wb") as stream:
        for values in ((3, 6), (13, 26), (23, 46), (33, 66)):
            if mode.get("actual") == "zero":
                values = (0, values[1])
            offsets = [stream.tell()]
            stream.write(struct.pack("<ff", *values))
            offsets.append(stream.tell())
            count = 9007199254740993 if mode.get("actual") == "wide" else values[0]
            stream.write(struct.pack("<q", count))
            offsets.append(stream.tell())
            stream.write(b"\x01")
            locations.append(offsets)
    layout = {
        "inputs": 0x60000000,
        "references": 0x60010000,
        "weights": [
            {
                "begin": 0
                if index == 1 and (mode.get("weightless_second") or mode.get("invalid_empty"))
                else 0x60020000 + index * 0x10000,
                "bytes": 0 if index == 1 and (mode.get("weightless_second") or mode.get("invalid_empty")) else 8,
            }
            for index in range(2)
        ],
    }
    plan = {"programs": programs, "schedule": schedule, "float_policy": {"atol": 2e-6, "rtol": 1e-5}}

    def compile_guest(*, compact=True, timing_only=False, mutate_reference=None, source_extra=""):
        transport = "timing_only_v1" if timing_only else (
            "all_element_comparison_v1" if compact else "full_arrays_v1")
        if mutate_reference:
            data = bytearray(references.read_bytes())
            mutate_reference(data)
            references.write_bytes(data)
        (tmp_path / "chain_data.h").write_text(
            render(
                plan,
                names,
                layout,
                {"offsets": locations},
                transport=transport,
                counter_symbol="merlin_chain_p0_fixture_counter",
                group_profiles=[{"reset": n["prefix"] + "merlin_group_profile_reset",
                                 "dump": n["prefix"] + "merlin_group_profile_dump"} for n in names]
                if mode.get("profile") else None,
            )
        )
        wrapper = tmp_path / "console.c"
        wrapper.write_text(
            """#include <stdio.h>
void console_init(void) {} void htif_puts(const char *s) { fputs(s,stdout); }
void htif_putc(char c) { putchar((unsigned char)c); } void htif_line_flush(int flag) { (void)flag; }
void htif_exit(int code) { (void)code; } void merlin_arena_reset(void) {}
unsigned long long merlin_memref_rank_mismatches(void) { return 0; }
"""
            + source_extra
        )
        if mode.get("assert_no_transport"):
            wrapper.write_text(
                "int fixture_transport_started;\n" + wrapper.read_text().replace(
                    "{ fputs(s,stdout); }", "{ fixture_transport_started=1; fputs(s,stdout); }"
                ).replace(
                    "{ putchar((unsigned char)c); }", "{ fixture_transport_started=1; putchar((unsigned char)c); }"
                )
            )
        if mode.get("runtime_errors"):
            wrapper.write_text(
                wrapper.read_text().replace(
                    "merlin_memref_rank_mismatches(void) { return 0; }",
                    "merlin_memref_rank_mismatches(void) { static int n; return !n++; }",
                )
            )
        blob_asm = tmp_path / "blobs.S"
        blob_asm.write_text(
            "\n".join(
                f'.section .merlin_chain_{name},"a"\n.incbin {json.dumps(str(path))}'
                for name, path in (("inputs", inputs), ("references", references))
            )
        )
        elf = tmp_path / "guest"
        run(
            [
                gcc,
                "-no-pie",
                "-fno-pie",
                "-O2",
                "-ffreestanding",
                "-fno-builtin",
                "-fno-fast-math",
                "-ffp-contract=off",
                "-DMERLIN_CHAIN_HOST_FIXTURE",
                "-I" + str(headers),
                "-I" + str(tmp_path),
                headers / "chain_guest.c",
                headers / "merlin_model.c",
                wrapper,
                blob_asm,
                *(Path(r["path"]) for n in names for r in n["objects"]),
                "-Wl,--section-start=.merlin_chain_inputs=0x60000000",
                "-Wl,--section-start=.merlin_chain_references=0x60010000",
                "-Wl,--section-start=.merlin_chain_weights_0=0x60020000",
                "-Wl,--section-start=.merlin_chain_weights_1=0x60030000",
                "-o",
                elf,
            ]
        )
        result = subprocess.run([str(elf)], capture_output=True, timeout=5)
        result.image = {
            "group_profile": {"enabled": bool(mode.get("profile"))},
            "transport": transport,
            "plan": plan,
            "reference": {"blob": pin(references), "offsets": locations},
            "chain_steps": 4,
            "output_roster": [
                {"stage": p["name"], "step": s, "index": i, "dtype": dtype, "shape": shape}
                for p in programs
                for s in range(p["steps"])
                for i, (shape, dtype) in enumerate(p["outputs"])
            ],
        }
        result.elf = elf
        return result

    return compile_guest


def test_two_program_recurrent_complete_guest(guest):
    from merlin.model_check.chain_image_readback import decode

    result = guest()
    assert result.returncode == 0, result.stderr
    text = result.stdout.decode()
    assert text.count("CHAIN_STEP v1 ") == 4
    assert text.count("CHAIN_ARRAY v1 ") == 12
    assert "CHAIN_END v1 4 12 0\n" in text
    assert "CHAIN_STEP v1 1 2 1 0\n" in text
    report = decode(result.image, result.stdout)
    assert report["numerical_status"] == report["runtime_status"] == "pass"
    assert len(report["comparisons"]) == 12


@pytest.mark.parametrize("guest", [{"weightless_second": True}], indirect=True)
def test_parameter_free_program_reuses_actual_empty_object_and_carries_state(guest):
    from merlin.model_check.chain_image_readback import decode

    result = guest()
    assert result.returncode == 0, result.stderr
    report = decode(result.image, result.stdout)
    assert report["numerical_status"] == report["runtime_status"] == "pass"
    assert len(report["comparisons"]) == 12


@pytest.mark.parametrize("guest", [{"invalid_empty": True}], indirect=True)
def test_empty_weight_extent_cannot_admit_a_weight_argument(guest):
    result = guest()
    assert result.returncode != 0
    assert b"CHAIN_ARRAY" not in result.stdout


def test_single_element_reference_mismatch_is_not_state_feedback(guest):
    from merlin.model_check.chain_image_readback import decode

    result = guest(mutate_reference=lambda b: b.__setitem__(slice(34, 38), struct.pack("<f", 24.0)))
    assert result.returncode == 1
    assert result.stdout.count(b"CHAIN_ARRAY v1 ") == 12
    assert b"CHAIN_END v1 4 12 1\n" in result.stdout
    report = decode(result.image, result.stdout)
    assert report["numerical_status"] == "fail"
    assert sum(row["mismatched"] for row in report["comparisons"]) == 1


def test_full_array_transport_still_emits_every_typed_array(guest):
    from merlin.runtime.out_bin import parse_binary_console_details

    result = guest(compact=False)
    assert result.returncode == 0
    outputs, _, frames = parse_binary_console_details(result.stdout)
    assert len(outputs) == len(frames) == 12
    assert frames["p1_s2_out1"].signed
    assert outputs["p1_s2_out0"][0] == [struct.unpack("<I", struct.pack("<f", x))[0] for x in (33, 66)]


@pytest.mark.parametrize("guest", [{"assert_no_transport": True}], indirect=True)
def test_continuous_timing_buffers_observations_and_never_claims_numerics(guest):
    from merlin.model_check.chain_image_readback import decode

    result = guest(timing_only=True)
    assert result.returncode == 0, result.stderr
    assert b"OUT_BIN" not in result.stdout and b"CHAIN_ARRAY" not in result.stdout
    report = decode(result.image, result.stdout)
    assert report["numerical_status"] == "not_evaluated_timing_only"
    assert report["runtime_status"] == "pass"
    assert report["output_array_count"] == 0 and report["declared_output_array_count"] == 12
    # The source counter advances once per read. Eight reads surround four
    # forwards; total also encloses the session boundaries, unlike their sum.
    assert report["timing"]["whole_chain"] == 9
    assert report["timing"]["forward_sum"] == 4
    assert report["timing"]["session_overhead"] == 5
    assert report["timing"]["readback_transport"] == 1
    lines = result.stdout.splitlines(keepends=True)
    timing = next(line for line in lines if line.startswith(b"CHAIN_TIMING"))
    step = lines[0]
    for changed in (
        result.stdout.replace(timing, b""),
        result.stdout.replace(timing, timing * 2),
        result.stdout.replace(timing, b"CHAIN_TIMING v1 3 4 1\n"),
        result.stdout.replace(timing, b"CHAIN_TIMING v1 9 3 1\n"),
        result.stdout.replace(step, step * 2),
        result.stdout + b"CHAIN_ARRAY v1 0 0 0\n",
        result.stdout + b"OUT_BIN_BEGIN bad\n",
        result.stdout.replace(b"CHAIN_END v1 4 0 0", b"CHAIN_END v1 4 12 0"),
    ):
        with pytest.raises(ValueError):
            decode(result.image, changed)


@pytest.mark.parametrize("guest", [{"runtime_errors": True}], indirect=True)
def test_timing_preserves_earlier_runtime_failure_without_numeric_verdict(guest):
    from merlin.model_check.chain_image_readback import decode

    result = guest(timing_only=True)
    assert result.returncode == 2
    report = decode(result.image, result.stdout)
    assert report["runtime_status"] == "fail"
    assert report["numerical_status"] == "not_evaluated_timing_only"


@pytest.mark.parametrize("guest", [{"profile": True}], indirect=True)
def test_timing_refuses_between_forward_profile_transport(guest):
    with pytest.raises(ValueError, match="between forwards"):
        guest(timing_only=True)


@pytest.mark.parametrize("guest", [{"actual": "nan"}, {"actual": "inf"}, {"actual": "bool"}], indirect=True)
def test_nonfinite_actual_and_noncanonical_bool_cannot_pass(guest):
    from merlin.model_check.chain_image_readback import decode

    result = guest()
    report = decode(result.image, result.stdout)
    assert result.returncode == 1
    assert report["numerical_status"] == "fail"
    assert len(report["comparisons"]) == 12


@pytest.mark.parametrize("guest", [{"actual": "zero"}, {"actual": "near"}], indirect=True)
def test_tolerance_and_signed_zero_do_not_require_digest_equality(guest):
    from merlin.model_check.chain_image_readback import decode

    result = guest()
    assert result.returncode == 0
    report = decode(result.image, result.stdout)
    assert report["numerical_status"] == "pass"
    assert any(row["actual_fnv1a64"] != row["reference_fnv1a64"] for row in report["comparisons"])


@pytest.mark.parametrize("guest", [{"actual": "wide"}], indirect=True)
def test_int64_off_by_one_above_double_precision_remains_exact(guest):
    from merlin.model_check.chain_image_readback import decode

    result = guest(mutate_reference=lambda b: b.__setitem__(slice(8, 16), struct.pack("<q", 9007199254740992)))
    report = decode(result.image, result.stdout)
    failed = [row for row in report["comparisons"] if not row["pass"]]
    assert result.returncode == 1 and len(failed) == 1
    assert failed[0]["mismatched"] == 1
    assert failed[0]["max_abs_error"] == 1.0


def test_nonfinite_reference_and_transport_mutations_fail(guest):
    from merlin.model_check.chain_image_readback import decode

    result = guest(mutate_reference=lambda b: b.__setitem__(slice(0, 4), struct.pack("<f", float("nan"))))
    assert result.returncode == 1
    assert decode(result.image, result.stdout)["numerical_status"] == "fail"
    lines = result.stdout.splitlines(keepends=True)
    array_line = next(line for line in lines if line.startswith(b"CHAIN_ARRAY"))
    variants = [
        result.stdout.replace(array_line, b"", 1),
        result.stdout.replace(array_line, array_line * 2, 1),
        result.stdout.replace(array_line, array_line.replace(b" v1 0 0 0 ", b" v1 1 0 0 "), 1),
    ]
    parts = array_line.decode().split()
    parts[11] = "0000000000000000"
    variants.append(result.stdout.replace(array_line, (" ".join(parts) + "\n").encode(), 1))
    variants += [b"", result.stdout[: result.stdout.index(array_line)], result.stdout.rsplit(b"CHAIN_END", 1)[0]]
    for changed in variants:
        with pytest.raises((ValueError, IndexError)):
            decode(result.image, changed)


@pytest.mark.parametrize("guest", [{"runtime_errors": True}], indirect=True)
def test_earlier_runtime_fault_is_not_cleared_by_later_zero(guest):
    from merlin.model_check.chain_image_readback import decode

    result = guest()
    report = decode(result.image, result.stdout)
    assert result.returncode == 2
    assert report["numerical_status"] == "pass"
    assert report["runtime_status"] == "fail"
    assert [row["memref_rank_mismatches"] for row in report["step_cycles"]] == [1, 0, 0, 0]


@pytest.mark.parametrize("guest", [{"profile": True}], indirect=True)
def test_profile_frames_bind_each_recurrent_invocation_without_changing_outputs(guest):
    from merlin.model_check.chain_image_readback import decode

    result = guest()
    report = decode(result.image, result.stdout)
    assert result.returncode == 0 and report["numerical_status"] == "pass"
    assert len(report["group_profiles"]) == 4
    assert all(p["host_gap_cycles"] == 1 and p["calls"] == 0 for p in report["group_profiles"])


def test_explicit_compact_collector_admission_and_roster(guest, tmp_path, monkeypatch):
    from merlin_experiments.post_freeze import compact_chain, firesim

    result = guest()
    console = tmp_path / "uartlog"
    console.write_bytes(result.stdout)
    image = {
        **result.image,
        "schema": firesim.CHAIN_SCHEMA,
        "phase": "phase1",
        "model": "fixture",
        "execution_scope": "complete_chain",
        "compiler_sha256": "0" * 64,
        "contract_sha256": "1" * 64,
        "boot_elf": {"role": "boot_elf", **pin(result.elf)},
    }
    manifest = tmp_path / "image.json"
    manifest.write_text(json.dumps(image))
    assets = [image["boot_elf"], {"role": "chain_manifest", **pin(manifest)}]
    model = {"id": "fixture", "assets": assets, "chain_steps": 4, "output_roster": image["output_roster"]}
    selected = []
    monkeypatch.setattr(compact_chain, "selected", lambda record: selected.append(record) or image)
    firesim._direct_image(model, "phase1", "0" * 64, "1" * 64)
    assert selected == [assets[1]]
    assert model["transport"] == "all_element_comparison_v1"
    assert compact_chain.collect(model, console)["numerical_status"] == "pass"
    model["output_roster"] = model["output_roster"][:-1]
    with pytest.raises(ValueError, match="roster"):
        compact_chain.collect(model, console)
    model["output_roster"] = image["output_roster"]
    image["transport"] = "unselected_new_transport"
    manifest.write_text(json.dumps(image))
    assets[1].update(pin(manifest))
    with pytest.raises(ValueError, match="unknown"):
        firesim._direct_image(model, "phase1", "0" * 64, "1" * 64)
