"""Two real lowered stages, complete observations and unchanged carried state."""

import hashlib
import json
import struct
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from merlin.common.paths import runtime_dir
from merlin.llvmlower import session_freestanding as F


def capture(root):
    for name, steps, weight in [("first", 1, [5, 7]), ("next", 3, [-2, 3])]:
        stage = root / "stages" / name
        stage.mkdir(parents=True)
        (stage / "model.mlir").write_text("""module {
  func.func @forward(%w: tensor<2xf32>, %x: tensor<2xf32>, %s: tensor<2xf32>)
      -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) {
    %e = tensor.empty() : tensor<2xf32>
    %y = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>,
        affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]}
        ins(%w, %x, %s : tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
        outs(%e : tensor<2xf32>) {
      ^bb0(%a: f32, %b: f32, %c: f32, %d: f32):
        %t = arith.addf %c, %b : f32
        %r = arith.addf %t, %a : f32
        linalg.yield %r : f32
    } -> tensor<2xf32>
    return %y, %x, %s, %w : tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>
  }
}""")
        data = np.array(weight, "<f4").tobytes()
        header = json.dumps(dict(w=dict(dtype="F32", shape=[2], data_offsets=[0, len(data)]))).encode()
        (stage / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + data)
        (stage / "weights.safetensors.manifest.json").write_text(
            json.dumps(
                {
                    "0": dict(kind="param", weight="w"),
                    "1": dict(kind="input", name="frame"),
                    "2": dict(kind="input", name="state"),
                }
            )
        )
        (stage / "input_order.json").write_text(json.dumps(dict(frame=0, state=1)))
        np.savez(stage / "inputs.npz", in0=np.array([3, 4], np.float32), in1=np.array([1, 2], np.float32))
        if steps > 1:
            np.savez(stage / "streams.npz", frames=np.array([[2, 4], [6, 8], [10, 12]], np.float32))
            np.savez(stage / "quality.npz", output0=np.zeros((3, 2), np.float32))
            (stage / "session_contract.yaml").write_text(
                yaml.safe_dump(
                    dict(
                        version=1,
                        kind="autoregressive_decode",
                        paper_ready=False,
                        stages=[name],
                        steps=steps,
                        inputs="streams.npz",
                        states=[dict(name="state", input_arg=2, output_index=0)],
                        streams=[dict(name="frame", input_arg=1, key="frames")],
                        quality=dict(scope="trajectory", golden="quality.npz", key="output0", output_index=0),
                    )
                )
            )
    contract = dict(
        version=2,
        stages=["first", "next"],
        programs=[dict(name=n, bundle=f"stages/{n}", steps=s) for n, s in [("first", 1), ("next", 3)]],
        stage_schedule=[
            dict(name=n, steps=s, timed=True, execution="compiled") for n, s in [("first", 1), ("next", 3)]
        ],
        bindings=[
            dict(
                name="state", **{"from": dict(program="first", output_index=0), "to": dict(program="next", input_arg=2)}
            )
        ],
        states=[dict(name="state")],
        streams=[],
        quality=dict(program="next"),
    )
    (root / "session_contract.yaml").write_text(yaml.safe_dump(contract))
    return root


def limits():
    return F.SessionMemoryLimits(
        weights_base=4096, address_limit=8192, max_weights_bytes=1024, max_static_io_bytes=8192, max_calls=4
    )


def test_streamed_weight_offsets_preserve_sources_and_every_output(tmp_path):
    root = capture(tmp_path / "source")
    original = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    out = tmp_path / "generated"
    record = F.generate(root, out, limits=limits())
    assert record["calls"] == 4 and len(record["outputs"]) == 8
    rows = record["weights"]
    assert [r["offset"] for r in rows] == [0, 64]
    raw = (out / "weights.bin").read_bytes()
    for row in rows:
        assert raw[row["offset"] : row["offset"] + row["bytes"]] == Path(row["path"]).read_bytes()
    assert all(p.read_bytes() == data for p, data in original.items())
    assert record["unbudgeted"] == ["code", "stack", "descriptors", "dynamic_activations"]
    with pytest.raises(ValueError, match="fresh"):
        F.generate(root, out, limits=limits())


@pytest.mark.parametrize(
    "field,value,diagnostic",
    [
        ("max_calls", 3, "call count"),
        ("max_static_io_bytes", 1, "static I/O"),
        ("max_weights_bytes", 8, "weights exceed"),
        ("address_limit", 4100, "weights exceed"),
        ("weights_base", True, "signed-64"),
        ("weights_base", 4097, "alignment"),
    ],
)
def test_explicit_construction_and_address_limits_refuse(tmp_path, field, value, diagnostic):
    root = capture(tmp_path / "source")
    with pytest.raises(ValueError, match=diagnostic):
        F.generate(root, tmp_path / "generated", limits=replace(limits(), **{field: value}))


@pytest.mark.parametrize("mutation", ["source", "added_source", "weights", "weight_metadata"])
def test_generation_refuses_changed_original_source_or_stage_weight_identity(tmp_path, monkeypatch, mutation):
    root = capture(tmp_path / "source")
    original_generate = F.session_bundle.generate

    def changed(*args):
        record = original_generate(*args)
        if mutation == "source":
            path = root / "stages/first/model.mlir"
            path.write_text(path.read_text() + "\n// changed original\n")
        elif mutation == "added_source":
            (root / "new-input").write_bytes(b"late")
        elif mutation == "weights":
            (Path(record["programs"][0]["generated"]) / "weights.bin").write_bytes(b"changed")
        else:
            record["programs"][0]["weights_sha256"] = "0" * 64
        return record

    monkeypatch.setattr(F.session_bundle, "generate", changed)
    with pytest.raises(ValueError, match="changed|differ"):
        F.generate(root, tmp_path / "generated", limits=limits())


def test_original_source_is_frozen_before_initial_session_read(tmp_path, monkeypatch):
    root = capture(tmp_path / "source")
    original_load = F.session_bundle.load

    def changed(*args):
        result = original_load(*args)
        path = root / "stages/first/model.mlir"
        path.write_text(path.read_text() + "\n// changed after first read\n")
        return result

    monkeypatch.setattr(F.session_bundle, "load", changed)
    with pytest.raises(ValueError, match="original session source changed"):
        F.generate(root, tmp_path / "generated", limits=limits())


def run(command, directory):
    command = [str(v) for v in command]
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=60)
    log = directory / "native-commands.json"
    rows = json.loads(log.read_text()) if log.exists() else []
    rows.append(
        dict(argv=command, cwd=str(directory), returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    )
    log.write_text(json.dumps(rows, indent=2) + "\n")
    assert result.returncode == 0, result.stderr + result.stdout
    return result


def guard_stage(stage, index):
    """Surround the actual generated native I/O arrays, without changing their values."""
    guarded, lines = [], []
    for line in (stage / "model_io.h").read_text().splitlines():
        if line.startswith(
            (
                "static float merlin_in_",
                "static const float merlin_stream_",
                "static _Alignas(64) unsigned char merlin_out_",
            )
        ):
            name = line.partition("[")[0].split()[-1]
            initializer = line.partition("=")[2].strip().removesuffix(";") or "{0}"
            size = int(line.partition("[")[2].partition("]")[0] or len(initializer.strip("{}").split(",")))
            dtype = "unsigned char" if "unsigned char" in line else "float"
            const = "const " if "const " in line else ""
            fill = "{" + ",".join(["165"] * 64) + "}"
            lines += [
                f"static {const}struct {{ unsigned char before[64]; _Alignas(64) {dtype} value[{size}];"
                f" unsigned char after[64]; }} guarded_{name} = {{{fill},{initializer},{fill}}};",
                f"#define {name} guarded_{name}.value",
            ]
            guarded.append(name)
        else:
            lines.append(line)
    (stage / "model_io.h").write_text("\n".join(lines) + "\n")
    checks = "\n".join(
        f"if(guarded_{name}.before[i]!=165 || guarded_{name}.after[i]!=165) return 1;" for name in guarded
    )
    with (stage / "stage_adapter.c").open("a") as stream:
        stream.write(f"\nint stage_{index}_guards(void) {{for(int i=0;i<64;i++){{{checks}}} return 0;}}\n")


def test_actual_two_stage_freestanding_session_observes_all_outputs_and_state(tmp_path):
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.lower import lower_model_file

    root = capture(tmp_path / "source")
    generated = tmp_path / "generated"
    record = F.generate(root, generated, limits=limits())
    cc = toolchain.clang()
    llvm = cc.parent
    objects = []
    rt = runtime_dir() / "c"
    for index, row in enumerate(record["session"]["programs"]):
        stage = Path(row["generated"])
        guard_stage(stage, index)
        lowered = lower_model_file(
            stage / "model.renamed.mlir",
            tmp_path / f"lower{index}",
            targets=(),
            textual=True,
            hoist_static_allocs=False,
        )
        raw, obj = tmp_path / f"raw{index}.o", tmp_path / f"stage{index}.o"
        run([cc, "-O2", "-c", lowered.ll_path, "-o", raw], tmp_path)
        symbols = run([llvm / "llvm-nm", "--defined-only", "--extern-only", "--format=posix", raw], tmp_path)
        preserve = {row["entrypoint"], "_mlir_ciface_" + row["entrypoint"]}
        renames = [
            f"--redefine-sym={name}=merlin_stage_{index}__{name}"
            for line in symbols.stdout.splitlines()
            if (name := line.split()[0]) not in preserve
        ]
        run([llvm / "llvm-objcopy", *renames, raw, obj], tmp_path)
        objects.append(obj)
        for name in ["model_call", "stage_adapter"]:
            obj = tmp_path / f"{name}{index}.o"
            run([cc, "-O2", "-I", rt, "-I", stage, "-c", stage / f"{name}.c", "-o", obj], tmp_path)
            objects.append(obj)
    # Link one unchanged payload image, with guard bytes surrounding the native blob.
    blob = generated / "weights.bin"
    assembly = tmp_path / "weights.S"
    assembly.write_text(f'''.section .rodata
.balign 64
.global weights_before
weights_before:
.fill 64,1,165
.global session_weights
session_weights:
.incbin "{blob}"
.global weights_after
weights_after:
.fill 64,1,165
.section .note.GNU-stack,"",@progbits
''')
    weight_object = tmp_path / "weights_blob.o"
    run([cc, "-c", assembly, "-o", weight_object], tmp_path)
    header = tmp_path / "address.h"
    header.write_text(
        "#include <stdint.h>\nextern const unsigned char session_weights[];\n"
        "#define MERLIN_SESSION_WEIGHTS_BASE_ADDR ((uintptr_t)session_weights)\n"
    )
    main = tmp_path / "main.c"
    main.write_text("""#include <stdio.h>
#include "merlin_session_freestanding.h"
extern int stage_0_guards(void), stage_1_guards(void);
extern const unsigned char weights_before[],weights_after[];
static int observe(int p,long s,int o,const void *data,size_t bytes,void *ctx) {
  const float *x=data; int *calls=ctx;
  if(bytes!=8 || stage_0_guards() || stage_1_guards()) return 1;
  for(int i=0;i<64;i++) if(weights_before[i]!=165 || weights_after[i]!=165) return 1;
  printf("%d %ld %d %.0f %.0f\\n",p,s,o,x[0],x[1]); ++*calls; return 0;
}
int main(void) {
  int calls=0;
  if(merlin_session_run_freestanding(0,0)!=-1) return 1;
  if(merlin_session_run_freestanding(observe,&calls)||calls!=16) return 2;
  return 0;
}
""")
    sources = [
        generated / "merlin_session.c",
        generated / "merlin_session_freestanding.c",
        rt / "merlin_model.c",
        main,
        weight_object,
    ]
    executable = tmp_path / "session"
    command = [cc, "-O2", "-I", rt, "-I", generated, "-include", header, *sources, *objects, "-lm", "-o", executable]
    run(command, tmp_path)
    actual = run([executable], tmp_path).stdout
    # Independent ordered reference, including every readonly echo and prior state.
    expected = []
    state = np.array([1, 2], np.float32)
    for program, frames, weight in [(0, [[3, 4]], [5, 7]), (1, [[2, 4], [6, 8], [10, 12]], [-2, 3])]:
        for step, frame in enumerate(frames):
            frame, weight = np.array(frame, np.float32), np.array(weight, np.float32)
            next_state = (state + frame) + weight
            for output, value in enumerate([next_state, frame, state, weight]):
                expected.append(f"{program} {step} {output} {value[0]:.0f} {value[1]:.0f}")
            state = next_state
    assert actual.splitlines() == expected
    # Same-shaped wrong binding reaches real execution but fails the complete observer.
    scheduler = generated / "merlin_session.c"
    original = scheduler.read_text()
    assert "merlin_stage_0_output(0)" in original
    scheduler.write_text(original.replace("merlin_stage_0_output(0)", "merlin_stage_0_output(1)"))
    wrong_binding = tmp_path / "wrong-binding"
    run([*command[:-1], wrong_binding], tmp_path)
    assert run([wrong_binding], tmp_path).stdout.splitlines() != expected
    copy_line = next(line for line in original.splitlines() if "memcpy(merlin_stage_1_input(2)" in line)
    scheduler.write_text(original.replace(copy_line, "/* ignored cross-stage state copy */"))
    ignored_copy = tmp_path / "ignored-state-copy"
    run([*command[:-1], ignored_copy], tmp_path)
    assert run([ignored_copy], tmp_path).stdout.splitlines() != expected
    scheduler.write_text(original)
    # A wrong stage base cannot borrow another stage's equal-shaped weight payload.
    runner = generated / "merlin_session_freestanding.c"
    original = runner.read_text()
    assert "+ 64ULL" in original
    runner.write_text(original.replace("+ 64ULL", "+ 0ULL"))
    wrong_base = tmp_path / "wrong-weight-base"
    run([*command[:-1], wrong_base], tmp_path)
    assert run([wrong_base], tmp_path).stdout.splitlines() != expected
    runner.write_text(original.replace("step, 0))", "0, 0))"))
    repeated_step = tmp_path / "repeated-step-zero"
    run([*command[:-1], repeated_step], tmp_path)
    assert run([repeated_step], tmp_path).stdout.splitlines() != expected
    # The observer is mandatory and no declared output may disappear silently.
    runner.write_text(original.replace("if (observe(1, step, 3,", "if (0 && observe(1, step, 3,"))
    omitted = tmp_path / "omitted-observation"
    run([*command[:-1], omitted], tmp_path)
    assert subprocess.run([omitted], capture_output=True, timeout=60).returncode == 2
    runner.write_text(original)
    # Compile the actual consumer without hosted C assumptions as well.
    run(
        [
            cc,
            "-ffreestanding",
            "-fno-builtin",
            "-I",
            generated,
            "-include",
            header,
            "-c",
            runner,
            "-o",
            tmp_path / "freestanding.o",
        ],
        tmp_path,
    )
    (tmp_path / "actual.txt").write_text(actual)
    (tmp_path / "expected.txt").write_text("\n".join(expected) + "\n")
    assert hashlib.sha256(blob.read_bytes()).hexdigest() == record["weights_sha256"]
