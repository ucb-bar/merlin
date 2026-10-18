"""Selected persistent settings compose with source arithmetic and allocation."""

import json
from dataclasses import replace

import numpy as np
import pytest
from test_pointwise_source import case as memory_case
from test_pointwise_source import execute as execute
from test_pointwise_source import source_program
from test_readout_model_fusion import _EXECUTOR, KEYS, OPS, WORD, command, params

from merlin.compile.scheduling.pointwise_source import compile_source_pointwise
from merlin.compile.scheduling.program_limits import DecodedInstruction
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets
from merlin.compile.scheduling.temporary_storage import StorageEffects
from merlin.sched.ir import Call, Const
from merlin.targetgen.isa_dialect_gen import decode_instruction, encode_instruction

CONFIGURED_OPS = (*OPS, "setup", "reset")
LOW, HIGH = (1 << 32) - 128, 127


def case(tmp_path, *, m=3, n=5, separate=False, repetitions=4, mutate=None):
    def declare(facts, software, fusion):
        facts["state_fields"] = 2
        facts["codes"] = list(range(len(CONFIGURED_OPS)))

        def ref(*parts):
            return {"fact": ["facts", *parts]}

        software["isa"]["resources"].append(
            dict(name="settings", kind="state", unit="fields", capacity=ref("state_fields"))
        )
        config = software["scheduling"]["configuration"]
        for name in OPS:
            config[name] = dict(reads=[], writes=[], clobbers=[], ordered=True, pure_write=False)
        for name in ("finish", "fused"):
            config[name]["reads"] = [
                dict(resource="settings", index={"constant": index}, value={"constant": value})
                for index, value in enumerate((LOW, HIGH))
            ]
            instruction = next(row for row in software["isa"]["instructions"] if row["name"] == name)
            instruction["effects"]["reads"].append("settings")
        template = software["isa"]["instructions"][0]
        for index, name in enumerate(("setup", "reset"), len(OPS)):
            instruction = json.loads(json.dumps(template))
            instruction.update(name=name, effects=dict(reads=[], writes=["settings"]))
            instruction["encoding"][0]["fixed"] = ref("codes", index)
            instruction["semantics"]["primitive"] = "neutral." + name
            software["isa"]["instructions"].append(instruction)
            software["scheduling"]["storage"]["instructions"][name] = dict(
                accesses=[], completion="synchronous", token=None, waits=[], exact_copy=False
            )
            field = dict(resource="settings", index={"argument": "src"})
            config[name] = dict(
                reads=[],
                writes=[dict(field, value={"argument": "source"})] if name == "setup" else [],
                clobbers=[] if name == "setup" else [field],
                ordered=True,
                pure_write=name == "setup",
            )
        if mutate is not None:
            mutate(facts, software, fusion)

    options = memory_case(tmp_path, m=m, n=n, separate=separate, mutate=declare)
    selected, original = options["selected_contracts"], options["adapter"]

    def project(call):
        if call.instr in ("setup", "reset"):
            return StorageEffects(repr((call.instr, call.args)), (), "synchronous")
        return original.project(call)

    options["adapter"] = replace(original, project=project)

    def encode(kernel):
        return b"".join(
            encode_instruction(selected.isa, call.instr, params(call)).to_bytes(WORD.size, "little")
            for call in kernel.body
        )

    def decode(image):
        calls = []
        for i in range(0, len(image), WORD.size):
            name, values = decode_instruction(selected.isa, int.from_bytes(image[i : i + WORD.size], "little"))
            calls.append(Call(name, tuple((key, Const(values[key])) for key in KEYS)))
        return tuple(calls)

    def instructions(image):
        return tuple(
            DecodedInstruction(
                i * WORD.size, WORD.size, project(call).accesses, completion="synchronous", relocatable=True
            )
            for i, call in enumerate(decode(image))
        )

    codec = options["encoding"]
    options["encoding"] = replace(
        codec, encode=encode, decode_calls=decode, program_decoder=replace(codec.program_decoder, decode=instructions)
    )
    setup = (command("setup", src=0, source=LOW), command("setup", src=1, source=HIGH)) * repetitions
    kernel = options["kernel"]
    options["kernel"] = replace(kernel, body=(*setup, kernel.body[0], *setup, kernel.body[1]))
    options["limits"] = replace(options["limits"], max_bytes=128 * WORD.size, max_instructions=128)
    options["configuration_qualification_identity"] = "independent-neutral-persistent-settings-v1"
    return options


EXECUTOR = (
    _EXECUTOR.replace(
        "uint64_t commands=0,reads=0,writes=0;",
        """
 uint32_t settings[2]={19,23};
 uint64_t commands=0,reads=0,writes=0;
""",
    )
    .replace(
        "if(r>=10||w>=10||weight>=10||bias>=10||!m||!n)return 2;",
        """
 if(op==5||op==6){
  if(r>=2)return 10;
  settings[r]=op==5?src:0;
  if(op==5&&wrong==2)settings[0]++;
  continue;
 }
 if(r>=10||w>=10||weight>=10||bias>=10||!m||!n)return 2;
""",
    )
    .replace(
        "sum < -128 ? -128 : sum > 127 ? 127 : sum",
        """
 sum < signed32(settings[0]) ? signed32(settings[0]) :
 sum > signed32(settings[1]) ? signed32(settings[1]) : sum
""",
    )
    .replace(
        "counts[0]=commands;counts[1]=reads;counts[2]=writes;",
        """
 counts[0]=commands;counts[1]=reads;counts[2]=writes;
 counts[3]=settings[0];counts[4]=settings[1];
""",
    )
)


@pytest.mark.parametrize("execute", [dict(source=EXECUTOR, counters=5)], indirect=True, ids=["stateful-machine"])
@pytest.mark.parametrize("m,n,separate,repetitions", [(3, 5, False, 4), (2, 11, True, 16), (1, 64, False, 1)])
def test_checked_source_executes_with_redundant_setup_removed(tmp_path, execute, m, n, separate, repetitions):
    module, operation = source_program(m=m, n=n)
    options = case(tmp_path, m=m, n=n, separate=separate, repetitions=repetitions)
    x = (np.arange(m * n, dtype=np.int32) * 11 - 128).reshape(m, n)
    bias = np.full((m, n), (1 << 31) - 2, "<i4")
    wide = x.astype(np.int64) + bias.astype(np.int64)
    expected = np.clip((wide + (1 << 31)) % (1 << 32) - (1 << 31), -128, 127).astype("i1").tobytes()
    observations = []
    for enabled in (False, True):
        result = compile_source_pointwise(
            module, operation, budgets=ScheduleBudgets(max_fusions=0, eliminate_configuration=enabled), **options
        )
        actual, counters = execute(result.images(), {"x": x.tobytes(), "b0": bias.tobytes()}, separate=separate)
        assert actual[: m * n] == expected
        assert counters[0] == (4 if enabled else 4 * repetitions + 2)
        assert counters[1:3] == (12 * m * n, 5 * m * n) and counters[3:] == (LOW, HIGH)
        assert result.conformance["configuration"]["known_final_state"] == {"settings:0": LOW, "settings:1": HIGH}
        observations.append(counters)
    assert observations[0][1:] == observations[1][1:]
    # A supposed setter with a hidden pulse is not idempotent. Independent
    # execution rejects the implementation despite plausible declarations.
    bad, _ = execute(result.images(), {"x": x.tobytes(), "b0": bias.tobytes()}, separate=separate, wrong=2)
    assert bad[: m * n] != expected


@pytest.mark.parametrize("fault", ["missing_initial", "wrong_value", "reset", "qualification", "capacity"])
def test_invalid_state_does_not_authorize_source_lowering(tmp_path, fault):
    module, operation = source_program()
    before = str(module)
    options = case(
        tmp_path,
        repetitions=1,
        mutate=((lambda facts, software, fusion: facts.update(state_fields=1)) if fault == "capacity" else None),
    )
    kernel = options["kernel"]
    if fault == "missing_initial":
        options["kernel"] = replace(kernel, body=tuple(call for call in kernel.body if call.instr != "setup"))
    elif fault == "wrong_value":
        options["kernel"] = replace(
            kernel,
            body=tuple(
                command("setup", src=0, source=LOW + 1) if call.instr == "setup" and params(call)["src"] == 0 else call
                for call in kernel.body
            ),
        )
    elif fault == "reset":
        options["kernel"] = replace(kernel, body=(*kernel.body[:-1], command("reset", src=0), kernel.body[-1]))
    elif fault == "qualification":
        options["configuration_qualification_identity"] = None
    expected = {
        "missing_initial": "established",
        "wrong_value": "established",
        "reset": "established",
        "qualification": "independent configuration",
        "capacity": "RTL state capacity",
    }
    with pytest.raises(ValueError, match=expected[fault]):
        compile_source_pointwise(module, operation, **options)
    assert str(module) == before


@pytest.mark.parametrize("fault", ["missing_frame", "state_write", "unordered", "copy"])
def test_scalar_state_contracts_fail_closed(tmp_path, fault):
    def change(facts, software, fusion):
        config = software["scheduling"]["configuration"]
        if fault == "missing_frame":
            del config["finish"]
        elif fault == "state_write":
            instruction = next(row for row in software["isa"]["instructions"] if row["name"] == "finish")
            instruction["effects"]["writes"].append("settings")
            config["finish"]["writes"] = [dict(resource="settings", index={"constant": 0}, value={"constant": 0})]
        elif fault == "unordered":
            config["finish"]["ordered"] = False
        else:
            software["scheduling"]["storage"]["instructions"]["finish"]["exact_copy"] = True

    with pytest.raises(ValueError):
        case(tmp_path, mutate=change)
