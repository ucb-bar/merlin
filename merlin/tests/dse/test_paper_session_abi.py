"""Synthetic coverage for the public MRLNSES2 whole-session ABI."""
from __future__ import annotations

import copy

import pytest
import yaml

from merlin.compare.paper_session_abi import (
    MAGIC,
    CallDescriptor,
    InputEndpoint,
    InputFrame,
    OutputFrame,
    assert_private_data_excluded,
    decode_request,
    decode_response,
    descriptor_from_contract,
    descriptor_from_dict,
    encode_request,
    encode_response,
    load_session_descriptor,
)


def _child(name: str, steps: int, *, stream: tuple[str, int] | None = None,
           state: tuple[str, int, int] | None = None, output: int = 0) -> dict:
    streams = [] if stream is None else [
        {"name": stream[0], "input_arg": stream[1], "key": stream[0]}]
    states = [] if state is None else [
        {"name": state[0], "input_arg": state[1], "output_index": state[2]}]
    return {
        "version": 1, "kind": "autoregressive_decode", "paper_ready": True,
        "stages": [name], "steps": steps,
        "stage_schedule": [{
            "name": name, "steps": steps,
            "execution": "compiled_recurrent" if steps > 1 else "compiled", "timed": True,
        }],
        "states": states, "streams": streams,
        "quality": {"scope": "trajectory", "output_index": output},
    }


def _prefill_decode() -> tuple[dict, dict[str, dict]]:
    root = {
        "version": 2, "kind": "autoregressive_decode", "paper_ready": True,
        "stages": ["prefill", "decode"],
        "stage_schedule": [
            {"name": "prefill", "steps": 1, "execution": "compiled", "timed": True},
            {"name": "decode", "steps": 3,
             "execution": "compiled_recurrent", "timed": True},
        ],
        "programs": [
            {"name": "prefill", "bundle": "stages/prefill", "steps": 1},
            {"name": "decode", "bundle": "stages/decode", "steps": 3},
        ],
        "bindings": [{
            "name": "kv_cache", "from": {"program": "prefill", "output_index": 1},
            "to": {"program": "decode", "input_arg": 1},
        }],
        "states": [{"name": "kv_cache"}], "streams": [],
        "quality": {"scope": "trajectory", "program": "decode"},
    }
    children = {
        "prefill": _child("prefill", 1, stream=("prompt", 0)),
        "decode": _child("decode", 3, stream=("token", 0),
                         state=("kv_cache", 1, 1)),
    }
    return root, children


def _request_frames(descriptor) -> list[InputFrame]:
    return [
        InputFrame(InputEndpoint(program, input_index), step,
                   f"{program}:{input_index}:{step}".encode())
        for program, input_index, step in descriptor.required_input_keys
    ]


def _output_frames(descriptor) -> list[OutputFrame]:
    return [OutputFrame(program, output, step, bytes((step,)))
            for program, output, step in descriptor.required_output_keys]


def test_prefill_decode_roundtrip_requires_the_complete_execution_schedule():
    root, children = _prefill_decode()
    descriptor = descriptor_from_contract(root, child_contracts=children)

    assert [(row.id, row.name, row.steps) for row in descriptor.programs] == [
        (0, "prefill", 1), (1, "decode", 3)]
    assert [(row.program, row.step) for row in descriptor.calls] == [
        (0, 0), (1, 0), (1, 1), (1, 2)]
    assert [row.endpoint.wire_id for row in descriptor.inputs] == ["p0:i0", "p1:i0"]
    assert descriptor.states[0].name == "kv_cache"
    assert descriptor.routes[0].target_input == 1

    request = encode_request(descriptor, reversed(_request_frames(descriptor)))
    assert request.startswith(MAGIC)
    assert decode_request(request, expected_descriptor=descriptor).frames == tuple(
        _request_frames(descriptor))

    outputs = _output_frames(descriptor)
    response = encode_response(descriptor, descriptor.calls, outputs)
    assert decode_response(response, expected_descriptor=descriptor).outputs == tuple(outputs)

    # Three decode outputs are not evidence that the one-time prefill ran.  The
    # response contract requires the complete root schedule as well.
    with pytest.raises(ValueError, match="stage or recurrent step was omitted"):
        encode_response(descriptor, descriptor.calls[1:], outputs)


def test_v1_recurrent_state_and_request_bytes_roundtrip_deterministically():
    contract = {
        "version": 1, "kind": "recurrent_frames", "paper_ready": True,
        "stages": ["visual_encode", "recurrent_step", "predict"], "steps": 3,
        "stage_schedule": [
            {"name": name, "steps": 3, "execution": "compiled_recurrent", "timed": True}
            for name in ("visual_encode", "recurrent_step", "predict")
        ],
        "streams": [{"name": "frame", "input_arg": 0, "key": "frame"}],
        "states": [
            {"name": "hidden_state", "input_arg": 1, "output_index": 1},
            {"name": "cell_state", "input_arg": 2, "output_index": 2},
        ],
        "quality": {"scope": "trajectory", "output_index": 0},
    }
    descriptor = descriptor_from_contract(contract)
    assert [(row.name, row.role, row.frames) for row in descriptor.inputs] == [
        ("frame", "stream", 3), ("hidden_state", "initial_state", 1),
        ("cell_state", "initial_state", 1),
    ]
    frames = _request_frames(descriptor)
    encoded_a = encode_request(descriptor, frames)
    encoded_b = encode_request(descriptor, list(reversed(frames)))
    assert encoded_a == encoded_b
    decoded = decode_request(encoded_a)
    assert decoded.descriptor.sha256 == descriptor.sha256
    assert decoded.frames == tuple(frames)


def test_prefix_denoise_action_counts_are_preserved_without_model_dispatch():
    root = {
        "version": 2, "kind": "action_chunk", "paper_ready": True,
        "stages": ["prefix_encode", "flow_denoise", "action_decode"],
        "stage_schedule": [
            {"name": "prefix_encode", "steps": 1, "execution": "compiled", "timed": True},
            {"name": "flow_denoise", "steps": 2,
             "execution": "compiled_recurrent", "timed": True},
            {"name": "action_decode", "steps": 1, "execution": "compiled", "timed": True},
        ],
        "programs": [
            {"name": "prefix_encode", "bundle": "stages/prefix", "steps": 1},
            {"name": "flow_denoise", "bundle": "stages/flow", "steps": 2},
            {"name": "action_decode", "bundle": "stages/action", "steps": 1},
        ],
        "bindings": [
            {"name": "flow_state_seed",
             "from": {"program": "prefix_encode", "output_index": 0},
             "to": {"program": "flow_denoise", "input_index": 0}},
            {"name": "decoded_action",
             "from": {"program": "flow_denoise", "output_index": 0},
             "to": {"program": "action_decode", "input_index": 0}},
        ],
        "states": ["flow_state"], "streams": [],
        "quality": {"scope": "trajectory", "program": "action_decode"},
    }
    children = {
        "prefix_encode": _child("prefix_encode", 1, stream=("observation", 0)),
        "flow_denoise": _child("flow_denoise", 2, state=("flow_state", 0, 0)),
        "action_decode": _child("action_decode", 1),
    }
    descriptor = descriptor_from_contract(root, child_contracts=children)
    assert [row.steps for row in descriptor.stages] == [1, 2, 1]
    assert [(row.program, row.step) for row in descriptor.calls] == [
        (0, 0), (1, 0), (1, 1), (2, 0)]


def test_malformed_duplicate_missing_and_private_frames_fail_closed():
    root, children = _prefill_decode()
    descriptor = descriptor_from_contract(root, child_contracts=children)
    frames = _request_frames(descriptor)

    with pytest.raises(ValueError, match="duplicate input frame"):
        encode_request(descriptor, [*frames, frames[0]])
    with pytest.raises(ValueError, match="missing="):
        encode_request(descriptor, frames[:-1])
    with pytest.raises(ValueError, match="magic"):
        decode_request(b"NOTASESSION")
    with pytest.raises(ValueError, match="truncated"):
        decode_request(encode_request(descriptor, frames)[:-1])

    leaked = copy.deepcopy(descriptor.to_dict())
    leaked["golden"] = "private/session_goldens.npz"
    with pytest.raises(ValueError, match="forbidden private/artifact field"):
        descriptor_from_dict(leaked)
    with pytest.raises(ValueError, match="forbidden private/artifact field"):
        assert_private_data_excluded({"nested": {"quality_golden": b"secret"}})


def test_stage_omissions_duplicates_and_bundle_traversal_are_rejected(tmp_path):
    root, children = _prefill_decode()
    omitted = copy.deepcopy(root)
    omitted["stage_schedule"] = omitted["stage_schedule"][1:]
    with pytest.raises(ValueError, match="exactly one row per declared stage"):
        descriptor_from_contract(omitted, child_contracts=children)

    duplicate = copy.deepcopy(root)
    duplicate["programs"][1]["name"] = "prefill"
    with pytest.raises(ValueError, match="duplicate program names"):
        descriptor_from_contract(duplicate, child_contracts={"prefill": children["prefill"]})

    capture = tmp_path / "capture"
    capture.mkdir()
    escaped = copy.deepcopy(root)
    escaped["programs"][0]["bundle"] = "../outside"
    (capture / "session_contract.yaml").write_text(
        yaml.safe_dump(escaped), encoding="utf-8")
    with pytest.raises(ValueError, match="normalized relative path"):
        load_session_descriptor(capture)

    with pytest.raises(ValueError, match="every v2 program requires"):
        descriptor_from_contract(root, child_contracts={"decode": children["decode"]})


def test_descriptor_rejects_duplicate_or_noncanonical_schedule_rows():
    root, children = _prefill_decode()
    descriptor = descriptor_from_contract(root, child_contracts=children)
    raw = descriptor.to_dict()
    raw["calls"][1] = copy.deepcopy(raw["calls"][0])
    raw["calls"][1]["ordinal"] = 1
    with pytest.raises(ValueError, match="duplicate program calls"):
        descriptor_from_dict(raw)

    with pytest.raises(ValueError, match="exact whole-session schedule"):
        encode_response(
            descriptor,
            [CallDescriptor(0, 1, 0), *descriptor.calls[1:]],
            _output_frames(descriptor),
        )
