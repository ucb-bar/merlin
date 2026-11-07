"""A global top-1 match does not establish every token's top-1 match."""

from __future__ import annotations

import hashlib

import numpy as np

from merlin.runtime.backends.zephyr_model import _gate


def test_complete_rank3_gate_reports_each_token_and_binds_full_arrays():
    golden = np.array([[[100.0, 0.0, 0.0], [0.0, 2.0, 1.0]]], dtype=np.float32)
    output = np.array([[[100.0, 0.0, 0.0], [0.0, 1.0, 2.0]]], dtype=np.float32)

    gate = _gate(output, {"w8a8": golden})

    assert gate["comparison_complete"] is True
    assert gate["n_compared"] == gate["n_reference"] == 6
    assert gate["w8a8_argmax"] is True  # Existing global field retains its meaning.
    assert gate["w8a8_argmax_per_token"] is False
    assert gate["w8a8_argmax_tokens_matching"] == 1
    assert gate["w8a8_argmax_tokens_compared"] == 2
    assert gate["w8a8_argmax_output_sha256"] == hashlib.sha256(output.tobytes()).hexdigest()
    assert gate["w8a8_argmax_reference_sha256"] == hashlib.sha256(golden.tobytes()).hexdigest()


def test_partial_rank3_prefix_does_not_claim_a_token_result():
    golden = np.array([[[100.0, 0.0, 0.0], [0.0, 2.0, 1.0]]], dtype=np.float32)
    gate = _gate(golden.ravel()[:3], {"w8a8": golden})

    assert gate["comparison_complete"] is False
    assert "w8a8_argmax_per_token" not in gate
    assert "w8a8_argmax_output_sha256" not in gate


def test_flat_runtime_output_uses_reference_token_shape():
    golden = np.array([[[4.0, 2.0], [1.0, 3.0]]], dtype=np.float32)
    gate = _gate(golden.ravel().copy(), {"w8a8": golden})

    assert gate["w8a8_argmax_per_token"] is True
    assert gate["w8a8_argmax_tokens_matching"] == gate["w8a8_argmax_tokens_compared"] == 2
