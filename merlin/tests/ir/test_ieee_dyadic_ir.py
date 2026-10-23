"""Execute emitted RNE, including dtype extremes and poisoned shift controls."""

import pytest

from merlin.targetgen.native_ieee_certificate import boundary_words, convert_word, verify_outputs
from merlin.targetgen.native_ieee_ir_check import conversion_function


@pytest.mark.parametrize(
    "source,target,policy",
    [
        ("f16", "f32", "payload_quiet"),
        ("bf16", "f32", "payload_preserve"),
        ("f32", "f16", "payload_quiet"),
        ("f32", "bf16", "positive_quiet"),
    ],
)
def test_emitted_conversion_matches_complete_storage_or_rounding_cells(source, target, policy):
    if source == "f32":
        words = boundary_words(target)
    else:
        words = range(65536)
    proposed = conversion_function(source, target, policy)(words)
    verify_outputs(words, proposed, source, target, policy)


def test_wrong_nan_quieting_cannot_receive_a_certificate():
    words = range(65536)
    correct = [convert_word(v, "f16", "f32") for v in words]

    def wrong(word, source, target, policy):
        return convert_word(word, source, target, "payload_preserve")

    with pytest.raises(ValueError, match="conversion proof refused"):
        verify_outputs(words, correct, "f16", "f32", "payload_quiet", converter=wrong)


def test_wrong_rounding_at_a_midpoint_is_refused():
    # Exactly halfway between 1 and its next half-precision neighbor: RNE picks 1.
    words = [0x3F801000]
    with pytest.raises(ValueError, match="conversion proof refused"):
        verify_outputs(words, [0x3C01], "f32", "f16", "payload_quiet")
