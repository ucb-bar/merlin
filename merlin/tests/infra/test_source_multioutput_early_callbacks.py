"""The ordinary owner freezes final decoders and core helpers before callbacks."""

import pytest
from test_source_multioutput_freshness import original_callbacks, selection, source

from merlin.compile.scheduling import multioutput_fusion_check as checker
from merlin.llvmlower.source_multioutput import discover_multioutput


def wrong_decoder(image):
    raise AssertionError("changed final decoder must never execute")


def wrong_verifier(owner, image, calls):
    raise AssertionError("changed original ISA image verifier must never execute")


def wrong_memory(owner, space, start, size):
    raise AssertionError("changed provenance reader must never execute")


def wrong_elementwise(*args, **kwargs):
    raise AssertionError("changed scalar semantic checker must never execute")


@pytest.mark.parametrize("stage", ["first-validator", "mapper-none"])
@pytest.mark.parametrize("role", ["final-decoder", "image-verifier"])
def test_original_final_decoder_code_is_frozen_before_mapping(tmp_path, stage, role):
    module = source()
    validations, mappings = [], []
    changed = False

    def corrupt():
        nonlocal changed
        if not changed:
            changed = True
            replacement = wrong_decoder if role == "final-decoder" else wrong_verifier
            # Preserve the original closure cardinality. This remains the SAME
            # callback object with different executable code, not a dataclass
            # field replacement that an ordinary value snapshot already sees.
            target.__code__ = replacement.__code__.replace(co_freevars=before.co_freevars)

    def validate_hook():
        validations.append(1)
        if stage == "first-validator":
            corrupt()

    def map_hook(value):
        mappings.append(value)
        if stage == "mapper-none":
            corrupt()
        return None

    validate, mapper, _ = original_callbacks(tmp_path, validate_hook, map_hook)
    selected, singles, prepared = selection(tmp_path, mapper)
    target = (
        selected.encoding.program_decoder.decode
        if role == "final-decoder"
        else selected.selected_contracts.verify_image.__func__
    )
    before = target.__code__
    try:
        with pytest.raises(ValueError, match="changed|replan"):
            discover_multioutput(module, selected, validate)
    finally:
        target.__code__ = before
    assert changed and not singles and not prepared
    assert len(mappings) == (0 if stage == "first-validator" else 1)
    if stage == "first-validator":
        assert validations == [1]


@pytest.mark.parametrize("helper", ["elementwise-binding", "memory-method-code"])
def test_first_original_validator_cannot_rebind_core_semantic_helpers(tmp_path, monkeypatch, helper):
    module = source()
    validations, mappings = [], []
    before_memory = checker._Memory.read.__code__
    before_elementwise = checker._elementwise

    def validate_hook():
        validations.append(1)
        if len(validations) != 1:
            return
        if helper == "elementwise-binding":
            monkeypatch.setattr(checker, "_elementwise", wrong_elementwise)
        else:
            checker._Memory.read.__code__ = wrong_memory.__code__.replace(co_freevars=before_memory.co_freevars)

    validate, mapper, _ = original_callbacks(tmp_path, validate_hook, lambda value: mappings.append(value))
    selected, singles, prepared = selection(tmp_path, mapper)
    try:
        with pytest.raises(ValueError, match="changed|replan"):
            discover_multioutput(module, selected, validate)
    finally:
        checker._Memory.read.__code__ = before_memory
        monkeypatch.setattr(checker, "_elementwise", before_elementwise)
    assert validations == [1] and not mappings and not singles and not prepared
