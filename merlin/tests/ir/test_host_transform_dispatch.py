"""Public dispatch stays closed while legacy callback and custody identities agree."""

from pathlib import Path

import pytest

from merlin.llvmlower import host_transform_dispatch as D
from merlin.llvmlower.bounded_rne_custody import BoundedRNECustody
from merlin.llvmlower.late_quant_rne import merlin_host_llvm_transform


def test_legacy_callback_and_error_imports_are_exact_aliases():
    from merlin.runtime.backends import _model_host_transform as legacy
    from merlin.runtime.backends import spike_model

    assert legacy._transform_host_ir is D._transform_host_ir
    assert spike_model._transform_host_ir is D._transform_host_ir
    assert legacy.SpikeModelError is D.HostLLVMTransformError
    assert spike_model.SpikeModelError is D.HostLLVMTransformError


def test_original_custody_pins_public_dispatcher_and_refuses_changed_code(tmp_path, monkeypatch):
    assembler = tmp_path / "llvm-as"
    assembler.write_bytes(b"selected assembler bytes")
    callback = merlin_host_llvm_transform(tmp_path, host_isa="portable")
    owner = BoundedRNECustody(callback)
    assert Path(D.__file__).resolve() in {pin.path for pin in owner._owners}
    assert not any("runtime/backends" in str(pin.path) for pin in owner._owners)
    assert owner._checker[-1][0] is D._transform_host_ir
    owner.revalidate()
    monkeypatch.setattr(D, "_transform_host_ir", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="footprint implementation changed"):
        owner.revalidate()


@pytest.mark.parametrize("invalid", ["mutated_source", "outside_workdir", "missing_result"])
def test_public_dispatcher_retains_source_and_output_rejections(tmp_path, invalid):
    source = tmp_path / "source.ll"
    source.write_text("original LLVM")
    outside = tmp_path / "outside.ll"
    outside.write_text("unowned output")

    def transform(original, work):
        if invalid == "mutated_source":
            original.write_text("changed LLVM")
            return original
        return outside if invalid == "outside_workdir" else work / "missing.ll"

    with pytest.raises(D.HostLLVMTransformError):
        D._transform_host_ir(source, tmp_path / "work", transform)
