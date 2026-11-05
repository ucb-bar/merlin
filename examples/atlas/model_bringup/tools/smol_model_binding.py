"""Explicit source binding for the unchanged original M1/M2 attention model."""
import subprocess
import sys
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked
from atlas_tooling import sha256
from run_model_tile import model_sources

MODEL_PIN = '47b8599d8ffbe5ddd96059ffe2b5be2427a410f5'


def model_binding(model_root):
    model_root = checked(model_root)
    revision = subprocess.check_output(['git','-C',str(model_root),'rev-parse','HEAD'],text=True).strip()
    if revision != MODEL_PIN:
        raise ValueError('Smol M2 requires the original admitted model revision '+MODEL_PIN)
    dirty = subprocess.run(['git','-C',str(model_root),'diff','--quiet','HEAD','--','npu_model'],check=False).returncode
    if dirty:
        raise ValueError('Smol M2 original model has tracked source changes or failed diff')
    return dict(revision=revision,tracked_source_diff_exit=dirty,imported_source_sha256=model_sources(model_root),
        curve_sha256=sha256(model_root/'npu_model/configs/memory_curves/ee290sim_vcs_probe.json'))


def configure_model(model_root):
    model_root = checked(model_root)
    model_binding(model_root)
    sys.path.insert(0,str(model_root))
    import npu_model
    import torch
    from npu_model.configs import isa_definition
    from npu_model.util.converter import input_to_program
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)
    model_sources(model_root)
    return npu_model,torch,input_to_program,MODEL_PIN
