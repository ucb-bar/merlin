import os
import subprocess

import pytest

from merlin.common.paths import repo_root
from merlin.integrations.model2mlir import capture_pin, capture_python_path, root


def test_public_module_input_freezes_and_executes_the_original(tmp_path):
    """Real exporter/producer/compiler calls, including shared state and refusals."""
    checkout = root()
    python = capture_python_path(checkout=checkout)
    if not python.is_file() or not checkout.is_dir():
        pytest.skip("configure the actual pinned model2MLIR capture environment")
    assert capture_pin(checkout)["ok"]
    script = tmp_path / "programs.py"
    script.write_text(
        "import ctypes, json, subprocess, sys\nfrom pathlib import Path\nimport torch\nfrom merlin import api\nfrom merlin.compiler.backend import layout\nfrom merlin.compiler.frontend.torch.program import freeze\nroot = Path(sys.argv[1])\n\nclass Layout(torch.nn.Module):\n\n    def forward(self, x):\n        return x.reshape(6)\n\ndef function(x):\n    return x.reshape(6)\nmodel = Layout().train()\ninputs = (torch.arange(6, dtype=torch.float32).reshape(2, 3),)\nexported = torch.export.export(model, inputs, strict=False)\ntorch.export.save(exported, root / 'source.pt2')\nname = 'module'\nprogram = model\nresult = api.compile(program, backend=layout, output=root / name, example_inputs=None if name in {'exported', 'archive'} else inputs)\nif name != 'archive':\n    record = result.metadata['frontend']['original_program']\n    saved = root / (name + '.capture') / 'original/program.pt2'\n    assert record['archive_sha256'] == api.Artifact.from_file(saved, kind='export', role='original').sha256\nassert model.training and torch.equal(inputs[0], torch.arange(6).reshape(2, 3))\nassert not any((p.name in {'golden.npy', 'inputs.npz'} for p in result.directory.rglob('*')))\nabi = result.metadata['c_call']\nlibrary = root / 'layout.so'\nsubprocess.run(['gcc', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC', *(str(result.directory / p) for p in abi['link_inputs']), '-o', str(library)], check=True)\ncall = getattr(ctypes.CDLL(str(library)), abi['symbol'])\ncall.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]\ncall.restype = ctypes.c_int\nactual = torch.empty(6, dtype=torch.float32)\nassert call(None, (ctypes.c_void_p * 1)(inputs[0].data_ptr()), (ctypes.c_void_p * 1)(actual.data_ptr())) == 0\nreference = torch.export.load(saved).module()(*inputs)\nassert torch.equal(actual.view(torch.int32), reference.view(torch.int32))\n\nclass Shared(torch.nn.Module):\n\n    def __init__(self):\n        super().__init__()\n        self.layer = torch.nn.Linear(3, 3, bias=False)\n\n    def forward(self, x):\n        return self.layer(x) + self.layer(x)\nshared = Shared().train()\nbefore = shared.layer.weight.detach().clone()\nfrozen, record = freeze(shared, example_inputs=inputs, workspace=root / 'shared')\nloaded = torch.export.load(frozen.path)\nassert len(loaded.graph_signature.parameters) == 1\nlinear_calls = [n for n in loaded.graph.nodes if n.op == 'call_function' and n.target == torch.ops.aten.linear.default]\nassert len(linear_calls) == 2\nassert torch.equal(before, shared.layer.weight) and shared.training\nassert not (root / 'rejected').exists() and (not (root / 'rejected.capture').exists())\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo_root() / "src"), environment.get("PYTHONPATH")]))
    subprocess.run(
        [str(python), str(script), str(tmp_path)], env=environment, capture_output=True, text=True, check=True
    )


def test_public_fx_input_freezes_and_executes_the_original(tmp_path):
    """Real exporter/producer/compiler calls, including shared state and refusals."""
    checkout = root()
    python = capture_python_path(checkout=checkout)
    if not python.is_file() or not checkout.is_dir():
        pytest.skip("configure the actual pinned model2MLIR capture environment")
    assert capture_pin(checkout)["ok"]
    script = tmp_path / "programs.py"
    script.write_text(
        "import ctypes, json, subprocess, sys\nfrom pathlib import Path\nimport torch\nfrom merlin import api\nfrom merlin.compiler.backend import layout\nfrom merlin.compiler.frontend.torch.program import freeze\nroot = Path(sys.argv[1])\n\nclass Layout(torch.nn.Module):\n\n    def forward(self, x):\n        return x.reshape(6)\n\ndef function(x):\n    return x.reshape(6)\nmodel = Layout().train()\ninputs = (torch.arange(6, dtype=torch.float32).reshape(2, 3),)\nexported = torch.export.export(model, inputs, strict=False)\ntorch.export.save(exported, root / 'source.pt2')\nname = 'fx'\nprogram = torch.fx.symbolic_trace(model)\nresult = api.compile(program, backend=layout, output=root / name, example_inputs=None if name in {'exported', 'archive'} else inputs)\nif name != 'archive':\n    record = result.metadata['frontend']['original_program']\n    saved = root / (name + '.capture') / 'original/program.pt2'\n    assert record['archive_sha256'] == api.Artifact.from_file(saved, kind='export', role='original').sha256\nassert model.training and torch.equal(inputs[0], torch.arange(6).reshape(2, 3))\nassert not any((p.name in {'golden.npy', 'inputs.npz'} for p in result.directory.rglob('*')))\nabi = result.metadata['c_call']\nlibrary = root / 'layout.so'\nsubprocess.run(['gcc', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC', *(str(result.directory / p) for p in abi['link_inputs']), '-o', str(library)], check=True)\ncall = getattr(ctypes.CDLL(str(library)), abi['symbol'])\ncall.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]\ncall.restype = ctypes.c_int\nactual = torch.empty(6, dtype=torch.float32)\nassert call(None, (ctypes.c_void_p * 1)(inputs[0].data_ptr()), (ctypes.c_void_p * 1)(actual.data_ptr())) == 0\nreference = torch.export.load(saved).module()(*inputs)\nassert torch.equal(actual.view(torch.int32), reference.view(torch.int32))\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo_root() / "src"), environment.get("PYTHONPATH")]))
    subprocess.run(
        [str(python), str(script), str(tmp_path)], env=environment, capture_output=True, text=True, check=True
    )


def test_public_function_input_freezes_and_executes_the_original(tmp_path):
    """Real exporter/producer/compiler calls, including shared state and refusals."""
    checkout = root()
    python = capture_python_path(checkout=checkout)
    if not python.is_file() or not checkout.is_dir():
        pytest.skip("configure the actual pinned model2MLIR capture environment")
    assert capture_pin(checkout)["ok"]
    script = tmp_path / "programs.py"
    script.write_text(
        "import ctypes, json, subprocess, sys\nfrom pathlib import Path\nimport torch\nfrom merlin import api\nfrom merlin.compiler.backend import layout\nfrom merlin.compiler.frontend.torch.program import freeze\nroot = Path(sys.argv[1])\n\nclass Layout(torch.nn.Module):\n\n    def forward(self, x):\n        return x.reshape(6)\n\ndef function(x):\n    return x.reshape(6)\nmodel = Layout().train()\ninputs = (torch.arange(6, dtype=torch.float32).reshape(2, 3),)\nexported = torch.export.export(model, inputs, strict=False)\ntorch.export.save(exported, root / 'source.pt2')\nname = 'function'\nprogram = function\nresult = api.compile(program, backend=layout, output=root / name, example_inputs=None if name in {'exported', 'archive'} else inputs)\nif name != 'archive':\n    record = result.metadata['frontend']['original_program']\n    saved = root / (name + '.capture') / 'original/program.pt2'\n    assert record['archive_sha256'] == api.Artifact.from_file(saved, kind='export', role='original').sha256\nassert model.training and torch.equal(inputs[0], torch.arange(6).reshape(2, 3))\nassert not any((p.name in {'golden.npy', 'inputs.npz'} for p in result.directory.rglob('*')))\nabi = result.metadata['c_call']\nlibrary = root / 'layout.so'\nsubprocess.run(['gcc', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC', *(str(result.directory / p) for p in abi['link_inputs']), '-o', str(library)], check=True)\ncall = getattr(ctypes.CDLL(str(library)), abi['symbol'])\ncall.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]\ncall.restype = ctypes.c_int\nactual = torch.empty(6, dtype=torch.float32)\nassert call(None, (ctypes.c_void_p * 1)(inputs[0].data_ptr()), (ctypes.c_void_p * 1)(actual.data_ptr())) == 0\nreference = torch.export.load(saved).module()(*inputs)\nassert torch.equal(actual.view(torch.int32), reference.view(torch.int32))\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo_root() / "src"), environment.get("PYTHONPATH")]))
    subprocess.run(
        [str(python), str(script), str(tmp_path)], env=environment, capture_output=True, text=True, check=True
    )
