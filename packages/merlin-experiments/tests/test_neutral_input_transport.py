"""Only EL1 changes the bytes passed across the immutable compiler boundary."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin_experiments.phase1.feedback import snapshot_execution as S
from merlin_experiments.phase1 import neutral_program_inputs as N


@pytest.mark.parametrize("format", [None, "neutral_math_v1", "shared_interface_v1"])
def test_candidate_input_bytes_follow_the_sealed_disclosure(tmp_path, monkeypatch, format):
    from merlin.targetgen import package_runtime as P

    source = tmp_path/"original.mlir"
    source.write_text('''module attributes {merlin_iface.version = "0.1", merlin_iface.target = "test", merlin_iface.abi_version = "0.1"} {
  %A = merlin_iface.tensor {name = "A", role = "input"} : tensor<1x1xi8>
  %Y = merlin_iface.movement %A {name = "Y", semantic = "mvin_mvout", output_dtype = "i32"} : (tensor<1x1xi8>) -> tensor<1x1xi32>
}
''')
    original = source.read_bytes()
    observed = []
    def invoke(argv, scratch, **kwargs):
        incoming = Path(kwargs["inputs"][1])
        if format == "neutral_math_v1":
            assert all(b"merlin_iface" not in path.read_bytes() for path in scratch.rglob("*") if path.is_file())
        observed.append((incoming.name, incoming.read_bytes(), argv))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    bundle = {} if format is None else dict(author_profile=dict(disclosure=dict(input_format=format)))
    owner = SimpleNamespace(owner=SimpleNamespace(bundle=bundle), work_root=tmp_path,
        template=Path("/candidate-io"), candidate=tmp_path/"candidate",
        _package=lambda pkg: None, _invoke=invoke)
    monkeypatch.setattr(P, "_admit_artifact_protocol", lambda *args: None)
    monkeypatch.setattr(P, "_resolve_argv", lambda pkg, name, incoming, outgoing, **kwargs: ["compiler", str(incoming)])
    monkeypatch.setattr(P, "_needs_interpreter", lambda *args: False)
    result = S.SnapshotExecutor.run_entrypoint(owner, object(), "parse", source)
    assert result.returncode == 0
    assert source.read_bytes() == original
    filename, payload, argv = observed[0]
    if format == "neutral_math_v1":
        assert payload == N.encode_source(original)
        assert b"merlin_iface" not in payload
        assert filename == "input.program.json"
    else:
        assert payload == original
        assert filename == "input.mlir"
    assert argv == ["compiler", "/candidate-io/"+filename]
