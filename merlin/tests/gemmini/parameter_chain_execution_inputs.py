"""Original-owner intake for the explicit public full-model regression.

The JSON supplies files/tools, never callbacks, approvals, captures to fabricate,
precision recipes or hardware permissions. The sealed numerical owner is reopened;
the ordinary ABI worker does not claim another numerical replay or hermetic runtime.
"""

import ast
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import packing_execution_inputs as P
import parameter_chain_reference as R
import yaml

from merlin.common.paths import repo_root
from merlin.common.strict_json import loads

SCHEMA = "merlin.test.public_parameter_chain_baseline.v1"
REQUIRED_TIERS = ("L0", "L1", "L2", "L3")


def _file(row):
    assert type(row) is dict and set(row) == {"path", "sha256"}
    path = P.canonical(row["path"])
    assert path.stat().st_size <= 64 << 20, "selected test input exceeds its public read bound"
    assert P.pin(path) == row, "selected original input changed"
    return path


def read_selection(path):
    path = P.canonical(str(path))
    assert path.stat().st_size <= 64 << 10
    selected = loads(path.read_bytes(), max_bytes=64 << 10)
    assert type(selected) is dict and set(selected) == {
        "schema",
        "execution",
        "attestation",
        "source_member",
        "public_case",
        "worker",
        "obligations_profile",
    }
    assert selected["schema"] == SCHEMA
    execution, execution_pin = P.read_selection(_file(selected["execution"]))
    _file(selected["attestation"])
    member = selected["source_member"]
    assert type(member) is str and Path(member).name == member and member.endswith(".py")
    public = selected["public_case"]
    assert type(public) is dict and set(public) == set(R.PUBLIC_FILES)
    paths = [_file(public[name]) for name in R.PUBLIC_FILES]
    assert len({p.parent for p in paths}) == 1 and all(p.name == name for p, name in zip(paths, R.PUBLIC_FILES))
    worker = selected["worker"]
    assert type(worker) is dict and set(worker) == {"python", "binary", "config", "torch_version", "torch_init"}
    python = Path(worker["python"])
    assert python.is_absolute() and python.is_file() and python.resolve() == _file(worker["binary"])
    config = _file(worker["config"])
    assert config == python.parent.parent / "pyvenv.cfg"
    fields = dict(
        (k.strip(), v.strip())
        for line in config.read_text().splitlines()
        for k, sep, v in (line.partition("="),)
        if sep
    )
    version = fields["version_info"]
    site = python.parent.parent / f"lib/python{'.'.join(version.split('.')[:2])}/site-packages/torch"
    assert _file(worker["torch_version"]) == site / "version.py"
    assert _file(worker["torch_init"]) == site / "__init__.py"
    torch_version = next(
        n.value.value
        for n in ast.parse((site / "version.py").read_text()).body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets)
    )
    profile = _file(selected["obligations_profile"]).read_text()
    block = profile.split("  required_oracle_tiers:\n", 1)[1].split("\ncapsules:", 1)[0]
    assert yaml.safe_load(block) == list(REQUIRED_TIERS), "all original obligations remain required"
    return SimpleNamespace(
        document=selected,
        pin=P.pin(path),
        execution=execution,
        execution_pin=execution_pin,
        python_version=version,
        torch_version=torch_version,
    )


def recheck(selected):
    assert P.pin(P.canonical(selected.pin["path"])) == selected.pin
    current = read_selection(selected.pin["path"])
    assert vars(current) == vars(selected), "operator selection changed"


def prepare(selected, destination, monkeypatch):
    """Return a LIVE materialized owner only after genuine source/value/ABI checks."""
    from merlin_experiments.capture_execution.m2m_origin import git_origin
    from merlin_experiments.phase0 import capture_execution_attestation as A
    from merlin_experiments.phase0 import capture_selection as selection_owner
    from merlin_experiments.phase0 import sealed_native_program as N

    from merlin.common import quant_formats as QF
    from merlin.compile.materialized_model_inputs import MaterializedModelInputs
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen import capsule_source as S
    from merlin.targetgen import target_registry as TR
    from merlin.targetgen.contract import schemas as CS
    from merlin.targetgen.corpus_spec import CorpusBinding

    recheck(selected)
    destination = Path(destination)
    assert destination.is_absolute() and not destination.exists()
    assert not any(p.is_symlink() for p in (destination, *destination.parents))
    destination.mkdir()
    providers, temporary = destination / "no-providers", destination / "host-tmp"
    providers.mkdir()
    temporary.mkdir()
    document = selected.document
    attestation = loads(_file(document["attestation"]).read_bytes(), max_bytes=1 << 20)
    selection_row = attestation["selection"]
    original_selection = loads(_file({k: selection_row[k] for k in ("path", "sha256")}).read_bytes(), max_bytes=4 << 20)
    loader = Path(original_selection["run_dir"]) / "snapshots/source/workload" / document["source_member"]
    loader_pin = P.pin(P.canonical(str(loader)))
    root = repo_root()
    m2m = Path(original_selection["plan"]["m2m_root"])
    m2m_identity = None
    controls = {
        "TMPDIR": str(temporary),
        "PYTHONDONTWRITEBYTECODE": "1",
        "MERLIN_TARGET_PATH": str(providers),
        "MERLIN_M2M_DIR": str(original_selection["plan"]["m2m_root"]),
        "MERLIN_M2M_PYTHON": document["worker"]["python"],
        "MERLIN_QUANT_FORMATS": "",
        "MERLIN_RTL_FACTS": selected.execution["facts"]["path"],
        "MERLIN_TARGET_CONTRACT": selected.execution["quiet_contract"]["path"],
        "MERLIN_CONTRACT_DIR": str(root / "merlin/contract"),
        "MERLIN_SCHEMAS_DIR": str(root / "merlin/schemas"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    for key, value in controls.items():
        monkeypatch.setenv(key, value)
    quiet_path = Path(selected.execution["quiet_contract"]["path"])
    quiet = yaml.safe_load(quiet_path.read_bytes())
    quiet_bytes = json.dumps(quiet, sort_keys=True)
    attestation_bytes = json.dumps(attestation, sort_keys=True)

    def readers():
        return (
            readers,
            read_selection,
            recheck,
            prepare,
            _file,
            P.read_selection,
            P.recheck_selection,
            P.pin,
            P.canonical,
            P.recheck_artifacts,
            *(f for f in vars(R).values() if inspect.isfunction(f)),
            A.require_verified_execution,
            N.read,
            N._root,
            N.Source.__init__,
            N.Source.available,
            N.Source.capture_loader,
            selection_owner.load,
            S.write_model_capsule,
            TR.observed_contract,
            TR.observed_contract.__wrapped__,
            git_origin,
            S._m2m_python,
            CR.load_capsule,
            CR._model_runtime_bundle,
            CR._content_identity,
            CR._file_identity,
            CS.validate_capsule,
            CS.validate,
            CS.load_schema,
            CS._check_output_policy_roster,
            CS._declared_dtypes,
            QF.has,
            QF.is_element_dtype,
            QF._alias_index.__wrapped__,
            QF.registry.__wrapped__,
            MaterializedModelInputs.bind.__func__,
            MaterializedModelInputs.verify,
            MaterializedModelInputs.record,
        )

    reader, reader_code, functions = readers, readers.__code__, readers()
    codes = tuple(f.__code__ for f in functions)
    constants = (tuple(REQUIRED_TIERS), tuple(R.PUBLIC_FILES), json.dumps(R.POLICY, sort_keys=True))
    caches = QF._alias_index, QF.registry
    pins = [P.pin(Path(inspect.getsourcefile(f)).resolve()) for f in functions]
    pins += [
        loader_pin,
        P.pin(P.canonical(str(m2m / "m2m/api.py"))),
        P.pin(root / "merlin/contract/schemas/capsule.schema.json"),
        P.pin(root / "merlin/schemas/quant_formats.registry.yaml"),
        P.pin(Path(CR.__file__).with_name("_capsule_bundle_worker.py")),
    ]

    def current():
        assert readers is reader and reader.__code__ is reader_code and reader() == functions
        assert all(f.__code__ is c for f, c in zip(functions, codes, strict=True))
        assert QF._alias_index is caches[0] and QF.registry is caches[1]
        P.recheck_artifacts(pins)
        recheck(selected)
        assert controls == {k: os.environ.get(k) for k in controls}
        assert (tuple(REQUIRED_TIERS), tuple(R.PUBLIC_FILES), json.dumps(R.POLICY, sort_keys=True)) == constants
        assert json.dumps(quiet, sort_keys=True) == quiet_bytes
        assert json.dumps(attestation, sort_keys=True) == attestation_bytes
        if m2m_identity is not None:
            assert git_origin(m2m, clean=True) == m2m_identity

    current()
    A.require_verified_execution(attestation)
    m2m_identity = git_origin(m2m, clean=True)
    assert m2m_identity["commit"] == original_selection["plan"]["m2m_commit"]
    current()
    captured = N.read(attestation, program=None, source_member=document["source_member"], max_payload_bytes=64 << 20)
    current()
    source, x, weights, wanted, prefixes = R.public_reference(document["public_case"])
    assert captured.pytorch_src.encode() == _file(document["public_case"]["capsule.pytorch.py"]).read_bytes()
    assert R.semantic_identity(captured.linalg_mlir, weights_path=Path(captured.weights_path)) == R.semantic_identity(
        source, weights_path=Path(document["public_case"]["capsule.weights.safetensors"]["path"])
    )
    assert R.geometry(captured.linalg_mlir) == x.shape[1]
    actual_weights = R.signed_weights(Path(captured.weights_path), x.shape[1])
    assert all(actual_weights[k].tobytes() == weights[k].tobytes() for k in weights)
    manifest = json.loads(_file(document["public_case"]["capsule.weights.safetensors.manifest.json"]).read_bytes())
    assert manifest == {
        **{str(i): {"kind": "param", "weight": f"w{i}"} for i in range(3)},
        "3": {"kind": "input", "name": "x"},
    }
    assert json.loads(Path(captured.meta["weights_manifest"]).read_bytes()) == manifest
    R.captured_values(captured, x, wanted)
    current()
    source_owner = N.Source(attestation, program=None, source_member=document["source_member"])
    target = quiet["name"]
    binding = CorpusBinding(
        target=target,
        tile_dim=None,
        operand_dtype="int8",
        accum_dtype="i32",
        integer=True,
        tiers=list(REQUIRED_TIERS),
        compare="exact_int",
        atol=0,
        rtol=0,
        readout_inputs={},
    )
    entry = dict(
        name="public_parameter_chain_3mib_development",
        cat="model",
        kind="model",
        op="model",
        loader=str(loader),
        capture_quantization="already_materialized",
        output_numeric_policy="captured_per_output_v1",
        outs=["wide", "state"],
        source_reference="operator-selected complete public development fixture",
    )
    with TR.observed_contract(target, quiet, source_path=quiet_path):
        capsule_dir = S.write_model_capsule(entry, binding, destination / "capsules", source=source_owner)
    current()
    assert source_owner.attestations == [attestation]
    capsule = CR.load_capsule(capsule_dir, contract=root / "merlin/contract")
    current()
    assert capsule["numeric_policy"] == R.POLICY and capsule["required_oracle_tiers"] == list(REQUIRED_TIERS)
    assert capsule["source_role"] == "pytorch_model_slice"
    with CR._model_runtime_bundle(capsule, timeout=300, artifact_root=destination / "runtime") as (
        bundle,
        provenance,
        verify,
    ):
        current()
        verify()
        assert provenance["interface_reused_byte_exact"] is True and provenance["live_recapture_used"] is False
        assert provenance["validation"]["capture_manifest_validated"] is True
        assert provenance["validation"]["golden_validated"] is True
        assert provenance["validation"]["golden_value_replay"] is False
        assert provenance["tool"]["python"] == document["worker"]["binary"]["path"]
        assert provenance["tool"]["python_version"] == selected.python_version
        assert provenance["tool"]["torch_version"] == selected.torch_version
        assert R.semantic_identity(
            (bundle / "model.mlir").read_text(), weights_path=bundle / "capsule.weights.safetensors"
        ) == R.semantic_identity(
            source, weights_path=Path(document["public_case"]["capsule.weights.safetensors"]["path"])
        )
        assert json.loads((bundle / "output_order.json").read_bytes()) == ["wide", "state"]
        with np.load(bundle / "goldens.npz", allow_pickle=False) as arrays:
            assert arrays.files == ["wide", "state"]
            R.exact_outputs(tuple(arrays[k] for k in arrays.files), wanted)
        artifacts = {
            p.name: {"sha256": P.pin(p)["sha256"], "size_bytes": p.stat().st_size}
            for p in bundle.iterdir()
            if p.is_file()
        }
        bound = MaterializedModelInputs.bind(
            bundle,
            artifacts=artifacts,
            validator_identity={
                **provenance["tool"],
                "validation": provenance["validation"],
                "scope": "original sealed capture and independent complete reference plus runtime ABI only",
            },
            source_identity={**provenance["source"], "original_sealed_attestation": document["attestation"]},
        )
        materialized = bound.record()
        A.require_verified_execution(attestation)
        current()
        verify()
        bound.verify(bundle)
        record = dict(
            selected_m2m_source=m2m_identity,
            scope="original verified capture consumed as NEW development source/ABI intake",
            attestation=document["attestation"],
            worker=provenance,
            materialized=materialized,
            prefixes=prefixes,
            original_pins=pins,
            obligations={t: "PENDING_NOT_GRADED" for t in REQUIRED_TIERS},
            worker_observation_limit="existing helper retains request/provenance, not raw I/stdout/stderr",
            no_capture_issuance_or_replay=True,
            no_runtime_CAS_copy=True,
        )
        (destination / "intake.json").write_text(json.dumps(record, indent=2) + "\n")
        return SimpleNamespace(
            bundle=bundle,
            bound=bound,
            expected=wanted,
            x=x,
            weights=weights,
            prefixes=prefixes,
            verify=verify,
            current=current,
            attestation=attestation,
            record=record,
        )
