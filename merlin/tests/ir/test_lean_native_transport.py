"""Hold-only protocol controls; fake module tests make no MLIR semantic claim."""

import hashlib
import json
import sys

import pytest

from merlin.llvmlower import pipeline
from merlin.llvmlower import proof_stage_transport as transport
from merlin.verify.lean.native_plan import ordinary


@pytest.fixture
def plan(monkeypatch):
    for key in ["MERLIN_GENERALIZE_BEFORE_FUSE", "MERLIN_FUSE_POST", "MERLIN_NO_DEALLOC", "MERLIN_PASSES"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(pipeline, "_sink_deallocs", lambda: False)
    return ordinary(features=frozenset(), index_bits=64, data_layout="e-p:64:64")


_FAKE_MODULE = """
class Kind:
    @staticmethod
    def isinstance(value): return False
class IR:
    IndexType = RankedTensorType = FunctionType = TypeAttr = Kind
class Module:
    def __init__(self):
        self.operation = self
        self.operands = self.results = self.attributes = self.regions = []
    def get_asm(self, **kwargs):
        return 'module { /* changed */ }' if (owner / 'mutate-module').exists() else 'module {}'
    def verify(self): return True
class Parser:
    @staticmethod
    def parse(payload, ctx):
        assert payload == 'module {}'
        return Module()
IR.Module = Parser
ir = IR()
def _pg_identity():
    return {'changed': True} if (owner / 'mutate-consumer').exists() else {}
def _run_stages(ctx, module, pipeline, erase, *rest):
    assert id(module) == original_module_id
    assert id(ctx) == original_context_id
    with (owner / 'executed.jsonl').open('a') as log:
        log.write(json.dumps(pipeline) + '\\n')
"""


def _child(tmp_path, plan, *, extra="", timeout=5):
    owner = tmp_path / "owner"
    owner.mkdir()
    source = tmp_path / "source.mlir"
    source.write_text("module {}")
    binding = {"input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "consumer": {}, "index_bits": 64}
    config = {
        "_PG_DIRECTORY": str(owner),
        "_PG_RUN": "fresh-run",
        "_PG_PLAN_SHA": plan.sha256,
        "_PG_PHASES": tuple((b.name, b.pass_entry) for b in plan.prefix),
        "_PG_PIPELINE": plan.pipeline(),
        "_PG_SUFFIX": ",".join(plan.suffix),
        "_PG_BINDING": binding,
        "_PG_MAX_BYTES": 1_000_000,
        "_PG_TIMEOUT": timeout,
        "_PG_RECORDS": [],
    }
    runner = tmp_path / "child.py"
    runner.write_text(
        "import json\nfrom pathlib import Path\n"
        + f"owner = Path({str(owner)!r})\n"
        + _FAKE_MODULE
        + "\n".join(f"{k} = {v!r}" for k, v in config.items())
        + "\n"
        + transport.NATIVE
        + f"\nctx = object()\nmodule = _pg_parse({str(source)!r}, ctx)\n"
        + "original_module_id = id(module)\noriginal_context_id = id(ctx)\n"
        + extra
        + "\n_run_stages(ctx, module, _PG_PIPELINE, 0)\n"
    )
    return owner, binding, [sys.executable, str(runner)]


def _proof(descriptor, owner):
    report, chain = owner / "proof-report.json", owner / "chain.json"
    report.write_text(json.dumps({"verdict": "verified"}))
    chain.write_text(json.dumps({"verdict": "verified"}))
    return transport.HoldAcknowledgment(
        hashlib.sha256((owner / "request.json").read_bytes()).hexdigest(), report, chain
    )


def _run(command, owner, plan, binding, callback=_proof):
    return transport.run_command(
        command,
        directory=owner,
        callback=callback,
        run="fresh-run",
        plan=plan,
        binding=binding,
        max_bytes=1_000_000,
        timeout=5,
    )


def test_hold_preserves_module_context_and_authored_order(tmp_path, plan):
    owner, binding, command = _child(tmp_path, plan)
    result = _run(command, owner, plan, binding)
    assert result.returncode == 0, result.stderr
    executed = [json.loads(x) for x in (owner / "executed.jsonl").read_text().splitlines()]
    assert executed == [b.pass_entry for b in plan.prefix] + [",".join(plan.suffix)]
    assert [p.name for p in (owner / "stages").iterdir() if p.is_dir() and not p.name.startswith(".")] == [
        f"{n:04d}_{b.name}" for n, b in enumerate(plan.boundaries.boundaries)
    ]
    assert transport.TOKEN in result.stdout
    assert "fragment" not in json.loads((owner / "ack.json").read_text())


@pytest.mark.parametrize("fault", ["stale", "abstracted", "callback-error"])
def test_parent_refusal_never_runs_suffix(tmp_path, plan, fault):
    owner, binding, command = _child(tmp_path, plan)

    def refuse(descriptor, root):
        proof = _proof(descriptor, root)
        if fault == "stale":
            proof.request_sha256 = "0" * 64
        elif fault == "abstracted":
            proof.chain.write_text('{"verdict": "abstracted"}')
        else:
            raise RuntimeError("proof unavailable")
        return proof

    with pytest.raises((ValueError, RuntimeError)):
        _run(command, owner, plan, binding, refuse)
    assert len((owner / "executed.jsonl").read_text().splitlines()) == 4
    assert not (owner / "ack.json").exists()


@pytest.mark.parametrize("fault", ["module", "consumer"])
def test_changed_actual_held_state_never_runs_suffix(tmp_path, plan, fault):
    owner, binding, command = _child(tmp_path, plan)

    def mutate(descriptor, root):
        proof = _proof(descriptor, root)
        (root / f"mutate-{fault}").touch()
        return proof

    result = _run(command, owner, plan, binding, mutate)
    assert result.returncode != 0
    assert "held native module or consumer changed" in result.stderr
    assert len((owner / "executed.jsonl").read_text().splitlines()) == 4
    assert transport.TOKEN not in result.stdout


def test_child_success_without_barrier_refuses(tmp_path, plan):
    owner = tmp_path / "owner"
    owner.mkdir()
    with pytest.raises(ValueError, match="without requesting"):
        _run([sys.executable, "-c", "pass"], owner, plan, {"consumer": {}})


def test_private_owner_must_be_fresh(tmp_path, plan):
    owner, binding, command = _child(tmp_path, plan)
    (owner / "ack.json").write_text("stale")
    with pytest.raises(ValueError, match="fresh private owner"):
        _run(command, owner, plan, binding)


def test_actual_ordinary_runner_binding_compiles_without_string_inference(tmp_path, plan):
    source = pipeline._select_runner(plan.pipeline(), frozenset(), emit=pipeline.EMIT_TRANSLATE)
    bound = transport.bind_runner(
        source, directory=tmp_path, run="fresh-run", plan=plan, binding={}, max_bytes=1_000_000, timeout=5
    )
    compile(bound, "<bound-native-runner>", "exec")
    assert "module = _pg_parse(src_path, ctx)" in bound
    assert "_PG_PHASES =" in bound
    assert transport.NATIVE in bound
    assert "pipeline.split" not in transport.NATIVE


@pytest.mark.parametrize("change", ["duplicate-call", "replace-parse", "rename-owner"])
def test_unrecognized_runner_shape_refuses_before_execution(tmp_path, plan, change):
    source = pipeline._select_runner(plan.pipeline(), frozenset(), emit=pipeline.EMIT_TRANSLATE)
    if change == "duplicate-call":
        source += "\n_run_stages(ctx, module, pipeline, 0)\n"
    elif change == "replace-parse":
        source = source.replace("ir.Module.parse(f.read(), ctx)", "ir.Module.parse(replacement, ctx)")
    else:
        source = source.replace("module = ir.Module.parse", "other = ir.Module.parse")
    with pytest.raises(ValueError, match="native proof transport"):
        transport.bind_runner(
            source, directory=tmp_path, run="fresh-run", plan=plan, binding={}, max_bytes=1_000_000, timeout=5
        )


@pytest.mark.parametrize(
    "field", ["run", "plan_sha256", "request_sha256", "held_sha256", "binding_sha256", "verdict", "fragment"]
)
def test_child_rejects_corrupted_acknowledgment_before_suffix(tmp_path, plan, monkeypatch, field):
    owner, binding, command = _child(tmp_path, plan)
    claim = transport.HoldAcknowledgment.claim

    def corrupt(self, request):
        result = claim(self, request)
        result[field] = "wrong"
        return result

    monkeypatch.setattr(transport.HoldAcknowledgment, "claim", corrupt)
    result = _run(command, owner, plan, binding)
    assert result.returncode != 0
    assert "stale or mismatched native proof acknowledgment" in result.stderr
    assert len((owner / "executed.jsonl").read_text().splitlines()) == 4
    assert transport.TOKEN not in result.stdout
