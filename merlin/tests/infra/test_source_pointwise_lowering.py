"""Fixed-capability pointwise discovery and shared model transaction controls."""

import json
from dataclasses import replace

import pytest
from test_pointwise_source import case, source_program, source_text
from xdsl.dialects import func

from merlin.common import mlir_query as mq
from merlin.llvmlower.prepared_model_transform import apply_prepared_model_transform
from merlin.llvmlower.source_model import bind_source_lowerings, source_provider_builder
from merlin.llvmlower.source_pointwise import PointwiseProgram, SourcePointwiseLowering
from merlin.transforms.certificates import ir_text


def fixture(tmp_path, *, mapper=None, prepare=None, **controls):
    caps_dir = tmp_path / "capabilities"
    caps_dir.mkdir()
    caps = case(caps_dir, m=1, n=1)
    mapped, prepared = [], []

    def lower(source):
        mapped.append(source)
        if mapper is not None:
            return mapper(source, caps)
        path = tmp_path / ("program_" + str(len(mapped)))
        path.mkdir()
        opts = case(path, m=source.shape[0], n=source.shape[1])
        assert opts["selected_contracts"].facts_bytes == caps["selected_contracts"].facts_bytes
        assert opts["selected_contracts"].software_bytes == caps["selected_contracts"].software_bytes
        return PointwiseProgram(opts["kernel"], opts["objects"], opts["bindings"])

    def implement(bindings, work):
        prepared.extend(bindings)
        if prepare is not None:
            prepare(bindings, work)

    def build(bindings, context):
        pytest.fail("discovery-only test must not build a provider")

    selected = SourcePointwiseLowering(
        caps["selected_contracts"],
        caps["adapter"],
        caps["encoding"],
        caps["limits"],
        "host",
        32,
        lower,
        implement,
        build,
        private_ownership=True,
        external_buffers_disjoint=True,
        scalar_qualification_identity=caps["fusion_qualification_identity"],
        fusion_emitters=tuple(caps["fusion_emitters"].items()),
    )
    return replace(selected, **controls), mapped, prepared


def run(tmp_path, selected, module):
    path = tmp_path / "source.mlir"
    path.write_text(ir_text(module))
    before = path.read_bytes()
    callback, builder = bind_source_lowerings((selected,), None, None, None)
    result = apply_prepared_model_transform(path, tmp_path / "preparation", callback)
    assert path.read_bytes() == before
    mq.parse(result).verify()
    return result, builder


@pytest.mark.parametrize("m,n", [(3, 5), (2, 11), (1, 64)])
def test_selected_software_semantics_drive_discovery_without_target_source_matching(tmp_path, m, n):
    selected, mapped, prepared = fixture(tmp_path)
    module, _ = source_program(m=m, n=n)
    result, builder = run(tmp_path, selected, module)
    assert len(mapped) == len(prepared) == 1
    assert mapped[0].shape == (m, n)
    assert set(mapped[0].__dataclass_fields__) == {"shape", "scalar_ir", "graph"}
    prepared[0].compilation.revalidate()
    assert prepared[0].compilation.conformance["actual_commands"] == 1
    assert prepared[0].writer.result_argument == 2
    assert source_provider_builder(builder) == builder
    body = next(op for op in mq.parse(result).body.block.ops if op.sym_name.data == "forward").body.block
    call = next(op for op in body.ops if type(op) is func.CallOp)
    assert tuple(call.arguments[:2]) == tuple(body.args[:2])
    assert call.arguments[-1].owner.name == "tensor.empty"
    record = json.loads((result.parent / "source_pointwise.json").read_text())
    assert record["bindings"][0]["source_conformance"]["actual_commands"] == 1


@pytest.mark.parametrize("unavailable", ["arithmetic", "qualified", "ownership", "alias", "mapping"])
def test_unsupported_cases_preserve_exact_source_without_provider(tmp_path, unavailable):
    controls = {
        "qualified": dict(scalar_qualification_identity=None),
        "ownership": dict(private_ownership=False),
        "alias": dict(external_buffers_disjoint=False),
        "mapping": dict(mapper=lambda source, caps: None),
    }.get(unavailable, {})
    selected, mapped, prepared = fixture(tmp_path, **controls)
    source = source_text().replace("%sum=arith.addi", "%sum=arith.subi") if unavailable == "arithmetic" else None
    module, _ = source_program(source)
    result, builder = run(tmp_path, selected, module)
    assert result.read_bytes() == (tmp_path / "source.mlir").read_bytes()
    assert not prepared and source_provider_builder(builder) is None
    assert len(mapped) == (1 if unavailable == "mapping" else 0)


def test_wrong_original_program_is_rejected_before_writer_or_implementation(tmp_path):
    def lower(source, caps):
        kernel = caps["kernel"]
        return PointwiseProgram(replace(kernel, body=tuple(reversed(kernel.body))), caps["objects"], caps["bindings"])

    selected, _, prepared = fixture(tmp_path, mapper=lower)
    module, _ = source_program(m=1, n=1)
    with pytest.raises(ValueError, match="undefined|overwritten|differs"):
        run(tmp_path, selected, module)
    assert not prepared
    assert not (tmp_path / "preparation" / "source_pointwise.mlir").exists()


@pytest.mark.parametrize("field", ["limits", "budgets", "domain", "window", "branch", "decoder", "decode", "prefetch"])
def test_mapping_cannot_inflate_fixed_policy_or_change_execution_domain(tmp_path, field):
    def lower(source, caps):
        if field == "limits":
            object.__setattr__(selected.limits, "max_instructions", selected.limits.max_instructions + 100)
        elif field == "budgets":
            object.__setattr__(selected.budgets, "max_fusions", selected.budgets.max_fusions + 100)
        elif field == "domain":
            object.__setattr__(selected, "execution_domain", "device")
        elif field == "window":
            window = selected.limits.windows[0]
            object.__setattr__(window, "size", window.size + 1024)
        elif field == "branch":
            object.__setattr__(selected.limits.branches[0], "maximum", 65536)
        elif field == "decoder":
            object.__setattr__(selected.encoding.program_decoder, "qualification_identity", "different")
        elif field == "prefetch":
            object.__setattr__(selected, "prefetch_qualification_identity", "candidate-asserted-overlap")
        else:
            object.__setattr__(selected.encoding.program_decoder, "decode", lambda image: ())
        return PointwiseProgram(caps["kernel"], caps["objects"], caps["bindings"])

    selected, _, prepared = fixture(tmp_path, mapper=lower)
    if field == "branch":
        from merlin.compile.scheduling.program_limits import BranchRule

        selected = replace(
            selected,
            limits=replace(
                selected.limits,
                branches=(BranchRule("relative", -8, 8, 1, "next_instruction"),),
            ),
        )
    module, _ = source_program(m=1, n=1)
    with pytest.raises(ValueError, match="selection changed"):
        run(tmp_path, selected, module)
    assert not prepared


@pytest.mark.parametrize("identity", [True, 1, "", " "])
def test_prefetch_qualification_requires_an_original_nonempty_identity(tmp_path, identity):
    with pytest.raises(ValueError, match="fixed typed capabilities"):
        fixture(tmp_path, prefetch_qualification_identity=identity)


def test_duplicate_source_family_selection_is_ambiguous(tmp_path):
    selected, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="ambiguous"):
        bind_source_lowerings((selected, selected), None, None, None)
