# ruff: noqa: E501 -- the embedded MLIR lines are single-line linalg.generic ops
"""The OS value bound, the full-tensor accumulator seed and the runtime readout scale of the fused-stage observation.

Three Phase 0 observations, each behind its own reviewed declaration:

* an output-stationary readout, observed only under ``mesh_mac.os_value_bound`` (a signed interval proved
  from literal weights);
* an identity-indexed ``[M, N]`` i32 bias, observed as the contraction's accumulator seed rather than as the
  per-column ``bias_add`` stage, only under the reviewed ``numerical_semantics.accumulator_seed_tensor``;
* a ``tensor<1xf32>`` function-argument readout scale, observed under the reviewed finite non-negative
  ``runtime_input`` domain and admitted only by a declaration that lists ``scale_sources: runtime_input``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0 import numeric_domains, program_admission
from merlin_experiments.phase0.source_fused_stages import (
    _top_level,
    accumulator_seed_tensor,
    observe,
    os_value_bound,
    prove_chains,
    runtime_scale_domain,
    stage_rows,
)
from test_phase0_program_admission import _CONTRACT, _HOST, _SPEC
from test_source_fused_stages import NUMERICS, build
from test_source_fused_window_kext import extension_program

from merlin.targetgen import quantization_numerics, software_spec
from merlin.targetgen.admission_conditions import enforce_provider_emission_obligations
from merlin.targetgen.software_spec import admit_operation, load_software_spec, validate_software_spec

REPO = Path(__file__).resolve().parents[3]
GEMMINI_SPEC = REPO / "examples" / "gemmini" / "target" / "software-spec.yaml"
F32 = "f32_product_rne"

# --------------------------------------------------------------------------------------------------
# Output-stationary value bound
# --------------------------------------------------------------------------------------------------

MESH = {
    "dataflows": ["WS", "OS"],
    "signed_bits": 20,
    "os_value_bound": {
        "status": "reviewed",
        "mode": "signed_interval",
        "proof": "constant_weight_all_prefix_intervals",
    },
}


def mesh(**override):
    """``MESH`` with top-level keys replaced; a ``os_value_bound`` mapping is merged key by key."""
    result = copy.deepcopy(MESH)
    for key, value in override.items():
        if key == "os_value_bound" and isinstance(value, dict):
            result[key].update(value)
        else:
            result[key] = value
    return result


def os_program(*, m=2, k=4096, n=2, weight="ones", bias=None):
    """contraction -> [bias] -> readout over an ``[m, k] x [k, n]`` i8 matmul.

    ``weight``: ``ones`` / ``minus_ones`` (literal splat weights) or ``runtime`` (a function argument).
    ``bias``: ``None``, ``column`` (a constant ``[n]`` vector) or ``matrix`` (a runtime ``[m, n]`` argument).
    """
    arguments = [f"%A: tensor<{m}x{k}xi8>"]
    if weight == "runtime":
        arguments.append(f"%W0: tensor<{k}x{n}xi8>")
    if bias == "matrix":
        arguments.append(f"%Bias: tensor<{m}x{n}xi32>")
    text = "builtin.module {\n  func.func @forward(" + ", ".join(arguments) + f") -> (tensor<{m}x{n}xi8>) {{\n"
    if weight != "runtime":
        splat = {"ones": 1, "minus_ones": -1}[weight]
        text += f"    %W0 = arith.constant dense<{splat}> : tensor<{k}x{n}xi8>\n"
    text += f"""\
    %mac0_empty = tensor.empty() : tensor<{m}x{n}xi32>
    %mac0_zero = arith.constant 0 : i32
    %mac0_init = linalg.fill ins(%mac0_zero : i32) outs(%mac0_empty : tensor<{m}x{n}xi32>) -> tensor<{m}x{n}xi32>
    %mac0 = linalg.generic {{indexing_maps = [affine_map<(d0,d1,d2)->(d0,d2)>, affine_map<(d0,d1,d2)->(d2,d1)>, affine_map<(d0,d1,d2)->(d0,d1)>], iterator_types = ["parallel", "parallel", "reduction"]}} ins(%A, %W0 : tensor<{m}x{k}xi8>, tensor<{k}x{n}xi8>) outs(%mac0_init : tensor<{m}x{n}xi32>) attrs = {{prov.composition_stage = "mac0"}} {{
    ^bb0(%x0: i8, %x1: i8, %dest: i32):
      %a = arith.extsi %x0 : i8 to i32
      %b = arith.extsi %x1 : i8 to i32
      %p = arith.muli %a, %b : i32
      %v = arith.addi %p, %dest : i32
      linalg.yield %v : i32
    }} -> tensor<{m}x{n}xi32>
"""
    source = "%mac0"
    if bias is not None:
        bias_map, bias_type = (
            ("affine_map<(d0, d1) -> (d1)>", f"tensor<{n}xi32>")
            if bias == "column"
            else ("affine_map<(d0, d1) -> (d0, d1)>", f"tensor<{m}x{n}xi32>")
        )
        if bias == "column":
            text += f"    %Bias = arith.constant dense<[{', '.join(str(i) for i in range(n))}]> : {bias_type}\n"
        text += f"""\
    %bias_empty = tensor.empty() : tensor<{m}x{n}xi32>
    %bias = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, {bias_map}, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}} ins(%mac0, %Bias : tensor<{m}x{n}xi32>, {bias_type}) outs(%bias_empty : tensor<{m}x{n}xi32>) attrs = {{prov.composition_stage = "bias"}} {{
    ^bb0(%x0: i32, %x1: i32, %dest: i32):
      %v = arith.addi %x0, %x1 : i32
      linalg.yield %v : i32
    }} -> tensor<{m}x{n}xi32>
"""
        source = "%bias"
    text += f"""\
    %readout_gains = arith.constant dense<[{", ".join(["0.25"] * n)}]> : tensor<{n}xf32>
    %readout_empty = tensor.empty() : tensor<{m}x{n}xi8>
    %readout = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]}} ins({source}, %readout_gains : tensor<{m}x{n}xi32>, tensor<{n}xf32>) outs(%readout_empty : tensor<{m}x{n}xi8>) attrs = {{prov.composition_stage = "readout"}} {{
    ^bb0(%x0: i32, %x1: f32, %dest: i8):
      %f = arith.sitofp %x0 : i32 to f32
      %p = arith.mulf %f, %x1 : f32
      %r = math.roundeven %p : f32
      %lo = arith.constant -128.0 : f32
      %hi = arith.constant 127.0 : f32
      %l = arith.maximumf %r, %lo : f32
      %h = arith.minimumf %l, %hi : f32
      %v = arith.fptosi %h : f32 to i8
      linalg.yield %v : i8
    }} -> tensor<{m}x{n}xi8>
    func.return %readout : tensor<{m}x{n}xi8>
  }}
}}
"""
    return text


def os_chains(tmp_path, *, dataflow="OS", mesh_mac=MESH, strategy=F32, full_tensor_seed=False, **program):
    path = tmp_path / "os.interface.mlir"
    path.write_text(os_program(**program))
    return prove_chains(
        path,
        numerics=NUMERICS,
        strategy=strategy,
        dataflow=dataflow,
        mesh_mac=mesh_mac,
        full_tensor_seed=full_tensor_seed,
    )


def stage_members(chains):
    return [(stage["member"], stage["stages"]) for chain in chains for stage in chain["stages"]]


def test_os_dataflow_without_a_reviewed_value_bound_is_not_observed(tmp_path):
    assert os_chains(tmp_path, mesh_mac=None) == []
    assert os_chains(tmp_path, mesh_mac={"dataflows": ["WS", "OS"], "signed_bits": 20}) == []


def test_os_chain_is_observed_under_the_reviewed_signed_interval(tmp_path):
    (chain,) = os_chains(tmp_path)
    assert chain["producer"] == 4
    # Only the proof's method and width are carried, never the per-column extremes.
    assert chain["carrier"]["os_value_bound"] == {
        "method": "constant_weight_all_prefix_intervals",
        "mesh_signed_bits": 20,
    }
    assert stage_members([chain]) == [(7, ["acc_scale"])]


def test_os_bound_is_exact_at_the_negative_end_of_the_interval(tmp_path):
    # K = 4096 ones: the lowest partial sum is -128 * 4096 = -2**19, which a signed 20-bit PE represents.
    path = tmp_path / "os.interface.mlir"
    path.write_text(os_program(k=4096))
    top, _, _ = _top_level(path)
    producer = top[4]
    bound = os_value_bound(producer, MESH)
    assert bound["column_lower"] == [-(1 << 19)] * 2
    assert bound["column_upper"] == [127 * 4096] * 2
    assert bound["mesh_signed_bits"] == 20 and bound["method"] == "constant_weight_all_prefix_intervals"
    # The bound is computed on a stripped clone: the written op keeps its provenance attributes.
    assert "prov.composition_stage" in producer.attributes


@pytest.mark.parametrize(
    "case",
    [
        dict(k=4097),  # -128 * 4097 < -2**19: the partial sum can wrap
        dict(weight="minus_ones"),  # K = 4096: the highest partial sum is +128 * 4096 = 2**19, one too many
    ],
    ids=["k_4097", "minus_ones_hit_plus_2_19"],
)
def test_os_bound_outside_the_signed_interval_is_not_observed(tmp_path, case):
    assert os_chains(tmp_path, **case) == []
    # The same program is an ordinary (bound-free) chain under WS: only the mesh bound refuses it.
    assert os_chains(tmp_path, dataflow="WS", **case) != []


def test_os_runtime_weight_is_not_observed(tmp_path):
    assert os_chains(tmp_path, weight="runtime") == []
    assert os_chains(tmp_path, weight="runtime", dataflow="WS") != []


UNREVIEWED = {
    "status_unreviewed": mesh(os_value_bound={"status": "unreviewed"}),
    "status_missing": mesh(os_value_bound={"status": None}),
    "mode_absolute": mesh(os_value_bound={"mode": "absolute"}),
    "proof_other": mesh(os_value_bound={"proof": "runtime_weight_bound"}),
    "no_os_dataflow": mesh(dataflows=["WS"]),
    "no_dataflows": mesh(dataflows=None),
    "no_declaration": {key: value for key, value in MESH.items() if key != "os_value_bound"},
    "no_mesh_width": mesh(signed_bits=None),
    "mesh_width_too_narrow": mesh(signed_bits=19),
}


@pytest.mark.parametrize("declaration", UNREVIEWED.values(), ids=UNREVIEWED.keys())
def test_os_chain_needs_the_whole_reviewed_declaration(tmp_path, declaration):
    assert os_chains(tmp_path, mesh_mac=declaration) == []


@pytest.mark.parametrize("bias", ["column", "matrix"])
def test_a_bias_under_os_leaves_no_chain(tmp_path, bias):
    # Not even a declared full-tensor seed (the matrix bias) is observed under OS.
    assert os_chains(tmp_path, bias=bias, full_tensor_seed=True) == []
    assert os_chains(tmp_path, bias=bias) == []
    # Under WS the same bias is observed (a per-column stage, or a seed once it is declared), so only OS refuses it.
    assert os_chains(tmp_path, bias=bias, dataflow="WS", full_tensor_seed=True) != []
    if bias == "column":
        assert os_chains(tmp_path, bias=bias, dataflow="WS") != []
    else:
        assert os_chains(tmp_path, bias=bias, dataflow="WS") == []  # undeclared: the bias ends the chain


def os_extension_program():
    """The K-extension composition over a literal weight (an OS candidate): ``%W0`` is a constant, not an argument."""
    text = _swap(extension_program(), "%A: tensor<16x16xi8>, %W0: tensor<16x16xi8>, ", "%A: tensor<16x16xi8>, ")
    return _swap(
        text,
        "    %mac0_empty =",
        "    %W0 = arith.constant dense<1> : tensor<16x16xi8>\n    %mac0_empty =",
    )


def test_a_k_extension_under_os_leaves_no_chain(tmp_path):
    path = tmp_path / "os-extension.mlir"
    path.write_text(os_extension_program())
    kwargs = dict(numerics=NUMERICS, strategy=F32, max_k=131071, pooling_fenced=True)
    # Under WS the extension is observed (control), so only the dataflow refuses it.
    ws = prove_chains(path, dataflow="WS", **kwargs)
    assert [stage["stages"] for chain in ws for stage in chain["stages"]] == [["k_extension"], ["relu", "acc_scale"]]
    # Under OS the K extension would enter the partial sums the bound covers: max_k is forced off.
    assert prove_chains(path, dataflow="OS", mesh_mac=MESH, **kwargs) == []


def test_os_value_bound_does_not_loosen_other_dataflows(tmp_path):
    assert os_chains(tmp_path, dataflow="unknown_dataflow") == []
    assert os_chains(tmp_path, dataflow="BOTH") == []


# ---- the software spec's declaration -------------------------------------------------------------


def declared_mesh(**override):
    """A mesh_mac mapping with a fully correct ``os_value_bound`` declaration."""
    declaration = {
        "status": "reviewed",
        "mode": "signed_interval",
        "interval": [-524288, 524287],
        "proof": "constant_weight_all_prefix_intervals",
        "scope": "every_retained_partial_sum_including_total_os_k",
        "evidence": {"review_status": "reviewed", "basis": "RTL: PE.scala:58,64; MacUnit.sv"},
    }
    declaration.update(override)
    return {"dataflows": ["WS", "OS"], "signed_bits": 20, "os_value_bound": declaration}


def test_os_value_bound_declaration_accepts_the_correct_interval():
    software_spec._validate_os_value_bound(declared_mesh())
    # A different width states its own interval.
    narrow = declared_mesh(interval=[-(1 << 15), (1 << 15) - 1])
    narrow["signed_bits"] = 16
    software_spec._validate_os_value_bound(narrow)
    # An unreviewed declaration is well formed; only the observation requires ``reviewed``.
    software_spec._validate_os_value_bound(declared_mesh(status="unreviewed"))


@pytest.mark.parametrize(
    "override",
    [
        dict(interval=[-524287, 524287]),  # symmetric: not the signed two's-complement interval
        dict(interval=[-524288, 524288]),
        dict(interval=[-524288, 524287, 0]),
        dict(interval=None),
        dict(mode="absolute"),
        dict(mode=None),
        dict(proof="runtime_weight_bound"),
        dict(scope="final_sum_only"),
        dict(status="bogus"),
        dict(status=None),
        dict(evidence=None),
        dict(evidence={"review_status": "reviewed"}),
        dict(evidence={"basis": "   "}),
        dict(evidence="see packet"),
    ],
    ids=lambda override: ",".join(f"{key}={value!r}" for key, value in override.items()),
)
def test_os_value_bound_declaration_rejects_a_malformed_claim(override):
    with pytest.raises(ValueError, match="os_value_bound"):
        software_spec._validate_os_value_bound(declared_mesh(**override))


@pytest.mark.parametrize(
    "change", [dict(dataflows=["WS"]), dict(dataflows=None), dict(signed_bits=None), dict(signed_bits=1)]
)
def test_os_value_bound_declaration_needs_the_os_dataflow_and_a_mesh_width(change):
    with pytest.raises(ValueError, match="os_value_bound"):
        software_spec._validate_os_value_bound({**declared_mesh(), **change})


# ---- the provider obligation projection ----------------------------------------------------------

_INTERVAL = {
    "all": [
        {"path": "execution.pe_partial_min", "integer": True, "min": -524288, "max": 0},
        {"path": "execution.pe_partial_max", "integer": True, "min": 0, "max": 524287},
    ]
}


def provider_proof(**bound):
    return {
        "status": "proven",
        "bindings": [{"symbol": "k", "software_declaration": "contraction"}],
        "kernels": [{"symbol": "k", "schedule": {"bound_proof": bound}}],
    }


def enforce(**bound):
    spec = {
        "operations": [
            {
                "id": "contraction",
                "emission_obligations": {"program": {"status": "compiler_obligation", "condition": _INTERVAL}},
            }
        ]
    }
    proof = enforce_provider_emission_obligations(provider_proof(**bound), spec)
    return proof["kernels"][0]["emission_obligations"]["contraction"]["program"]["observation_status"]


def test_signed_interval_extremes_are_projected_as_pe_partial_min_and_max():
    # The extremes are the lowest column minimum and the highest column maximum; -2**19 is in the interval.
    assert enforce(column_lower=[-524288, -3], column_upper=[5, 524287]) == "matched"
    assert enforce(column_lower=[-3, -4], column_upper=[5, 6]) == "matched"


@pytest.mark.parametrize(
    "bound",
    [dict(column_lower=[-524289, 0], column_upper=[5, 5]), dict(column_lower=[-3, 0], column_upper=[5, 524288])],
    ids=["below_interval", "above_interval"],
)
def test_a_realized_bound_outside_the_declared_interval_violates_the_obligation(bound):
    with pytest.raises(ValueError, match="provider emission obligation contraction.program violated"):
        enforce(**bound)


@pytest.mark.parametrize(
    "bound",
    [
        {},
        dict(column_lower=[], column_upper=[]),
        dict(column_lower=[-3, -2], column_upper=[5]),  # lengths differ
        dict(column_lower=[-3.0], column_upper=[5]),  # not an exact int
        dict(column_lower=[True], column_upper=[5]),  # bool is not an int bound
        dict(column_lower=(-3,), column_upper=(5,)),  # not a list
    ],
    ids=["absent", "empty", "length_mismatch", "float", "bool", "tuple"],
)
def test_an_unreadable_interval_is_not_projected(bound):
    # No projection: the obligation stays unknown (absent observations are retained, never invented).
    assert enforce(**bound) == "unknown"


def test_the_absolute_bound_alternative_still_projects_independently():
    spec = {
        "operations": [
            {
                "id": "contraction",
                "emission_obligations": {
                    "program": {
                        "status": "compiler_obligation",
                        "condition": {"path": "execution.pe_partial_abs_bound", "integer": True, "max": 524287},
                    }
                },
            }
        ]
    }
    proof = enforce_provider_emission_obligations(provider_proof(max_mesh_partial_sum=524287), spec)
    assert proof["kernels"][0]["emission_obligations"]["contraction"]["program"]["observation_status"] == "matched"
    with pytest.raises(ValueError, match="violated"):
        enforce_provider_emission_obligations(provider_proof(max_mesh_partial_sum=524288), spec)


# --------------------------------------------------------------------------------------------------
# Accumulator seed
# --------------------------------------------------------------------------------------------------


def seed_declaration(**override):
    """A correct ``numerical_semantics.accumulator_seed_tensor`` declaration (``None`` deletes a key)."""
    declared = {
        "status": "reviewed",
        "forms": ["full_tensor"],
        "sources": ["constant", "runtime_input"],
        "seed_load": "unscaled_full_i32_shrink0_once",
        "accumulation": "wrapping_i32",
        "evidence": {"review_status": "reviewed", "basis": "RTL: AccumulatorMem.scala:70; LoadController.scala:108"},
    }
    declared.update(override)
    return {key: value for key, value in declared.items() if value is not None}


def seed_evidence(*, contract=None, scale_sources=None, domain=True, seed=True):
    """The program-admission fixture's evidence with the readout numerics (and optionally a scale domain).

    ``seed``: ``True`` declares the reviewed full-tensor seed, ``False`` none, a mapping is declared as given.
    """
    spec = copy.deepcopy(_SPEC)
    if scale_sources is not None:
        spec["operations"]["elementwise_map"]["scale_sources"] = scale_sources
    spec = validate_software_spec(spec, "fixture")
    spec["numerical_semantics"]["readout"] = dict(NUMERICS)
    if seed is not False:
        spec["numerical_semantics"]["accumulator_seed_tensor"] = seed_declaration() if seed is True else seed
    if domain:
        spec["quantization"] = {"numerics": {"policies": {"scale_source": {"runtime_input": runtime_policy()}}}}
    return SimpleNamespace(
        software_spec=spec, contract=contract or _CONTRACT, host_capabilities=_HOST, whole_program_admission=None
    )


def runtime_policy(**domain):
    """The reviewed runtime_input scale policy (``domain`` keys override; a ``None`` value deletes one)."""
    reviewed = {"status": "reviewed", "minimum": 0.0, "nonfinite": "forbid", "basis": "RTL: AccumulatorScale.scala"}
    reviewed.update(domain)
    return {
        "value": "runtime_input",
        "evidence": "rtl_derived",
        "domain": {key: value for key, value in reviewed.items() if value is not None},
    }


def written(tmp_path, text, name="capsule.interface.mlir"):
    path = tmp_path / name
    path.write_text(text)
    return path


def entries_of(path, evidence):
    return program_admission.account_program(path, name="c", target="fixture", evidence=evidence)


def by_indices(entries, indices):
    return [entry for entry in entries if (entry["observed_signature"].get("source_op_indices") or None) == indices]


@pytest.mark.parametrize(
    ("form", "source", "member"), [("matrix", "runtime_input", 5), ("rows_distinct", "constant", 6)]
)
def test_observe_replaces_the_seed_member_with_a_copy_of_the_admitted_producer_entry(tmp_path, form, source, member):
    evidence = seed_evidence()
    path = written(tmp_path, build(bias_form=form))
    entries = entries_of(path, evidence)
    (producer,) = [e for e in entries if e["observed_signature"].get("semantic_family") == "contraction"]
    assert producer["accelerator_admission"]["status"] == "admitted"
    [standalone] = by_indices(entries, [member])
    assert standalone["observed_signature"].get("observation") != "source_accumulator_seed"

    observed = observe(path, entries, evidence=evidence, readout_strategy=F32, dataflow="WS")

    [seeded] = by_indices(observed, [member])
    signature = seeded["observed_signature"]
    assert signature["observation"] == "source_accumulator_seed"
    assert signature["accumulator_seed"] == {
        "dtype": "i32",
        "shape": [16, 16],
        "source": source,
        "producer": 3,
    }
    # A copy of the producer's own entry: same family and admission, judged with it; named as the
    # seed load so no count reads it as a second contraction.
    assert seeded["operation"] == "accumulator_seed"
    assert signature["semantic_family"] == "contraction"
    assert seeded["accelerator_admission"] == producer["accelerator_admission"]
    assert {
        k: v for k, v in signature.items() if k not in {"observation", "accumulator_seed", "source_op_indices"}
    } == {k: v for k, v in producer["observed_signature"].items() if k not in {"source_op_indices"}}
    # The producer is untouched, and the seed carries no epilogue stage of its own.
    assert by_indices(observed, [3]) == by_indices(entries, [3])
    assert "bias_add" not in [e["operation"] for e in observed]
    # The readout is observed as the usual fused stages.
    assert {e["operation"] for e in observed if e["observed_signature"].get("observation") == "source_fused_stage"} == {
        "relu",
        "acc_scale",
    }


def test_observe_leaves_a_seed_unchanged_when_the_producer_is_not_admitted(tmp_path):
    evidence = seed_evidence()
    path = written(tmp_path, build(bias_form="matrix"))
    entries = entries_of(path, evidence)
    [producer] = [e for e in entries if e["observed_signature"].get("semantic_family") == "contraction"]
    producer["accelerator_admission"] = {**producer["accelerator_admission"], "status": "unsupported"}
    snapshot = copy.deepcopy(entries)
    assert observe(path, entries, evidence=evidence, readout_strategy=F32, dataflow="WS") == snapshot


def test_the_full_tensor_seed_needs_the_reviewed_declaration(tmp_path):
    path = written(tmp_path, build(bias_form="matrix"))
    declared = seed_evidence()
    entries = entries_of(path, declared)
    snapshot = copy.deepcopy(entries)
    assert observe(path, copy.deepcopy(entries), evidence=declared, readout_strategy=F32, dataflow="WS") != snapshot
    for seed in (
        False,
        seed_declaration(status="unreviewed"),
        seed_declaration(status=None),
        seed_declaration(forms=[]),
        seed_declaration(forms=["per_column"]),
        seed_declaration(forms=None),
        {},
    ):
        evidence = seed_evidence(seed=seed)
        assert not accumulator_seed_tensor(evidence.software_spec)
        # The bias ends the chain as before: no seed, no readout through it, every entry as it was.
        observed = observe(path, copy.deepcopy(entries), evidence=evidence, readout_strategy=F32, dataflow="WS")
        assert observed == snapshot, seed


@pytest.mark.parametrize("form", ["column", "rows_equal"])
def test_a_per_column_bias_does_not_need_the_seed_declaration(tmp_path, form):
    path = written(tmp_path, build(bias_form=form))
    evidence = seed_evidence(seed=False)
    observed = observe(path, entries_of(path, evidence), evidence=evidence, readout_strategy=F32, dataflow="WS")
    assert "bias_add" in [e["operation"] for e in observed]
    assert "accumulator_seed" not in [e["operation"] for e in observed]


def test_accumulator_seed_tensor_is_read_from_the_selected_spec():
    assert accumulator_seed_tensor({"numerical_semantics": {"accumulator_seed_tensor": seed_declaration()}}) is True
    assert accumulator_seed_tensor({}) is False
    assert accumulator_seed_tensor({"numerical_semantics": {}}) is False
    unreviewed = seed_declaration(status="unreviewed")
    assert accumulator_seed_tensor({"numerical_semantics": {"accumulator_seed_tensor": unreviewed}}) is False


def test_observe_does_not_seed_under_os_without_the_bound(tmp_path):
    evidence = seed_evidence()
    path = written(tmp_path, build(bias_form="matrix"))
    entries = entries_of(path, evidence)
    snapshot = copy.deepcopy(entries)
    assert observe(path, entries, evidence=evidence, readout_strategy=F32, dataflow="OS") == snapshot


def test_a_runtime_seed_has_no_bound_so_only_the_f32_product_readout_observes_through_it(tmp_path):
    def members_of(form, *, strategy, **kw):
        path = written(tmp_path, build(bias_form=form, **kw), name=f"{form}-{strategy}.mlir")
        return stage_members(prove_chains(path, numerics=NUMERICS, strategy=strategy, full_tensor_seed=True))

    # f32_product_rne observes through a runtime seed.
    assert members_of("matrix", strategy=F32) == [(5, []), (8, ["relu", "acc_scale"])]
    # Another strategy needs |acc| <= 2**24, which a runtime seed does not bound: the seed stands alone.
    assert members_of("matrix", strategy="exact_rational_host", gain="0.0625") == [(5, [])]
    # A constant seed has a bound (8 + 128 * 128 * 16 < 2**24), so its exact-strategy readout is proven.
    assert members_of("rows_distinct", strategy="exact_rational_host", gain="0.0625") == [
        (6, []),
        (9, ["relu", "acc_scale"]),
    ]


# ---- the declaration -----------------------------------------------------------------------------


def test_the_seed_declaration_validator_accepts_the_correct_declaration():
    software_spec._validate_accumulator_seed_tensor(seed_declaration())
    software_spec._validate_accumulator_seed_tensor(seed_declaration(sources=["constant"]))
    software_spec._validate_accumulator_seed_tensor(seed_declaration(sources=["runtime_input"]))
    software_spec._validate_accumulator_seed_tensor(seed_declaration(status="unreviewed"))


@pytest.mark.parametrize(
    "override",
    [
        dict(status=None),
        dict(status="bogus"),
        dict(forms=None),
        dict(forms=[]),
        dict(forms=["full_tensor", "per_column"]),
        dict(forms=["per_column"]),
        dict(forms="full_tensor"),
        dict(sources=None),
        dict(sources=[]),
        dict(sources=["bogus"]),
        dict(sources=["constant", "bogus"]),
        dict(seed_load=None),
        dict(seed_load="scaled_i32"),
        dict(accumulation=None),
        dict(accumulation="saturating_i32"),
        dict(evidence=None),
        dict(evidence={"review_status": "reviewed"}),
        dict(evidence={"basis": "   "}),
        dict(evidence="see packet"),
    ],
    ids=lambda override: ",".join(f"{key}={value!r}" for key, value in override.items()),
)
def test_the_seed_declaration_validator_rejects_a_malformed_declaration(override):
    with pytest.raises(ValueError, match="accumulator_seed_tensor"):
        software_spec._validate_accumulator_seed_tensor(seed_declaration(**override))


@pytest.mark.parametrize("declared", [None, [], "reviewed", ["full_tensor"]])
def test_the_seed_declaration_must_be_a_mapping(declared):
    with pytest.raises(ValueError, match="accumulator_seed_tensor"):
        software_spec._validate_accumulator_seed_tensor(declared)


def test_the_software_spec_validates_the_seed_declaration_where_it_is_declared():
    document = copy.deepcopy(_SPEC)
    document["numerical_semantics"]["accumulator_seed_tensor"] = seed_declaration()
    assert validate_software_spec(document, "fixture")["numerical_semantics"]["accumulator_seed_tensor"]
    document["numerical_semantics"]["accumulator_seed_tensor"] = seed_declaration(accumulation="saturating_i32")
    with pytest.raises(ValueError, match="accumulator_seed_tensor"):
        validate_software_spec(document, "fixture")


# --------------------------------------------------------------------------------------------------
# Runtime readout scale
# --------------------------------------------------------------------------------------------------

#: The function arguments of the runtime-scale program, in order: (A, W0, S).
SCALE_INPUTS = [
    {"name": "A0", "shape": [16, 16], "dtype": "i8"},
    {"name": "W0", "shape": [16, 16], "dtype": "i8"},
    {"name": "S", "shape": [1], "dtype": "f32"},
]


def _swap(text, old, new, count=1):
    assert text.count(old) == count, old
    return text.replace(old, new)


def scale_program(*, scale="argument", scale_type="tensor<1xf32>", relu=True, bias=False, bias_form="rows_equal"):
    """The composition template with ``%S`` (a ``tensor<1xf32>`` argument) read through ``(d0, d1) -> (0)``.

    ``scale``: ``argument`` (the readout reads ``%S``), ``computed`` (a value derived from ``%S``) or
    ``constant`` (a literal): the argument ``%S`` stays in the signature in every case.
    """
    text = build(gain_arg=True, bias=bias, bias_form=bias_form, relu=relu)
    text = _swap(text, "%readout_gains: tensor<16xf32>", f"%S: {scale_type}")
    text = _swap(text, "affine_map<(d0,d1)->(d1)>", "affine_map<(d0,d1)->(0)>")
    read = "%S"
    if scale == "computed":
        text = _swap(text, "    %readout_empty =", f"    %S2 = arith.negf %S : {scale_type}\n    %readout_empty =")
        read = "%S2"
    elif scale == "constant":
        text = _swap(
            text,
            "    %readout_empty =",
            "    %S2 = arith.constant dense<[0.1666666716337204]> : tensor<1xf32>\n    %readout_empty =",
        )
        read = "%S2"
    return _swap(
        text, "%readout_gains : tensor<16x16xi32>, tensor<16xf32>", f"{read} : tensor<16x16xi32>, {scale_type}"
    )


def scale_chains(tmp_path, *, runtime_scale=2, inputs=SCALE_INPUTS, strategy=F32, **program):
    path = written(tmp_path, scale_program(**program), name="scale.mlir")
    return prove_chains(path, numerics=NUMERICS, strategy=strategy, runtime_scale=runtime_scale, inputs=inputs)


def runtime_readouts(chains):
    return [
        stage
        for chain in chains
        for stage in chain["stages"]
        if "acc_scale" in stage["stages"] and stage.get("scale_source") == "runtime_input"
    ]


def test_a_tensor_argument_scale_is_observed_at_its_declared_position_under_f32_product_rne(tmp_path):
    (chain,) = scale_chains(tmp_path)
    (readout,) = runtime_readouts([chain])
    assert readout["stages"] == ["relu", "acc_scale"]
    assert readout["gains"] is None
    assert readout["result"] == {"dtype": "i8", "shape": [16, 16]}
    assert chain["producer"] == 3 and readout["member"] == 5


def test_a_runtime_scale_is_observed_without_a_relu_too(tmp_path):
    (readout,) = runtime_readouts(scale_chains(tmp_path, relu=False))
    assert readout["stages"] == ["acc_scale"]


@pytest.mark.parametrize("strategy", ["exact_rational_host", "half_away", None])
def test_a_runtime_scale_is_refused_under_any_other_strategy(tmp_path, strategy):
    assert scale_chains(tmp_path, strategy=strategy) == []


@pytest.mark.parametrize("position", [0, 1, 3, -1, None])
def test_a_runtime_scale_is_refused_at_a_different_position(tmp_path, position):
    assert scale_chains(tmp_path, runtime_scale=position) == []


def parsed_ops(tmp_path, **program):
    """The names of the top-level ops of a scale program: proves the refusal is not a failure to parse."""
    from merlin.common import mlir_query as mq

    top, _, _ = _top_level(written(tmp_path, scale_program(**program), name="parsed.mlir"))
    return [mq.op_name(op) for op in top]


def test_a_two_element_scale_tensor_is_refused(tmp_path):
    inputs = [*SCALE_INPUTS[:2], {"name": "S", "shape": [2], "dtype": "f32"}]
    assert parsed_ops(tmp_path, scale_type="tensor<2xf32>")[-1] == "linalg.generic"
    assert scale_chains(tmp_path, scale_type="tensor<2xf32>", inputs=inputs) == []


@pytest.mark.parametrize(("scale", "defining_op"), [("computed", "arith.negf"), ("constant", "arith.constant")])
def test_a_scale_that_is_not_a_function_argument_is_refused(tmp_path, scale, defining_op):
    # ``%S`` stays an argument at position 2 and the declared inputs match, so only the source of the gain differs.
    assert defining_op in parsed_ops(tmp_path, scale=scale)
    assert scale_chains(tmp_path, scale=scale) == []
    assert runtime_readouts(scale_chains(tmp_path)) != []


@pytest.mark.parametrize(
    "inputs",
    [
        None,
        [],
        SCALE_INPUTS[:2],
        [*SCALE_INPUTS, {"name": "extra", "shape": [1], "dtype": "f32"}],
        [SCALE_INPUTS[0], SCALE_INPUTS[2], SCALE_INPUTS[1]],
        [*SCALE_INPUTS[:2], {**SCALE_INPUTS[2], "dtype": "f16"}],
        [*SCALE_INPUTS[:2], {**SCALE_INPUTS[2], "shape": [1, 1]}],
        [{**SCALE_INPUTS[0], "shape": [8, 16]}, *SCALE_INPUTS[1:]],
    ],
    ids=["none", "empty", "fewer", "extra", "reordered", "scale_dtype", "scale_shape", "other_shape"],
)
def test_inputs_that_differ_from_the_function_arguments_disable_the_runtime_scale(tmp_path, inputs):
    assert scale_chains(tmp_path, inputs=inputs) == []


def test_a_runtime_scale_is_never_observed_when_none_is_declared(tmp_path):
    path = written(tmp_path, scale_program(), name="undeclared.mlir")
    assert prove_chains(path, numerics=NUMERICS, strategy=F32) == []
    assert prove_chains(path, numerics=NUMERICS, strategy=F32, inputs=SCALE_INPUTS) == []


def test_stage_rows_of_a_runtime_chain_state_the_runtime_scale_source(tmp_path):
    (chain,) = scale_chains(tmp_path)
    rows = stage_rows(chain)
    assert list(rows) == [5]
    assert [row["operation"] for row in rows[5]] == ["relu", "acc_scale"]
    for row in rows[5]:
        assert row["composed_observation"]["scale_source"] == "runtime_input"
        assert row["composed_observation"]["scale_granularity"] == "tensor"
        assert "logical_output_column_source" not in row
    # A constant gain states no source (a command constant).
    (constant,) = prove_chains(written(tmp_path, build(), name="const.mlir"), numerics=NUMERICS, strategy=F32)
    for member_rows in stage_rows(constant).values():
        for row in member_rows:
            assert "scale_source" not in row["composed_observation"]


# ---- a runtime scale needs a clean accumulator ---------------------------------------------------


def chains_with_scale(tmp_path, text, *, inputs, runtime_scale, **kw):
    path = written(tmp_path, text, name="clean.mlir")
    return prove_chains(path, numerics=NUMERICS, strategy=F32, runtime_scale=runtime_scale, inputs=inputs, **kw)


def test_a_runtime_scale_is_not_observed_behind_a_bias_add(tmp_path):
    # The same readout is observed on a clean accumulator ...
    assert runtime_readouts(scale_chains(tmp_path)) != []
    # ... and behind a row-constant bias (args A, W0, S) it is not: the bias stage stands alone.
    text = scale_program(bias_form="rows_equal", bias=True)
    chains = chains_with_scale(tmp_path, text, inputs=SCALE_INPUTS, runtime_scale=2)
    assert stage_members(chains) == [(6, ["bias_add"])]
    assert runtime_readouts(chains) == []
    # Behind a per-column runtime bias vector (args A, W0, Bias, S) likewise.
    text = scale_program(bias_form="column", bias=True)
    inputs = [*SCALE_INPUTS[:2], {"name": "Bias", "shape": [16], "dtype": "i32"}, SCALE_INPUTS[2]]
    chains = chains_with_scale(tmp_path, text, inputs=inputs, runtime_scale=3)
    assert stage_members(chains) == [(5, ["bias_add"])]
    assert runtime_readouts(chains) == []


def test_a_runtime_scale_is_not_observed_behind_a_seed(tmp_path):
    text = scale_program(bias_form="rows_distinct", bias=True)
    chains = chains_with_scale(tmp_path, text, inputs=SCALE_INPUTS, runtime_scale=2, full_tensor_seed=True)
    assert stage_members(chains) == [(6, [])]  # the seed stands, the runtime readout is refused
    assert runtime_readouts(chains) == []
    # The same program with a constant gain would be read out behind the seed.
    constant = chains_with_scale(
        tmp_path, build(bias_form="rows_distinct"), inputs=None, runtime_scale=None, full_tensor_seed=True
    )
    assert stage_members(constant) == [(6, []), (9, ["relu", "acc_scale"])]


def test_a_runtime_scale_is_not_observed_behind_a_k_extension(tmp_path):
    text = extension_program()
    text = _swap(text, "%Residual: tensor<16x16xi8>)", "%Residual: tensor<16x16xi8>, %S: tensor<1xf32>)")
    text = _swap(
        text,
        f"    %readout_gains = arith.constant dense<[{', '.join(['0.1666666716337204'] * 16)}]> : tensor<16xf32>\n",
        "",
    )
    text = _swap(text, "affine_map<(d0,d1)->(d1)>", "affine_map<(d0,d1)->(0)>")
    text = _swap(text, "%readout_gains : tensor<16x16xi32>, tensor<16xf32>", "%S : tensor<16x16xi32>, tensor<1xf32>")
    inputs = [
        {"name": "A0", "shape": [16, 16], "dtype": "i8"},
        {"name": "W0", "shape": [16, 16], "dtype": "i8"},
        {"name": "R", "shape": [16, 16], "dtype": "i8"},
        {"name": "S", "shape": [1], "dtype": "f32"},
    ]
    chains = chains_with_scale(tmp_path, text, inputs=inputs, runtime_scale=3, max_k=131071)
    assert [stage["stages"] for chain in chains for stage in chain["stages"]] == [["k_extension"]]
    assert runtime_readouts(chains) == []


# ---- the reviewed domain -------------------------------------------------------------------------


def spec_with_policy(policy):
    return {"quantization": {"numerics": {"policies": {"scale_source": {"runtime_input": policy}}}}}


def test_runtime_scale_domain_returns_the_reviewed_domain():
    assert runtime_scale_domain(spec_with_policy(runtime_policy())) == {"minimum": 0.0, "nonfinite": "forbid"}
    assert runtime_scale_domain(spec_with_policy(runtime_policy(minimum=0))) == {
        "minimum": 0.0,
        "nonfinite": "forbid",
    }
    assert runtime_scale_domain(spec_with_policy(runtime_policy(minimum=0.5))) == {
        "minimum": 0.5,
        "nonfinite": "forbid",
    }


def _without_domain():
    policy = runtime_policy()
    del policy["domain"]
    return policy


NO_DOMAIN = {
    "minimum_negative": runtime_policy(minimum=-1),
    "minimum_tiny_negative": runtime_policy(minimum=-1e-30),
    "nonfinite_allow": runtime_policy(nonfinite="allow"),
    "nonfinite_missing": runtime_policy(nonfinite=None),
    "status_unreviewed": runtime_policy(status="unreviewed"),
    "status_missing": runtime_policy(status=None),
    "minimum_inf": runtime_policy(minimum=float("inf")),
    "minimum_nan": runtime_policy(minimum=float("nan")),
    "minimum_bool": runtime_policy(minimum=True),
    "minimum_string": runtime_policy(minimum="0.0"),
    "minimum_missing": runtime_policy(minimum=None),
    "domain_missing": _without_domain(),
    "evidence_host_definition": {**runtime_policy(), "evidence": "host_definition"},
    "value_other": {**runtime_policy(), "value": "constant"},
}


@pytest.mark.parametrize("policy", NO_DOMAIN.values(), ids=NO_DOMAIN.keys())
def test_runtime_scale_domain_is_none_unless_reviewed_finite_and_non_negative(policy):
    assert runtime_scale_domain(spec_with_policy(policy)) is None


@pytest.mark.parametrize("spec", [{}, {"quantization": None}, {"quantization": {"numerics": {}}}], ids=str)
def test_runtime_scale_domain_is_none_without_a_runtime_input_policy(spec):
    assert runtime_scale_domain(spec) is None


def test_quantization_validator_accepts_a_correct_scale_domain():
    quantization_numerics._validate_scale_domain(runtime_policy()["domain"], "d")
    quantization_numerics._validate_scale_domain(runtime_policy(minimum=-3, nonfinite="allow")["domain"], "d")
    # ``unreviewed`` is a legal status: only the observation requires ``reviewed``.
    quantization_numerics._validate_scale_domain(runtime_policy(status="unreviewed")["domain"], "d")


@pytest.mark.parametrize(
    "domain",
    [
        {"status": "reviewed", "minimum": 0.0, "nonfinite": "forbid"},  # basis missing
        {"minimum": 0.0, "nonfinite": "forbid", "basis": "x"},  # status missing
        {"status": "reviewed", "nonfinite": "forbid", "basis": "x"},  # minimum missing
        {"status": "reviewed", "minimum": 0.0, "basis": "x"},  # nonfinite missing
        {"status": "reviewed", "minimum": 0.0, "nonfinite": "forbid", "basis": "x", "extra": 1},
        {"status": "reviewed", "minimum": float("inf"), "nonfinite": "forbid", "basis": "x"},
        {"status": "reviewed", "minimum": float("-inf"), "nonfinite": "forbid", "basis": "x"},
        {"status": "reviewed", "minimum": float("nan"), "nonfinite": "forbid", "basis": "x"},
        {"status": "reviewed", "minimum": True, "nonfinite": "forbid", "basis": "x"},
        {"status": "reviewed", "minimum": "0", "nonfinite": "forbid", "basis": "x"},
        {"status": "reviewed", "minimum": 0.0, "nonfinite": "maybe", "basis": "x"},
        {"status": "bogus", "minimum": 0.0, "nonfinite": "forbid", "basis": "x"},
        {"status": "reviewed", "minimum": 0.0, "nonfinite": "forbid", "basis": "  "},
        {"status": "reviewed", "minimum": 0.0, "nonfinite": "forbid", "basis": None},
        None,
        [],
        "domain",
    ],
)
def test_quantization_validator_rejects_a_malformed_scale_domain(domain):
    with pytest.raises(ValueError, match="quantization.numerics"):
        quantization_numerics._validate_scale_domain(
            domain, "quantization.numerics.policies.scale_source.runtime_input.domain"
        )


# ---- the screen of the capsule's own scale input -------------------------------------------------

DOMAIN = {"minimum": 0.0, "nonfinite": "forbid"}


def golden_with(monkeypatch, scale, name="S"):
    monkeypatch.setattr(
        numeric_domains.golden_store,
        "load_golden",
        lambda directory: {"oracle_provenance": {"inputs": {name: {"decoded": scale}}}},
    )


def screen_scale(monkeypatch, scale, domain=DOMAIN, name="S"):
    golden_with(monkeypatch, scale, name="S")
    return numeric_domains.screen_runtime_scale({"inputs": []}, Path("."), name, domain)


def test_a_capsule_without_a_reviewed_domain_is_unknown(monkeypatch):
    golden_with(monkeypatch, [0.1666])
    screen = numeric_domains.screen_runtime_scale({}, Path("."), "S", None)
    assert screen["status"] == "unknown" and screen["input"] == "S"
    assert "no reviewed runtime readout-scale domain" in screen["reason"]


@pytest.mark.parametrize("scale", [[0.1666], [0.0], [-0.0], [[3.5]], [1, 2.5], [1e30]])
def test_a_scale_within_the_domain_is_screened_within_domain(monkeypatch, scale):
    screen = screen_scale(monkeypatch, scale)
    assert screen["status"] == "within_domain"
    assert screen["domain"] == DOMAIN and screen["input"] == "S"


@pytest.mark.parametrize(
    "scale", [[-0.5], [float("inf")], [float("-inf")], [float("nan")], [0.5, -1e-9], [[1.0, float("nan")]]]
)
def test_a_scale_outside_the_domain_is_unsupported(monkeypatch, scale):
    screen = screen_scale(monkeypatch, scale)
    assert screen["status"] == "unsupported"
    assert "outside the reviewed domain" in screen["reason"]


def test_the_domain_minimum_is_held(monkeypatch):
    assert screen_scale(monkeypatch, [0.4], domain={"minimum": 0.5, "nonfinite": "forbid"})["status"] == "unsupported"
    assert screen_scale(monkeypatch, [0.5], domain={"minimum": 0.5, "nonfinite": "forbid"})["status"] == "within_domain"


@pytest.mark.parametrize("scale", [[], [[]], "0.5", ["0.5"], [None]])
def test_an_unreadable_or_empty_scale_is_unknown(monkeypatch, scale):
    assert screen_scale(monkeypatch, scale)["status"] == "unknown"


def test_a_scale_input_missing_from_the_reference_is_unknown(monkeypatch):
    golden_with(monkeypatch, [0.5], name="other")
    assert numeric_domains.screen_runtime_scale({}, Path("."), "S", DOMAIN)["status"] == "unknown"


def test_an_unreadable_golden_is_unknown(monkeypatch):
    def refuse(directory):
        raise OSError("no golden")

    monkeypatch.setattr(numeric_domains.golden_store, "load_golden", refuse)
    assert numeric_domains.screen_runtime_scale({}, Path("."), "S", DOMAIN)["status"] == "unknown"


def test_a_golden_without_input_provenance_falls_back_to_the_capsule_leaves(monkeypatch):
    monkeypatch.setattr(numeric_domains.golden_store, "load_golden", lambda directory: {"outputs": {}})
    monkeypatch.setattr(
        numeric_domains,
        "materialize_capsule_leaves",
        lambda capsule: {"S": SimpleNamespace(data=[0.25])},
    )
    assert numeric_domains.screen_runtime_scale({}, Path("."), "S", DOMAIN)["status"] == "within_domain"
    monkeypatch.setattr(
        numeric_domains, "materialize_capsule_leaves", lambda capsule: {"S": SimpleNamespace(data=[-0.25])}
    )
    assert numeric_domains.screen_runtime_scale({}, Path("."), "S", DOMAIN)["status"] == "unsupported"


# ---- the declaration side: scale_sources ---------------------------------------------------------

FUSED_SIGNATURE = {
    "family": "elementwise_map",
    "operand_dtype": "i8",
    "readout_dtype": "i32",
    "composed_with": ["contraction"],
    "epilogues": ["relu", "acc_scale"],
    "scale_granularity": "tensor",
}


def declared_spec(scale_sources):
    document = copy.deepcopy(_SPEC)
    if scale_sources is not None:
        document["operations"]["elementwise_map"]["scale_sources"] = scale_sources
    return validate_software_spec(document, "fixture")


def admit(spec, **signature):
    return admit_operation(spec, "relu", {**FUSED_SIGNATURE, **signature}, "fused_accelerator")


def test_a_declaration_with_scale_sources_admits_a_runtime_input_scale():
    spec = declared_spec(["constant", "runtime_input"])
    assert admit(spec, scale_source="runtime_input")["status"] == "admitted"
    assert admit(spec, scale_source="constant")["status"] == "admitted"
    assert admit(spec)["status"] == "admitted"


@pytest.mark.parametrize("scale_sources", [None, ["constant"]], ids=["undeclared", "constant_only"])
def test_a_declaration_without_runtime_input_refuses_it(scale_sources):
    spec = declared_spec(scale_sources)
    refused = admit(spec, scale_source="runtime_input")
    assert refused["status"] == "unsupported"
    assert "scale_source 'runtime_input' is not admitted" in refused["reason"]
    # An observation without a source is a command constant, so both declarations admit it as before.
    assert admit(spec)["status"] == "admitted"
    assert admit(spec, scale_source="constant")["status"] == "admitted"


def test_a_runtime_only_declaration_refuses_a_constant_scale():
    spec = declared_spec(["runtime_input"])
    assert admit(spec, scale_source="runtime_input")["status"] == "admitted"
    refused = admit(spec)
    assert refused["status"] == "unsupported" and "scale_source 'constant' is not admitted" in refused["reason"]


@pytest.mark.parametrize("bad", [["bogus"], [], "runtime_input", ["constant", "bogus"], [None], {"constant": True}])
def test_validation_rejects_malformed_scale_sources(bad):
    with pytest.raises(ValueError, match="scale_sources must list"):
        declared_spec(bad)


def test_observe_admits_a_runtime_scale_only_under_a_declaration_that_names_the_source(tmp_path):
    hardware = copy.deepcopy(_CONTRACT)
    hardware["compute_units"][0]["semantic_capabilities"][1]["result_dtypes"] = ["i32", "i8"]
    path = written(tmp_path, scale_program(), name="admit.mlir")

    def stages(scale_sources, *, domain=True, **kw):
        evidence = seed_evidence(contract=hardware, scale_sources=scale_sources, domain=domain)
        entries = entries_of(path, evidence)
        observed = observe(
            path,
            entries,
            evidence=evidence,
            readout_strategy=F32,
            dataflow="WS",
            runtime_scale=kw.get("runtime_scale", 2),
            inputs=kw.get("inputs", SCALE_INPUTS),
        )
        return [e for e in observed if e["observed_signature"].get("observation") == "source_fused_stage"], observed

    fused, _ = stages(["constant", "runtime_input"])
    assert [e["operation"] for e in fused] == ["relu", "acc_scale"]
    for entry in fused:
        assert entry["observed_signature"]["composed_observation"]["scale_source"] == "runtime_input"
        assert [row["status"] for row in entry["software_admissions"]] == ["admitted"]
        assert entry["accelerator_admission"]["status"] == "admitted"

    fused, _ = stages(None)
    assert [e["operation"] for e in fused] == ["relu", "acc_scale"]
    for entry in fused:
        [row] = entry["software_admissions"]
        assert row["status"] == "unsupported" and "scale_source 'runtime_input' is not admitted" in row["reason"]
        assert entry["accelerator_admission"]["status"] != "admitted"

    # Without the reviewed domain the readout is not observed at all (the standalone entries stand).
    fused, observed = stages(["constant", "runtime_input"], domain=False)
    assert fused == []
    # ...and without a declared position likewise.
    fused, _ = stages(["constant", "runtime_input"], runtime_scale=None)
    assert fused == []


# --------------------------------------------------------------------------------------------------
# The reviewed Gemmini declarations
# --------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gemmini():
    return load_software_spec(GEMMINI_SPEC)


def test_gemmini_elementwise_map_admits_constant_and_runtime_input_scales(gemmini):
    [row] = [row for row in gemmini["operations"] if row["id"] == "elementwise_map"]
    assert row["signature"]["scale_sources"] == ["constant", "runtime_input"]
    # The standalone and operand-sum maps keep the constant-only default.
    for other in gemmini["operations"]:
        if other["id"] != "elementwise_map":
            assert "scale_sources" not in other["signature"], other["id"]


def test_gemmini_declares_the_reviewed_os_value_bound(gemmini):
    mesh_mac = gemmini["numerical_semantics"]["native_unit_contract"]["mesh_mac"]
    declared = mesh_mac["os_value_bound"]
    assert declared["status"] == "reviewed"
    assert declared["mode"] == "signed_interval"
    assert declared["interval"] == [-524288, 524287]
    assert declared["proof"] == "constant_weight_all_prefix_intervals"
    assert "OS" in mesh_mac["dataflows"] and mesh_mac["signed_bits"] == 20
    assert declared["interval"] == [-(1 << 19), (1 << 19) - 1]
    software_spec._validate_os_value_bound(mesh_mac)


def test_gemmini_contraction_accepts_the_signed_interval_as_an_alternative_to_the_absolute_bound(gemmini):
    [row] = [row for row in gemmini["operations"] if row["id"] == "contraction"]
    condition = row["emission_obligations"]["execution_program"]["condition"]
    (pe_bound,) = [child for child in condition["all"] if "any" in child]
    paths = [{leaf["path"] for leaf in alternative["all"]} for alternative in pe_bound["any"]]
    assert "execution.pe_partial_abs_bound" in paths[0]
    assert {"execution.pe_partial_min", "execution.pe_partial_max"} <= paths[1]
    assert all("execution.pe_bound_scope" in alternative for alternative in paths)


def test_gemmini_runtime_input_domain_is_reviewed_so_a_runtime_scale_is_observed(gemmini):
    policy = gemmini["quantization"]["numerics"]["policies"]["scale_source"]["runtime_input"]
    assert policy["domain"]["status"] == "reviewed"
    assert "OPERATOR-SIGNOFF-20261028-ADDENDUM.txt sha256 c8a13df3" in " ".join(policy["domain"]["basis"].split())
    assert policy["domain"]["minimum"] == 0.0
    assert policy["domain"]["nonfinite"] == "forbid"
    assert runtime_scale_domain(gemmini) == {"minimum": 0.0, "nonfinite": "forbid"}
    # Only a reviewed domain lets Phase 0 observe a runtime scale.
    unreviewed = copy.deepcopy(gemmini)
    unreviewed["quantization"]["numerics"]["policies"]["scale_source"]["runtime_input"]["domain"]["status"] = "unreviewed"
    assert runtime_scale_domain(unreviewed) is None


def test_gemmini_full_tensor_seed_is_reviewed_so_a_seed_is_observed(gemmini):
    declared = gemmini["numerical_semantics"]["accumulator_seed_tensor"]
    assert declared["status"] == "reviewed"
    assert "OPERATOR-SIGNOFF-20261028-ADDENDUM.txt sha256 c8a13df3" in declared["evidence"]["reviewed_by"]
    assert declared["forms"] == ["full_tensor"] and declared["sources"] == ["constant", "runtime_input"]
    assert declared["seed_load"] == "unscaled_full_i32_shrink0_once" and declared["accumulation"] == "wrapping_i32"
    software_spec._validate_accumulator_seed_tensor(declared)
    assert accumulator_seed_tensor(gemmini) is True
    unreviewed = copy.deepcopy(gemmini)
    unreviewed["numerical_semantics"]["accumulator_seed_tensor"]["status"] = "unreviewed"
    assert accumulator_seed_tensor(unreviewed) is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda mesh_mac: mesh_mac["os_value_bound"].update(interval=[-524287, 524287]),
        lambda mesh_mac: mesh_mac["os_value_bound"].update(mode="absolute"),
        lambda mesh_mac: mesh_mac["os_value_bound"].pop("evidence"),
        lambda mesh_mac: mesh_mac.update(dataflows=["WS"]),
    ],
    ids=["asymmetric_interval", "mode", "evidence", "no_os"],
)
def test_the_loaded_gemmini_spec_is_validated_on_the_mesh_declaration(mutate):
    document = yaml.safe_load(GEMMINI_SPEC.read_text())
    mutate(document["numerical_semantics"]["native_unit_contract"]["mesh_mac"])
    with pytest.raises(ValueError, match="os_value_bound"):
        validate_software_spec(document, source="mutated")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda declared: declared.update(forms=["per_column"]),
        lambda declared: declared.update(sources=[]),
        lambda declared: declared.update(seed_load="scaled"),
        lambda declared: declared.update(accumulation="saturating"),
        lambda declared: declared.pop("evidence"),
    ],
    ids=["forms", "sources", "seed_load", "accumulation", "evidence"],
)
def test_the_loaded_gemmini_spec_is_validated_on_the_seed_declaration(mutate):
    document = yaml.safe_load(GEMMINI_SPEC.read_text())
    mutate(document["numerical_semantics"]["accumulator_seed_tensor"])
    with pytest.raises(ValueError, match="accumulator_seed_tensor"):
        validate_software_spec(document, source="mutated")
