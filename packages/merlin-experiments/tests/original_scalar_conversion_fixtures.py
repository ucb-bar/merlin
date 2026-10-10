"""Independent declared one-call examples, then actual public schema/conversion.

No new model capture creates these declared original graphs. The ordinary
one-operator importer supplies its own actual trace for construction checking.
These artificial members never replace the protected original denominator.
"""

import json
import os
from pathlib import Path

import pytest
from merlin_experiments.phase0 import operator_schema_intake as O
from merlin_experiments.phase0 import original_call_sources as C
from merlin_experiments.phase0 import original_scalar_conversion as V
from merlin_experiments.phase0.component_semantic_basis import PROVENANCE, SCHEMA, BasisSource, ComponentSemanticBasis
from merlin_experiments.phase0.rtl_intake import issue_independent_hardware_intake
from merlin_experiments.phase0.software_intake import REVIEW_SCHEMA, issue_independent_software_intake
from test_original_scalar_binary_sources import declarations

from merlin.targetgen.rtl.source_selection import produce_selection

DECLARED_CALLS = (
    ("aten.mul.Tensor", 1.0),
    ("aten.div.Tensor", 2.23606797749979),
    ("aten.mul.Tensor", -0.0),
    ("aten.div.Tensor", 0.1),
    ("aten.mul.Tensor", 1.0 + 2**-24),
    ("aten.mul.Tensor", 1),
)


def conversion_budget(*, version=1):
    if type(version) is not int or version not in {1, 2}:
        raise ValueError("native scalar fixture requires its explicit supported version")
    budget = {
        "max_source_bytes": 2000000,
        "max_nesting": 64,
        "max_operations": 30,
        "max_tensor_elements": 1000,
        "max_members": 100,
        "max_total_tensor_elements": 10000,
        "max_total_source_bytes": 100000000,
        "timeout_s": 180,
    }
    if version == 2:
        source_slots = len(DECLARED_CALLS) * len(C.required_source_cohorts())
        if source_slots > budget["max_members"]:
            raise ValueError("complete native fixture exceeds its original slot bound")
        # Every original slot reserves one complete source and three conversion
        # products. The enclosing observation frame gets its own full bound.
        product_slots = 3 * source_slots + 1
        budget["max_total_source_bytes"] = (source_slots + product_slots) * budget["max_source_bytes"]
        budget.update(max_promotion_tensor_elements=1000, max_total_promotion_tensor_elements=10000)
    return budget


def write(path, value):
    from hashlib import sha256

    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    path.chmod(0o600)
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


@pytest.fixture(scope="module")
def native_originals(tmp_path_factory):
    return prepare_native_originals(tmp_path_factory)


@pytest.fixture(scope="module")
def integer_native_originals(tmp_path_factory):
    return prepare_native_originals(tmp_path_factory, version=2)


def prepare_native_originals(tmp_path_factory, *, version=1):
    """Keep v1 defaults; v2 freshly observes the original integer Tensor binding."""
    if type(version) is not int or version not in {1, 2}:
        raise ValueError("native scalar fixture requires its explicit supported version")
    names = (
        "MERLIN_TEST_TORCH_PYTHON",
        "MERLIN_TEST_M2M_ROOT",
        "MERLIN_TEST_M2M_COMMIT",
        "MERLIN_TEST_OPERATOR_DECLARATIONS",
        "MERLIN_TEST_TORCH_SOURCE_ROOT",
        "MERLIN_TEST_FIRTOOL",
        "MERLIN_TEST_MLIR_OPT",
    )
    if version == 2:
        names += ("MERLIN_TEST_TENSOR_ARGUMENT_COMPILER",)
    if any(not os.environ.get(name) for name in names):
        pytest.skip("registered scalar controls require explicit public/native source and tool selections")
    python, capture, commit, declarations_path, checkout, firtool, parser = (os.environ[name] for name in names[:7])
    owner = tmp_path_factory.mktemp("independent-original-scalar-conversion")
    source = owner / "unit.fir"
    source.write_text(
        "FIRRTL version 2.0.0\ncircuit Unit :\n"
        "  module Unit : @[generators/test_unit/src/Independent.scala 1:1]\n"
        "    input x : UInt<8>\n    output y : UInt<8>\n    y <= x\n"
    )
    bundle = produce_selection(
        target="test_unit",
        firrtl=source,
        generator="test_unit",
        config="IndependentUnit",
        core_root="Unit",
        firtool=Path(firtool),
        output=owner / "rtl",
    )
    descriptor = owner / "target.yaml"
    descriptor.write_text("target: test_unit\n")
    forbidden = (owner / "absent-private-answer-prefix",)
    hardware = issue_independent_hardware_intake(
        target="test_unit",
        descriptor=descriptor,
        source_bundle=bundle,
        forbidden_roots=forbidden,
        output=owner / "hardware",
    )
    members, links = [], []
    for index, (target, literal) in enumerate(DECLARED_CALLS):
        trace = declarations(target, literal)[0]
        pin = write(owner / f"declared-{index}.json", trace)
        members.append(
            {
                "id": f"independent-{index}",
                "kind": "model2mlir_frontend_trace",
                **pin,
                "schema": trace["schema"],
                "operation_semantics": [target],
                "effect_semantics": [],
            }
        )
        links.append({"owner": "scalar_map", "member": f"independent-{index}", "operations": [target]})
    roster = owner / "basis.json"
    basis_pin = write(roster, {"schema": SCHEMA, "status": "reviewed", "provenance": PROVENANCE, "members": members})
    basis = ComponentSemanticBasis.load(
        roster.read_bytes(),
        source=BasisSource(str(roster), basis_pin["sha256"], "semantic-basis-roster"),
        parent=owner,
        routing={},
    )
    # Preserve the incompatible historical integer gate; this fixture selects
    # no f32 arithmetic policy or new numerical admission.
    numerics = {
        "operand_dtype": "int8",
        "accumulator_dtype": "i32",
        "readout_dtype": "i32",
        "model": {"engine": "integer_reference"},
        "subnormal_operand_flush": False,
        "overflow": "bounded_exact",
    }
    spec = owner / "software.json"
    spec_pin = write(
        spec,
        {
            "schema": "merlin.software_spec.v1",
            "target": "test_unit",
            "status": "reviewed",
            "numerical_semantics": numerics,
            "operations": {"scalar_map": {"families": ["elementwise_map"], "hardware": "standalone"}},
        },
    )
    review = owner / "review.json"
    write(
        review,
        {
            "schema": REVIEW_SCHEMA,
            "target": "test_unit",
            "source": spec_pin,
            "semantic_basis": basis_pin,
            "numerical_choices": numerics,
            "operation_basis": links,
        },
    )
    software = issue_independent_software_intake(
        hardware=hardware, source=spec, review=review, forbidden_roots=forbidden, output_root=owner / "software"
    )
    schema_selection = owner / "schema-selection.json"
    write(
        schema_selection,
        {
            "schema": O.TENSOR_SELECTION_SCHEMA if version == 2 else O.SELECTION_SCHEMA,
            "status": "reviewed",
            "software_intake_sha256": software.sha256,
            "namespace": "aten",
            "python": python,
            "canonical_source": {
                "checkout": checkout,
                "commit": "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0",
                "path": declarations_path,
            },
            **(
                {"tensor_arguments": {"compiler": os.environ["MERLIN_TEST_TENSOR_ARGUMENT_COMPILER"]}}
                if version == 2
                else {}
            ),
        },
    )
    intake = O.issue_independent_operator_schema_intake(
        software=software, selection=schema_selection, forbidden_roots=forbidden, output=owner / "schemas"
    )
    sources = C.observe(
        schema_record=intake.record(),
        basis=basis,
        numerical_semantics=numerics,
        budget={"schema": C.BUDGET_SCHEMA, **dict.fromkeys(C._LIMITS, 100000)},
        destination=owner / "sources",
        version=7 if version == 2 else 6,
    )
    selection = owner / "conversion-selection.json"
    write(
        selection,
        {
            "schema": V.INTEGER_SELECTION_SCHEMA if version == 2 else V.SELECTION_SCHEMA,
            "source_record_sha256": V._digest(sources),
            "operator_schema_intake_sha256": intake.sha256,
            "semantic_basis_sha256": basis.source.sha256,
            "capture_checkout": capture,
            "capture_commit": commit,
            "mlir_opt": parser,
            "budget": conversion_budget(version=version),
        },
    )
    return V.prepare(
        schema_intake=intake,
        basis=basis,
        source_record=sources,
        numerical_semantics=numerics,
        selection=selection,
        destination=owner / "registered",
    )
