"""Canonical materialization identities do not weaken artifact authentication."""

from dataclasses import replace
import unittest

from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options, materialization_cache
from merlin.semantic_compiler.materialization import (
    MaterializationCache, canonical_request_record, materialization_key,
)
from merlin.semantic_compiler.model import (
    ConstantBinding, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps,
)


def request(prefix="a", *, source_identity="source"):
    tensor = TensorType((1,), "bf16", "raw-bit-pattern")
    left, right, constant, output = (prefix + name for name in ("left", "right", "constant", "output"))
    return KernelRequest(
        nodes=(SemanticNode(left, "input", (), tensor, effect="input"),
               SemanticNode(right, "input", (), tensor, effect="input"),
               SemanticNode(constant, "constant", (), tensor, effect="constant"),
               SemanticNode(output, "add", (left, right, constant), tensor,
                            attrs=(("numeric_semantics", "exact"), ("prov.location", prefix)),
                            index_maps=pointwise_identity_maps(1, 3))),
        outputs=(output,), output_storages=("external",),
        input_storages=((left, "external"), (right, "external")),
        target_identity="test-target", source_identity=source_identity,
        constants=(ConstantBinding(constant, "bf16-le-raw", "803f"),),
        physical_extents=((left, 1), (right, 1), (constant, 1)),
        boundary_layouts=((left, "contiguous"), (right, "contiguous"), (constant, "contiguous")),
        output_layouts=("contiguous",),
    )


def key(source, **kwargs):
    choices = dict(target_choice={"provider": "test", "engine": "native"},
                   options=CodegenOptions(), compilation_identity={"snapshot": "snapshot-1", "toolchain": "tools-1"})
    choices.update(kwargs)
    return materialization_key(source, **choices)


class MaterializationIdentityTest(unittest.TestCase):
    def test_alpha_ids_and_provenance_match_without_mutating_source(self):
        source = request()
        before = source.record()
        self.assertEqual(key(source), key(request("b")))
        self.assertEqual(source.record(), before)
        self.assertEqual(source.digest(), request().digest())

    def test_source_identity_requires_explicit_provenance_policy(self):
        left, right = request(source_identity="a" * 64), request(source_identity="b" * 64)
        self.assertNotEqual(key(left), key(right))
        self.assertEqual(key(left, source_identity_policy="provenance"), key(right, source_identity_policy="provenance"))
        with self.assertRaises(ValueError):
            canonical_request_record(left, source_identity_policy="automatic")

    def test_every_semantic_boundary_and_choice_is_retained(self):
        source = request()
        node = source.nodes[-1]
        variants = [
            replace(source, nodes=(*source.nodes[:-1], replace(node, inputs=tuple(reversed(node.inputs))))),
            replace(source, nodes=(*source.nodes[:-1], replace(node, attrs=(("numeric_semantics", "approximate"),)))),
            replace(source, nodes=(*source.nodes[:-1], replace(node, attrs=(*node.attrs, ("provenance", "semantic"))))),
            replace(source, nodes=(*source.nodes[:-1], replace(node, effect="pure", index_maps=()))),
            replace(source, input_storages=tuple(reversed(source.input_storages))),
            replace(source, input_storages=((source.nodes[0].id, "local"), source.input_storages[1])),
            replace(source, output_storages=("local",)),
            replace(source, target_identity="other-target"),
            replace(source, lowering_policy="diagnostic"),
            replace(source, constants=(replace(source.constants[0], data_hex="0040"),)),
            replace(source, physical_extents=tuple((name, 2) for name, _ in source.physical_extents)),
            replace(source, boundary_layouts=tuple((name, "transposed") for name, _ in source.boundary_layouts)),
            replace(source, output_layouts=("transposed",)),
            replace(source, nodes=(*source.nodes[:-1], replace(node, type=replace(node.type, numerical_policy="different")))),
        ]
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(key(source), key(variant))
        self.assertNotEqual(key(source), key(source, target_choice={"provider": "other"}))
        self.assertNotEqual(key(source), key(source, compilation_identity={"snapshot": "snapshot-2"}))
        for options in (CodegenOptions(reduction_unroll="2"), CodegenOptions(program_size_limit_bytes=1024),
                        CodegenOptions(io_memory_bytes=2048)):
            self.assertNotEqual(key(source), key(source, options=options))

    def test_attribute_string_values_are_not_renamed(self):
        source = request()
        node = replace(source.nodes[-1], attrs=(("semantic_reference", source.nodes[0].id),))
        source = replace(source, nodes=(*source.nodes[:-1], node))
        row = canonical_request_record(source)
        self.assertEqual(row["nodes"][-1]["attrs"]["semantic_reference"], "aleft")

    def test_hash_collision_and_namespace_are_guarded_by_full_identity(self):
        first = key(request())
        second = key(request(source_identity="different"))
        second = replace(second, digest=first.digest)
        storage = {}
        cache = MaterializationCache("provider-a", storage)
        cache.put(first, "first")
        self.assertIsNone(cache.get(second))
        cache.put(second, "second")
        self.assertEqual(cache.get(first), "first")
        self.assertEqual(cache.get(second), "second")
        cache.put(first, "replacement")
        self.assertEqual(cache.get(first), "replacement")
        self.assertIsNone(MaterializationCache("provider-b", storage).get(first))

    def test_invocation_storage_is_scoped_and_options_are_snapshotted(self):
        self.assertIsNone(materialization_cache())
        with codegen_options(CodegenOptions()):
            cache = MaterializationCache("provider", materialization_cache())
            cache.put(key(request()), "value")
            self.assertEqual(MaterializationCache("provider", materialization_cache()).get(key(request("b"))), "value")
        with codegen_options(CodegenOptions()):
            self.assertIsNone(MaterializationCache("provider", materialization_cache()).get(key(request())))
        self.assertIsNone(materialization_cache())


if __name__ == "__main__":
    unittest.main()
