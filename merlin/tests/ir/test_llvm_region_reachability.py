"""One ordinary traversal preserves graph state and legacy custom dispatch."""

import sys

import pytest
from test_llvm_region_boundary import source

from merlin.llvmlower import llvm_region_boundary as B


def construct(cls=B.LLVMRegionBoundary, text=None, bits=64):
    return cls(source(bits) if text is None else text, function_name="work", argument_types=("ptr", f"i{bits}"))


def profiled_construct(cls=B.LLVMRegionBoundary, text=None):
    calls = []

    def profile(frame, event, arg):
        if event == "call" and frame.f_code is B._DEFAULT_REACHABLE.__code__:
            calls.append(frame.f_locals["target"])

    prior = sys.getprofile()
    sys.setprofile(profile)
    try:
        graph = construct(cls, text)
    finally:
        sys.setprofile(prior)
    return graph, calls


@pytest.mark.parametrize("bits,count", [(16, 3), (32, 19), (64, 7)])
def test_ordinary_and_inherited_graph_fields_match_legacy_dispatch(bits, count):
    class LegacyDispatch(B.LLVMRegionBoundary):
        def reachable(self, *args, **kwargs):
            return super().reachable(*args, **kwargs)

    class SelectedModule(B.LLVMRegionBoundary):
        def __init__(self, text, **kwargs):
            super().__init__(text, **kwargs)

    text = source(bits, count)
    actual = construct(text=text, bits=bits)
    inherited = construct(SelectedModule, text, bits)
    legacy = construct(LegacyDispatch, text, bits)
    assert actual.__dict__ == inherited.__dict__ == legacy.__dict__
    for start in actual.blocks:
        for target in actual.blocks:
            assert actual.reachable(start, target) == legacy.reachable(start, target)
            assert actual.dominates(start, target) == legacy.dominates(start, target)


def test_default_and_inherited_paths_do_not_repeat_public_reachability():
    class SelectedModule(B.LLVMRegionBoundary):
        pass

    for cls in (B.LLVMRegionBoundary, SelectedModule):
        graph, calls = profiled_construct(cls)
        assert len(graph.blocks) == 8 and calls == []


def test_unreachable_graph_has_same_refusal_under_legacy_reference():
    class LegacyDispatch(B.LLVMRegionBoundary):
        def reachable(self, *args, **kwargs):
            return super().reachable(*args, **kwargs)

    text = source().replace("}\n", "dead:\n  ret void\n}\n")
    for cls in (B.LLVMRegionBoundary, LegacyDispatch):
        with pytest.raises(ValueError, match="^unreachable LLVM CFG block$"):
            construct(cls, text)


def test_inherited_override_observes_original_order_and_partial_state():
    events = []

    class Middle(B.LLVMRegionBoundary):
        def reachable(self, start, target, avoid=None):
            events.append((start, target, hasattr(self, "context"), hasattr(self, "cfg_encoded")))
            return target != "guard"

    class SelectedModule(Middle):
        pass

    with pytest.raises(ValueError, match="unreachable"):
        construct(SelectedModule)
    assert events == [("entry", target, False, False) for target in ("entry", "header", "guard")]


def test_instance_override_and_replacement_during_first_callback():
    events = []

    class SelectedModule(B.LLVMRegionBoundary):
        def __init__(self, text, **kwargs):
            def second(start, target):
                events.append(("second", start, target))
                return False

            def first(start, target):
                events.append(("first", start, target))
                self.reachable = second
                return True

            self.reachable = first
            super().__init__(text, **kwargs)

    with pytest.raises(ValueError, match="unreachable"):
        construct(SelectedModule)
    assert events == [("first", "entry", "entry"), ("second", "entry", "header")]


def test_property_dispatch_is_not_evaluated_early_or_skipped():
    events = []

    class SelectedModule(B.LLVMRegionBoundary):
        @property
        def reachable(self):
            events.append(("lookup", hasattr(self, "context")))

            def callback(start, target):
                events.append((start, target))
                return target != "header"

            return callback

    with pytest.raises(ValueError, match="unreachable"):
        construct(SelectedModule)
    assert events == [("lookup", False), ("entry", "entry"), ("lookup", False), ("entry", "header")]


def test_custom_attribute_resolution_uses_legacy_callbacks():
    events = []

    class SelectedModule(B.LLVMRegionBoundary):
        def __getattribute__(self, name):
            if name == "reachable":

                def callback(start, target):
                    events.append((start, target))
                    return target != "header"

                return callback
            return super().__getattribute__(name)

    with pytest.raises(ValueError, match="unreachable"):
        construct(SelectedModule)
    assert events == [("entry", "entry"), ("entry", "header")]


@pytest.mark.parametrize("name", ["entry", "blocks", "edges"])
def test_field_descriptor_preserves_legacy_traversals(name):
    def get(self):
        return self.__dict__["retained_" + name]

    def put(self, value):
        self.__dict__["retained_" + name] = value

    cls = type("SelectedModule", (B.LLVMRegionBoundary,), {name: property(get, put)})
    graph, calls = profiled_construct(cls)
    assert calls == list(graph.blocks)


def test_custom_setattr_and_metaclass_conservatively_retain_legacy_path():
    class SelectedModule(B.LLVMRegionBoundary):
        def __setattr__(self, name, value):
            super().__setattr__(name, value)

    class Meta(type):
        pass

    class MetaModule(B.LLVMRegionBoundary, metaclass=Meta):
        pass

    for cls in (SelectedModule, MetaModule):
        graph, calls = profiled_construct(cls)
        assert calls == list(graph.blocks)


def test_rebound_base_method_does_not_get_treated_as_original(monkeypatch):
    events = []

    def replacement(self, start, target, avoid=None):
        events.append((start, target))
        return target != "header"

    monkeypatch.setattr(B.LLVMRegionBoundary, "reachable", replacement)
    with pytest.raises(ValueError, match="unreachable"):
        construct()
    assert events == [("entry", "entry"), ("entry", "header")]


def test_instance_override_can_preserve_original_unreachable_acceptance():
    class SelectedModule(B.LLVMRegionBoundary):
        def __init__(self, text, **kwargs):
            self.reachable = lambda *args, **kwargs: True
            super().__init__(text, **kwargs)

    text = source().replace("}\n", "dead:\n  ret void\n}\n")
    graph = construct(SelectedModule, text)
    assert "dead" in graph.blocks


@pytest.mark.parametrize("attribute", ["__code__", "__defaults__"])
def test_modified_original_function_state_keeps_legacy_behavior(monkeypatch, attribute):
    if attribute == "__code__":

        def replacement(self, start, target, avoid=None):
            return False

        value = replacement.__code__
    else:
        value = ("entry",)
    monkeypatch.setattr(B._DEFAULT_REACHABLE, attribute, value)
    with pytest.raises(ValueError, match="unreachable"):
        construct()


def test_custom_instance_dict_is_not_read_by_dispatch_guard():
    class SelectedModule(B.LLVMRegionBoundary):
        @property
        def __dict__(self):
            raise AssertionError("legacy constructor does not read an instance dictionary")

    graph, calls = profiled_construct(SelectedModule)
    assert calls == list(graph.blocks)


def test_unknown_reachable_descriptor_is_resolved_only_on_instances():
    events = []

    class Descriptor:
        def __get__(self, instance, owner):
            assert instance is not None

            def callback(start, target):
                events.append((start, target))
                return target != "header"

            return callback

    class SelectedModule(B.LLVMRegionBoundary):
        reachable = Descriptor()

    with pytest.raises(ValueError, match="unreachable"):
        construct(SelectedModule)
    assert events == [("entry", "entry"), ("entry", "header")]


def test_absent_class_method_retains_dynamic_fallback(monkeypatch):
    events = []

    class SelectedModule(B.LLVMRegionBoundary):
        def __getattr__(self, name):
            if name != "reachable":
                raise AttributeError(name)

            def callback(start, target):
                events.append((start, target))
                return target != "header"

            return callback

    monkeypatch.delattr(B.LLVMRegionBoundary, "reachable")
    with pytest.raises(ValueError, match="unreachable"):
        construct(SelectedModule)
    assert events == [("entry", "entry"), ("entry", "header")]
