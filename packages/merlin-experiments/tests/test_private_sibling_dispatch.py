"""Original member attribution and actual borrowed wrapper CFG obligations."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, StringAttr
from xdsl.parser import Parser

from merlin.targetgen.native_dispatch_accounting import _mandatory_command_blocks
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
from merlin.targetgen.owned_host_model_execution import _mandatory_source_commands
from merlin.targetgen.private_sibling_dispatch import borrowed_host_chain, mandatory_member_commands


def test_shared_head_kernel_keeps_distinct_original_call_groups():
    from copy import deepcopy

    from merlin.targetgen.private_sibling_dispatch import component_for_symbol, composition_members

    components = [
        {"physical_symbol": "reused", "physical_group": group, "matrix_invocations": 2,
         "members": [{"head": head, "source_group": index * 2 + head} for head in range(2)]}
        for index, group in enumerate((5, 9))
    ]
    composition = {"schema": "merlin.original_head_source_composition.v1",
                   "components": components, "ordinary_joined_groups": {},
                   "required_source_groups": [0, 1, 2, 3]}
    assigned = [{"symbol": "reused", "group": group} for group in (5, 9)]
    assert [m["source_group"] for m in composition_members(composition, assigned=assigned)] == [0, 1, 2, 3]
    for component in components:
        assert component_for_symbol(composition, "reused", group=component["physical_group"]) is component
    assert component_for_symbol(composition, "reused", group=99) is None
    with pytest.raises(ValueError, match="unambiguous"):
        component_for_symbol(composition, "reused")
    with pytest.raises(ValueError, match="per-slice invocation custody"):
        composition_members(composition, assigned=[*assigned, assigned[0]])
    omitted = deepcopy(composition)
    omitted["components"] = omitted["components"][:1]
    with pytest.raises(ValueError, match="omits or duplicates"):
        composition_members(omitted, assigned=assigned)
    duplicated = deepcopy(composition)
    duplicated["components"][1]["members"][0]["source_group"] = 0
    with pytest.raises(ValueError, match="omits or duplicates"):
        composition_members(duplicated, assigned=assigned)


def parsed(source):
    return Parser(make_llvm_context(), source).parse_module()


def test_each_original_member_needs_exact_nodes_and_unavoidable_actual_work():
    module = parsed("""builtin.module { llvm.func @kernel(%p: !llvm.ptr) {
      %x = llvm.load %p : !llvm.ptr -> i64
      %y = llvm.load %p : !llvm.ptr -> i64
      llvm.return
    }}""")
    function = module.body.block.first_op
    commands = [op for op in function.walk() if op.name == "llvm.load"]
    members = [{"source_group": i + 7, "source_region_id": f"r{i}", "source_node_ids": [f"n{i}"]}
               for i in range(2)]
    obligations = [{"source_group": i + 7, "region_id": f"r{i}", "source_op_index": 9 + 2 * i,
                    "semantic_family": "contraction"} for i in range(2)]
    for i, op in enumerate(commands):
        op.attributes.update({"prov.region_id": StringAttr(f"r{i}"),
                              "prov.source_node_ids": ArrayAttr([StringAttr(f"n{i}")])})

    def check(obligations=obligations):
        return mandatory_member_commands(function, {"contraction": set(commands)}, obligations, members,
                                         mandatory=_mandatory_source_commands)

    assert [row["source_op_index"] for row in check()] == [9, 11]
    transport = parsed("builtin.module { llvm.func @load() { llvm.return } }").body.block.first_op
    assert len(mandatory_member_commands(
        function, {"contraction": set(commands), "movement": {transport}}, obligations, members,
        mandatory=_mandatory_source_commands)) == 2
    commands[1].attributes["prov.source_node_ids"] = ArrayAttr([StringAttr("foreign")])
    with pytest.raises(ValueError, match="complete original member"):
        check()
    commands[1].attributes.update(commands[0].attributes)
    with pytest.raises(ValueError, match="operation 11"):
        check()
    # Even one required root may not borrow an optional sibling's commands.
    with pytest.raises(ValueError, match="operation 11"):
        check([obligations[1]])
    with pytest.raises(ValueError, match="supporting source"):
        check([{**obligations[0], "semantic_family": "elementwise_map"}])


def host_module(body):
    return parsed("""builtin.module {
      llvm.func @_mlir_ciface_shared_borrowed(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
      llvm.func @shared_borrowed(%p: !llvm.ptr) {
        llvm.call @_mlir_ciface_shared_borrowed(%p,%p,%p,%p,%p)
          : (!llvm.ptr,!llvm.ptr,!llvm.ptr,!llvm.ptr,!llvm.ptr) -> ()
        llvm.return
      }
      llvm.func @shared(%p: !llvm.ptr) {
        llvm.call @shared_borrowed(%p) : (!llvm.ptr) -> ()
        llvm.return
      }
      llvm.func @forward(%p: !llvm.ptr, %c: i1) { """ + body + "\n}}")


def test_actual_nested_wrapper_chain_is_unavoidable_once_and_complete():
    composition = {"borrowed_entry_symbol": "shared_borrowed", "host_entry_symbol": "shared",
                   "physical_symbol": "shared_device", "members": [{}, {}]}

    def check(body):
        module = host_module(body)
        return borrowed_host_chain(module, module.body.block.last_op, composition,
                                   mandatory=_mandatory_command_blocks)

    result = check("llvm.call @shared(%p) : (!llvm.ptr) -> ()\nllvm.return")
    assert result["chain"] == ["forward", "shared", "shared_borrowed", "_mlir_ciface_shared_borrowed"]
    with pytest.raises(ValueError, match="omits or repeats"):
        check("llvm.call @shared(%p) : (!llvm.ptr) -> ()\n"
              "llvm.call @shared(%p) : (!llvm.ptr) -> ()\nllvm.return")
    with pytest.raises(ValueError, match="bypassed"):
        check("llvm.cond_br %c, ^work, ^done\n^work:\n"
              "llvm.call @shared(%p) : (!llvm.ptr) -> ()\nllvm.br ^done\n^done:\nllvm.return")
    with pytest.raises(ValueError, match="repeat|bypassed"):
        check("llvm.br ^work\n^work:\nllvm.call @shared(%p) : (!llvm.ptr) -> ()\n"
              "llvm.cond_br %c, ^work, ^done\n^done:\nllvm.return")
    with pytest.raises(ValueError, match="source-owned"):
        check("llvm.call @shared_borrowed(%p) : (!llvm.ptr) -> ()\nllvm.return")


def test_borrowed_leaf_wrong_abi_or_defined_body_refuses():
    composition = {"borrowed_entry_symbol": "shared_borrowed", "host_entry_symbol": "shared",
                   "physical_symbol": "shared_device", "members": [{}, {}, {}]}
    module = host_module("llvm.call @shared(%p) : (!llvm.ptr) -> ()\nllvm.return")
    with pytest.raises(ValueError, match="complete pointer-only"):
        borrowed_host_chain(module, module.body.block.last_op, composition, mandatory=_mandatory_command_blocks)
