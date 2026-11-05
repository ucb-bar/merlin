"""Bind one generated private byte allocation to a checked staged Q1 source.

An independently regenerated complete module fixes every instruction and edge.
This reader proves only the allocation and its pointer-use closure. Original
primitive DMA bounds/noescape/completion and stock compiler custody remain
separate caller obligations; no host store or ordinary call is admitted.
"""
from dataclasses import dataclass
from hashlib import sha256
from math import prod

from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp, i8, i64

from merlin.compile.scheduling.dense_loaded_sum_composition import DenseLoadedSumComposition
from merlin.xdsl_dialects._common import text
from .fresh_tensor_writer import PrivateWorkspaceContract, validate_private_workspaces
from .host_builder import PTR


def _digest(module):
    return sha256(text(module, generic=True).encode()).hexdigest()


def _normalized(module):
    # Ordinary parsing may move inherent loop annotations from attributes to
    # properties. Normalize both actual and independently regenerated modules.
    from xdsl.parser import Parser
    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
    return text(Parser(make_llvm_context(), text(module)).parse_module(), generic=True)


@dataclass(frozen=True)
class PrivateWorkspaceOperand:
    """One actual original target operand, qualified by its target owner.

    Supplying this value does not establish the primitive's memory semantics.
    The reader checks current operation membership and the complete use closure.
    """
    operation: object
    operand: int
    direction: str
    qualification_identity: str


@dataclass(frozen=True)
class GeneratedPrivateWorkspace:
    module: ModuleOp
    reference: ModuleOp
    composition: DenseLoadedSumComposition
    workspace: PrivateWorkspaceContract
    entry_symbol: str
    target_operands: tuple[PrivateWorkspaceOperand, ...]
    source_sha256: str
    reference_sha256: str
    matched_uses: int

    def revalidate(self):
        if _digest(self.module) != self.source_sha256 or _digest(self.reference) != self.reference_sha256:
            raise ValueError('generated quantized workspace source changed')
        current = recognize_quantized_workspace(self.module, self.reference, self.composition,
            self.workspace, entry_symbol=self.entry_symbol, target_operands=self.target_operands)
        if current.record() != self.record():
            raise ValueError('generated quantized workspace binding changed')

    def record(self):
        return dict(schema='merlin.generated_quantized_workspace.v1', source_sha256=self.source_sha256,
            reference_sha256=self.reference_sha256, composition_identity=self.composition.identity,
            original_source_identity=self.composition.source.identity, entry_symbol=self.entry_symbol,
            private_bytes=self.workspace.bytes, alignment=self.workspace.alignment,
            target_pointer_uses=self.matched_uses,
            bound='complete ordinary regenerated instructions and CFG; one invocation-private byte allocation',
            source_lifetime='original Q1 cover, real handoff drain and original loaded-sum phase drains',
            scope='allocation/use closure only; original target effects, pointer bounds, compiler preservation and final object remain independently owned')


def recognize_quantized_workspace(module, reference, composition, workspace, *, entry_symbol, target_operands):
    """Recognize a regenerated staged-Q1 image without permitting host writes.

    ``reference`` is the original owner's independent normal source emission,
    never a saved verdict or executable substitute. Its exact equality is
    necessary in addition to the current source/consumer and allocation checks.
    Every opaque target pointer operand still requires original qualification.
    """
    if (type(module) is not ModuleOp or type(reference) is not ModuleOp
            or type(composition) is not DenseLoadedSumComposition
            or type(workspace) is not PrivateWorkspaceContract):
        raise ValueError('current original modules, checked staged source and private workspace required')
    composition.revalidate()
    validate_private_workspaces((workspace,))
    if (workspace.bytes != prod(composition.kernel.tensor('quantized_workspace').shape)
            or workspace.effect_witness_sha256 != composition.identity):
        raise ValueError('private workspace differs from the complete original Q1 source lifetime')
    module.verify(); reference.verify()
    if _normalized(module) != _normalized(reference):
        raise ValueError('actual quantized workspace image differs from the independent ordinary emission')
    functions = [op for op in module.body.block.ops if isinstance(op, llvm.FuncOp) and op.sym_name.data == entry_symbol]
    if len(functions) != 1 or not functions[0].body.blocks:
        raise ValueError('one current original quantized entry required')
    function = functions[0]; entry = function.body.blocks.first
    if function.function_type.output != llvm.LLVMVoidType() or any(a.type != PTR for a in entry.args):
        raise ValueError('private workspace requires the original pointer-only void entry')
    operations = tuple(function.walk())
    allocations = [op for op in operations if isinstance(op, llvm.AllocaOp)]
    if len(allocations) != 1:
        raise ValueError('exactly one generated Q1 allocation required')
    allocation = allocations[0]
    owner = allocation.size.owner
    if (allocation.parent is not entry or owner.name != 'llvm.mlir.constant'
            or owner.value.value.data != workspace.bytes or allocation.elem_type != i8
            or allocation.alignment.value.data < workspace.alignment
            or allocation.alignment.value.data % workspace.alignment):
        raise ValueError('generated Q1 allocation size, position, byte type or alignment changed')
    for op in operations:
        if (op.name in {'llvm.store','llvm.call','llvm.call_intrinsic','llvm.invoke','llvm.callbr'}
                or op.name.startswith('llvm.intr.')):
            raise ValueError('quantized workspace admits no host store or ordinary call')
    if type(target_operands) is not tuple or not target_operands:
        raise ValueError('complete original primitive pointer-operand roster required')
    declared = {}
    for use in target_operands:
        if (type(use) is not PrivateWorkspaceOperand or use.operation not in operations
                or not isinstance(use.operation, llvm.InlineAsmOp) or use.operation.results
                or type(use.operand) is not int or not 0 <= use.operand < len(use.operation.operands)
                or use.direction not in ('read','write') or not isinstance(use.qualification_identity,str)
                or not use.qualification_identity.strip() or (use.operation,use.operand) in declared):
            raise ValueError('private workspace needs exact current qualified primitive operands')
        declared[use.operation,use.operand] = use.direction
    seen, pending, actual = set(), [allocation.res], set()
    while pending:
        value = pending.pop()
        if value in seen:continue
        seen.add(value)
        for use in value.uses:
            op, index = use.operation, use.index
            if isinstance(op, llvm.GEPOp) and index == 0 and value.type == PTR and op.elem_type == i8:
                pending.extend(op.results)
            elif isinstance(op, llvm.PtrToIntOp) and index == 0 and value.type == PTR and op.output.type == i64:
                pending.extend(op.results)
            elif (op,index) in declared:
                actual.add((op,index))
            else:
                raise ValueError('generated private Q1 pointer has an unaccounted escape')
    if actual != set(declared) or set(declared.values()) != {'read','write'}:
        raise ValueError('private Q1 pointer roster is incomplete or names another allocation')
    composition.revalidate()
    return GeneratedPrivateWorkspace(module,reference,composition,workspace,entry_symbol,target_operands,
        _digest(module),_digest(reference),len(actual))
