"""Inactive ordered-unroll remainder preparation, complete actual module pairs.

Production replay is a host correspondence check, never Lean acceptance. Every
new query and witness remains uncompiled/unobserved under the shared native hold.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

from .lean_graph_pipeline_gate_projection import canonical, sha, source_ref, project_lean_sources

PROFILE = 'ordered_unroll_bulk_remainder_source_v1'
KIND = 'ordered_loop_unrolling'
DOMAIN = 'static_nonnegative_signed32or64_bulk_and_nonempty_tail_complete_outcome'
PREMISES = (
    'trusted complete xDSL parser/projection and exact source delivery binding',
    'one common frozen Scalar evaluator/registry/total pure primitive/partial external interpretation',
    'concrete finite indexIterations and checked signed control/literal bounds; no backend fidelity claim',
    'external successful-result arity contract only; no source success or whole-run equality premise',
    'private SSA stacks unobserved; full returns/fault/terminal world retained',
    'one changed top-level loop per two-argument entry; one/two carries; region-free body; explicit external calls only',
    'native fidelity, runtime value typing/poison, model relevance and review remain separate',
)


@dataclass(frozen=True)
class Selection:
    factor: int
    index_bits: int

    def validate(self):
        if type(self.factor) is not int or not 2 <= self.factor <= 4 or type(self.index_bits) is not int or self.index_bits not in (32,64):
            raise ValueError('explicit factor2..4 and signed32/64 selection required')


@dataclass(frozen=True)
class PreparedTail:
    profile: str
    kind: str
    domain: str
    premises: tuple[str, ...]
    python_version: str
    before: str
    after: str
    selection: Selection
    projection: str
    controls: str
    producer_diagnostics: str
    query: str
    roots: tuple[str, ...]
    sources: tuple

    @property
    def identity(self):
        return sha(canonical(asdict(self)))

    def describe(self):
        return {**asdict(self),'identity':self.identity,'query_sha256':sha(self.query),
                'query_status':'UNCOMPILED','outcome':'UNOBSERVED','qualification':'UNQUALIFIED',
                'authority':False,'compiler_pass_chain':'not_established','provider':'UNSELECTED_HELD',
                'public_route':'UNSUPPORTED/HELD'}


def _admit(module):
    from xdsl.dialects import builtin, func, scf
    from .lean_graph_input_admission import admit_module
    admit_module(module,calls='external-only')
    module.verify()
    for op in module.walk():
        if op.successors or (op.regions and type(op) not in (builtin.ModuleOp,func.FuncOp,scf.ForOp)):
            raise ValueError('unsupported CFG/region in complete tail domain')
        for value in (*op.operands,*op.results,*(a for r in op.regions for b in r.blocks for a in b.args)):
            t=value.type
            if type(t) is builtin.IndexType:
                continue
            if not (type(t) is builtin.IntegerType and t.signedness.data==builtin.Signedness.SIGNLESS and 1<=t.width.data<=256):
                raise ValueError('only scalar signless words and concrete index controls supported')


def query_roots(query):
    namespaces=[];roots=[]
    for line in query.splitlines():
        words=line.split()
        if not words:
            continue
        if words[0]=='namespace':
            namespaces.append(words[1])
        elif words[0]=='end':
            namespaces.pop()
        elif words[0]=='theorem':
            roots.append('.'.join([*namespaces,words[1]]))
    return tuple(roots)


def _sources():
    paths={Path(__file__),Path(__file__).with_name('lean_graph_pipeline_tail_emit.py'),
           Path(__file__).with_name('lean_graph_pipeline_tail_models.py')}
    for name,module in tuple(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if name.startswith(('merlin.','xdsl.')) and path and Path(path).suffix=='.py':
            paths.add(Path(path))
    return tuple(source_ref(p) for p in sorted(paths)) + project_lean_sources((
        'Merlin.GraphStatePipelineTail','Merlin.GraphStatePipelineLiterals',
        'Merlin.GraphStatePipelineScope','Merlin.GraphStatePipelineProgram','Merlin.GraphStateScalar'))


def prepare(before,after,selection):
    from merlin.transforms.certificates import parse_ir,ir_text
    from merlin.transforms.unrolling import unroll_ordered_loops
    from .lean_graph_input_admission import admit_text
    from .lean_graph_state_scalar_capture import project,metadata
    from .lean_graph_state_pipeline_capture import index_operation
    from .lean_graph_pipeline_tail_emit import emit
    if type(selection) is not Selection:
        raise ValueError('typed tail selection required; copied JSON grants no authority')
    selection.validate()
    for text in (before,after):
        if type(text) is not str or len(text.encode())>4*1024*1024:
            raise ValueError('bounded complete module text required')
        admit_text(text)
    source,target=parse_ir(before),parse_ir(after)
    _admit(source);_admit(target)
    clone=source.clone()
    diagnostic=unroll_ordered_loops(clone,(selection.factor,),index_bits=selection.index_bits)
    if not diagnostic or any(w.iterations<=w.factor or w.remainder==0 for w in diagnostic):
        raise ValueError('actual changed bulk+nonempty tail required; no-tail/inline remain separate')
    if ir_text(clone)!=ir_text(target):
        raise ValueError('actual complete result differs from pinned ordered producer output')
    pair=project(source,target,context=(source,target))
    descriptors={}
    for module in (source,target):
        for op in module.walk():
            control=index_operation(op)
            if control is not None:
                descriptors[canonical(metadata(op))]=control
    controls=[{'identity':i,**descriptors[canonical(row['descriptor'])]}
              for i,row in enumerate(pair['registry']) if canonical(row['descriptor']) in descriptors]
    # Index arithmetic already present in the source is not licensed by an
    # unroll bound. Its runtime operand bounds need a separate local proof.
    for op in source.walk():
        c=index_operation(op)
        if c is not None and c['kind']!='index-constant':
            raise ValueError('source index arithmetic needs separate range evidence')
    capture={'stages':[pair],'index_operations':controls,'factor':selection.factor,'index_bits':selection.index_bits}
    query=emit(capture)
    roots=('Merlin.GraphState.Pipeline.Tail.progression_append',
           'Merlin.GraphState.Pipeline.Tail.loop_append',
           'Merlin.GraphState.Pipeline.Tail.ordered_bulk_tail',
           'Merlin.GraphState.Pipeline.Scope.check_sound',
           'Merlin.GraphState.Pipeline.unroll_loop',
           'Merlin.GraphState.Scalar.loopEval_congr',
           'Merlin.GraphState.Pipeline.entries_cons',*query_roots(query))
    if len(roots)!=len(set(roots)) or roots[-1]!='merlin_unroll_tail_transform':
        raise ValueError('closed unique final/support root roster required')
    return PreparedTail(PROFILE,KIND,DOMAIN,PREMISES,sys.version,before,after,selection,canonical(pair),canonical(controls),
                        canonical([asdict(w) for w in diagnostic]),query,roots,_sources())


def request(profile,before,after,selection):
    if profile!=PROFILE or type(profile) is not str:
        raise ValueError('exact tail profile required; no no-tail/minimal profile fallback')
    p=prepare(before,after,selection)
    final='merlin_unroll_tail_transform'
    return {**p.describe(),'schema':'merlin.unroll_tail_future_request.v1',
            'final_root':final,'support_roots':p.roots[:-1],
            'exact_statement':'theorem '+final+p.query.split('theorem '+final,1)[1],
            'project_sources':[asdict(s) for s in p.sources if s.path.endswith('.lean')],
            'delivery_obligation':'same live complete source/result completion plus actual boundary selection; no report authority',
            'query_contract':'pure_declarations_separate_root_audit_v1'}


def matches(saved,profile,before,after,selection):
    return type(saved) is dict and canonical(saved)==canonical(request(profile,before,after,selection))


def compiler_eligible(_candidate):
    return False


def provider_arguments(*_args,**_kwargs):
    raise RuntimeError('tail provider unselected/held; new source review and native context required')
