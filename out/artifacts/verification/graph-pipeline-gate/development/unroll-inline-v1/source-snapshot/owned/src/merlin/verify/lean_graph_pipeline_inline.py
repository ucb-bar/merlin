"""Inactive empty/full-inline source preparation; no native authority or IO.

Complete actual producer association and independent typed projection are host
observations. The closed operational query must still be checked in Lean.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

from .lean_graph_pipeline_gate_projection import canonical, sha, source_ref, project_lean_sources
from .lean_graph_pipeline_tail import Selection, query_roots, _admit

PROFILE='ordered_unroll_empty_full_inline_source_v1'
KIND='ordered_loop_unrolling'
DOMAIN='static_nonnegative_signed32or64_count_le_factor_complete_outcome'
PREMISES=(
    'trusted complete xDSL parser/projection and exact source delivery binding',
    'same frozen Scalar evaluator/common descriptor registry/total pure primitive/partial external interpretation',
    'concrete finite indexIterations; checked signed index/literal/count facts; no backend fidelity premise discharged',
    'external local successful-result arity only; no source success, pure opaque call or body/whole-run equality premise',
    'one top-level source loop per two-argument changed entry; count0..factor2..4; one/two carries; region-free body and suffix',
    'complete returns/faults/terminal world observed; private SSA stack unobserved; native/poison/model/review separate',
)


@dataclass(frozen=True)
class PreparedInline:
    profile: str
    kind: str
    domain: str
    premises: tuple[str,...]
    python_version: str
    before: str
    after: str
    selection: Selection
    projection: str
    controls: str
    producer_diagnostics: str
    field_facts: str
    query: str
    roots: tuple[str,...]
    sources: tuple

    @property
    def identity(self):return sha(canonical(asdict(self)))

    def describe(self):
        return {**asdict(self),'identity':self.identity,'query_sha256':sha(self.query),
                'query_status':'UNCOMPILED','outcome':'UNOBSERVED','qualification':'UNQUALIFIED',
                'authority':False,'provider':'UNSELECTED_HELD','public_route':'UNSUPPORTED/HELD',
                'compiler_pass_chain':'not_established'}


def _sources():
    paths={Path(__file__),Path(__file__).with_name('lean_graph_pipeline_inline_emit.py'),
           Path(__file__).with_name('lean_graph_pipeline_inline_models.py')}
    for name,module in tuple(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if name.startswith(('merlin.','xdsl.')) and path and Path(path).suffix=='.py':paths.add(Path(path))
    return tuple(source_ref(p) for p in sorted(paths))+project_lean_sources((
        'Merlin.GraphStatePipelineInline','Merlin.GraphStatePipelineIndex',
        'Merlin.GraphStatePipelineLiterals','Merlin.GraphStatePipelineProgram','Merlin.GraphStateScalar'))


def _field_facts(source,target,selection):
    from xdsl.dialects import func,scf
    from merlin.transforms.normalization import constant_integer
    rows=[]
    targets={op.sym_name.data:op for op in target.body.block.ops if type(op) is func.FuncOp}
    for f in source.body.block.ops:
        if type(f) is not func.FuncOp or not f.body.blocks:continue
        loops=[op for op in f.body.block.ops if type(op) is scf.ForOp]
        if not loops:continue
        if len(loops)!=1:raise ValueError('one top-level loop per changed entry required')
        loop=loops[0];lower,upper,step=(constant_integer(v) for v in (loop.lb,loop.ub,loop.step))
        if any(type(v) is not int for v in (lower,upper,step)) or lower<0 or upper<lower or step<1:
            raise ValueError('static nonnegative/ordered bounds and positive stride required')
        count=(upper-lower+step-1)//step
        if count>selection.factor:raise ValueError('only empty/full-inline mode admitted')
        before_ops=list(f.body.block.ops);pos=before_ops.index(loop)
        target_ops=list(targets[f.sym_name.data].body.block.ops)
        body_ops=list(loop.body.block.ops)[:-1]
        if any(op.regions or op.successors for op in body_ops):raise ValueError('region/CFG inside source body unsupported')
        lanes=[]
        for lane in range(count):
            offset=pos+lane*(1+len(body_ops))
            if offset>=len(target_ops):raise ValueError('missing actual inline lane')
            op=target_ops[offset]
            if len(op.results)!=1 or constant_integer(op.results[0])!=lower+lane*step:
                raise ValueError('actual lane IV constant differs from counted progression')
            lanes.append({'ordinal':lane,'actual_index':constant_integer(op.results[0]),'target_operation':offset})
        rows.append({'entry':f.sym_name.data,'source_controls':[lower,upper,step],'count':count,
                     'factor':selection.factor,'mode':'empty' if count==0 else 'full-inline',
                     'lane_constants':lanes,'carry_types':[str(v.type) for v in loop.iter_args],
                     'source_body_operations':len(body_ops),'target_lane_operations':count*(1+len(body_ops)),
                     'source_suffix_operations':len(before_ops)-pos-1,
                     'target_suffix_position':pos+count*(1+len(body_ops)),
                     'producer_witness_used_as_proof':False})
    if not rows:raise ValueError('an actual source loop is required')
    return rows


def prepare(before,after,selection):
    from merlin.transforms.certificates import parse_ir,ir_text
    from merlin.transforms.unrolling import unroll_ordered_loops
    from .lean_graph_input_admission import admit_text
    from .lean_graph_state_scalar_capture import project,metadata
    from .lean_graph_state_pipeline_capture import index_operation
    from .lean_graph_pipeline_inline_emit import emit
    from xdsl.dialects import arith
    if type(selection) is not Selection:raise ValueError('typed selected inline profile required; no JSON authority')
    selection.validate()
    for text in (before,after):
        if type(text) is not str or len(text.encode())>4*1024*1024:raise ValueError('complete bounded module text required')
        admit_text(text)
    source,target=parse_ir(before),parse_ir(after)
    for module in (source,target):
        _admit(module)
        for op in module.walk():
            flags=op.properties.get('overflowFlags')
            if flags is not None and flags!=arith.IntegerOverflowAttr('none'):
                raise ValueError('flagged arithmetic outside this inline source profile')
            if op.properties.get('fastmath') is not None:
                raise ValueError('fast-math outside scalar integer inline profile')
    facts=_field_facts(source,target,selection)
    clone=source.clone();diagnostics=unroll_ordered_loops(clone,(selection.factor,),index_bits=selection.index_bits)
    if not diagnostics or any(w.iterations>w.factor for w in diagnostics):raise ValueError('actual empty/full-inline change required')
    if ir_text(clone)!=ir_text(target):raise ValueError('complete actual inline result differs from selected pinned producer')
    pair=project(source,target,context=(source,target));descriptors={}
    for module in (source,target):
        for op in module.walk():
            control=index_operation(op)
            if control is not None:descriptors[canonical(metadata(op))]=control
    for op in source.walk():
        control=index_operation(op)
        if control is not None and control['kind']!='index-constant':
            raise ValueError('source index arithmetic needs separate runtime range evidence')
    controls=[{'identity':i,**descriptors[canonical(r['descriptor'])]} for i,r in enumerate(pair['registry']) if canonical(r['descriptor']) in descriptors]
    query=emit({'stages':[pair],'index_operations':controls,'factor':selection.factor,'index_bits':selection.index_bits})
    roots=('Merlin.GraphState.Pipeline.Inline.empty_no_body',
           'Merlin.GraphState.Pipeline.Inline.counted_continuation',
           'Merlin.GraphState.Pipeline.Inline.failed_continuation',
           'Merlin.GraphState.Pipeline.indexIterations_count',
           'Merlin.GraphState.Pipeline.entries_cons',*query_roots(query))
    if len(roots)!=len(set(roots)) or roots[-1]!='merlin_unroll_inline_transform':raise ValueError('exact final/support roster required')
    return PreparedInline(PROFILE,KIND,DOMAIN,PREMISES,sys.version,before,after,selection,canonical(pair),canonical(controls),
                          canonical([asdict(w) for w in diagnostics]),canonical(facts),query,roots,_sources())


def request(profile,before,after,selection):
    if type(profile) is not str or profile!=PROFILE:raise ValueError('exact inline profile required; no other profile fallback')
    p=prepare(before,after,selection);final='merlin_unroll_inline_transform'
    return {**p.describe(),'schema':'merlin.unroll_inline_future_request.v1',
            'final_root':final,'support_roots':p.roots[:-1],
            'exact_statement':'theorem '+final+p.query.split('theorem '+final,1)[1],
            'project_sources':[asdict(s) for s in p.sources if s.path.endswith('.lean')],
            'query_contract':'pure_declarations_separate_root_audit_v1',
            'delivery_obligation':'same live complete source/result completion plus actual boundary selection; no saved report authority'}


def matches(saved,profile,before,after,selection):
    return type(saved) is dict and canonical(saved)==canonical(request(profile,before,after,selection))


def compiler_eligible(_candidate):return False


def provider_arguments(*_args,**_kwargs):
    raise RuntimeError('inline provider held; source review and justified new native context required')
