"""Static metadata specialization never reads tensor payload values."""

from targetgen_fixtures import run_m2m_script as _torch


def test_static_metadata_queries_use_descriptors_and_preserve_inputs():
    _torch('''
import torch
from merlin.capture.static_metadata import specialize_static_metadata
sources=[torch.arange(12.).reshape(3,4), torch.arange(24.).reshape(3,8)[:,1::2],
         torch.ones(1,4).expand(3,4), torch.empty(0,4)]
queries=[(torch.ops.aten.sym_numel.default,()), (torch.ops.aten.sym_size.int,(-1,)),
         (torch.ops.aten.sym_stride.int,(0,)), (torch.ops.aten.sym_storage_offset.default,()),
         (torch.ops.aten.sym_is_contiguous.default,())]
for query,args in queries:
 class Model(torch.nn.Module):
  def forward(self,x): return torch.ops.aten.scalar_tensor.default(query(x,*args))
 for x in sources:
  model=Model()
  folded,receipt=specialize_static_metadata(model,(x,))
  assert receipt["queries"] == [{"operator":str(query),"value":query(x,*args)}]
  assert receipt["inputs"] == [{"shape":list(x.shape),"stride":list(x.stride()),"storage_offset":x.storage_offset()}]
  assert torch.equal(folded(x),model(x))
  assert len([n for n in folded.graph.nodes if n.op=="placeholder"]) == 1
  assert not any(n.target==query for n in folded.graph.nodes)
class Mixed(torch.nn.Module):
 def forward(self,x): return x+torch.ops.aten.sym_numel.default(x)
original=Mixed()
assert specialize_static_metadata(original,(sources[0],)) == (original,None)
''')


def test_descriptor_queries_are_recognized_by_meta_evaluation_not_by_name():
    _torch('''
import torch
from merlin.capture.static_metadata import specialize_static_metadata
from merlin.targetgen.core_aten_capture import loader_source
x=torch.arange(24.).reshape(3,8)[:,1::2]
def model(call):
 class Model(torch.nn.Module):
  def forward(self,x): return torch.ops.aten.scalar_tensor.default(call(x))
 return Model()
# Descriptor-only results fold whatever the operator is called.
for call in (torch.ops.aten.dim.default, torch.ops.aten.numel.default, torch.ops.aten.is_contiguous.default,
             lambda x: torch.ops.aten.size.int(x,1)):
 folded,receipt=specialize_static_metadata(model(call),(x,))
 assert receipt and len(receipt["queries"])==1 and torch.equal(folded(x),model(call)(x)), call
# A data-dependent scalar, seeded randomness or a data result never folds.
for call in (torch.ops.aten._local_scalar_dense.default, torch.ops.aten.is_nonzero.default):
 m=model(call) if call is not torch.ops.aten.is_nonzero.default else model(lambda x: call(x[0,0]))
 assert specialize_static_metadata(m,(x,))[1] is None, call
class Seeded(torch.nn.Module):
 def forward(self,x): return torch.ops.aten.sym_numel.default(x)+torch.rand(())
assert specialize_static_metadata(Seeded(),(x,))[1] is None
# An input without a meta descriptor (a quantized tensor) keeps the ordinary path.
q=torch.quantize_per_tensor(x.contiguous(),0.5,0,torch.qint8)
assert specialize_static_metadata(model(torch.ops.aten.sym_numel.default),(q,))[1] is None
# The capture loader selects the fold from the operator schema, not a roster.
def flag(overload):
 tensor=dict(kind="tensor",dtype="float32",shape=[2],stride=[1],storage_offset=0,values=[1.,2.],requires_grad=False)
 namespace={}
 exec(loader_source(dict(overload=overload,arguments=dict(args=dict(kind="tuple",items=[tensor]),kwargs={}),expected=1)),namespace)
 return namespace["STATIC_METADATA_ONLY"]
assert flag("aten.dim.default") and flag("aten.sym_numel.default") and flag("aten.is_contiguous.default")
assert not flag("aten._local_scalar_dense.default") and not flag("aten.abs.default")
assert not flag("aten._foreach_zero_.default")
''')


def test_a_query_that_meta_tensors_answer_differently_does_not_fold():
    _torch('''
import torch
from merlin.capture.static_metadata import specialize_static_metadata
x=torch.ones(3); x.add_(1); x.add_(1)
assert torch.ops.aten._version.default(x)==2
class Model(torch.nn.Module):
 def forward(self,x):
  return torch.ops.aten.scalar_tensor.default(torch.ops.aten._version.default(x))
model=Model()
graph,proof=specialize_static_metadata(model,(x,))
assert graph is model and proof is None, proof
''')


def test_a_call_that_writes_an_input_does_not_fold_or_run_on_it():
    _torch('''
import torch
from merlin.capture.static_metadata import specialize_static_metadata
class Model(torch.nn.Module):
 def forward(self,x):
  torch.ops.aten._foreach_add_.Scalar([x],1.0)
  return torch.ops.aten.scalar_tensor.default(torch.ops.aten.sym_numel.default(x))
x=torch.ones(3)
assert specialize_static_metadata(Model(),(x,))[1] is None and torch.equal(x,torch.ones(3)), x
''')
