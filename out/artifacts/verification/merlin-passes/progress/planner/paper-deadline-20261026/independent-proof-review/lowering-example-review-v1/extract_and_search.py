"""Only AST literal extraction and bounded known verification metadata search.
No project import, IR parser, solver, tool/provider/binary/network execution.
"""
from pathlib import Path
import ast,json,hashlib,os
B=Path(__file__).resolve().parent;test=B/'source/merlin/tests/ir/test_verify_linalg_generic.py';tree=ast.parse(test.read_text());values={}
for n in tree.body:
 if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id in ['_SOURCE_TEXT','_OOT_INTERFACE_TEXT','_OOT_INTERFACE_SHA256']:values[n.targets[0].id]=ast.literal_eval(n.value)
D=B/'examples';D.mkdir();refs=[]
def save(name,text,origin,label):
 p=D/name;p.write_text(text);refs.append({'path':str(p.relative_to(B)),'sha256':hashlib.sha256(text.encode()).hexdigest(),'bytes':len(text.encode()),'origin':origin,'status':label})
save('source-original16.mlir',values['_SOURCE_TEXT'],'_SOURCE_TEXT literal in selected test','SAVED_TEST_EMBEDDED_OOT_DIAGNOSTIC_INPUT_NOT_NEW_PARSE')
save('oot-original16.mlir',values['_OOT_INTERFACE_TEXT'],'_OOT_INTERFACE_TEXT literal in selected test','SAVED_OOT_DIAGNOSTIC_OUTPUT_TEXT')
assert refs[-1]['sha256']==values['_OOT_INTERFACE_SHA256']
smallS=values['_SOURCE_TEXT'].replace('16x16','2x2');smallT=values['_OOT_INTERFACE_TEXT'].replace('16x16','2x2');save('source-small2.mlir',smallS,'_source(shape=2) string replacement before project parsing','DERIVED_SMALL_TEST_TEXT_NOT_EXECUTED');save('oot-small2.mlir',smallT,'_oot_text(shape=2) exact textual substitution','DERIVED_SMALL_TEST_TEXT_NOT_EXECUTED')
mutations=[]
for n in tree.body:
 if not isinstance(n,ast.FunctionDef):continue
 for dec in n.decorator_list:
  if isinstance(dec,ast.Call)and isinstance(dec.func,ast.Attribute)and dec.func.attr=='parametrize'and len(dec.args)>1:
   try:cases=ast.literal_eval(dec.args[1])
   except(ValueError,TypeError):continue
   if n.name=='test_generic_mutations_cannot_receive_false_proof':
    for i,(changes,status)in enumerate(cases):
     text=smallS
     for old,new in changes:assert old in text;text=text.replace(old,new,1)
     name=f'source-case-{i}-{status}.mlir';save(name,text,n.name,'TEST_EXPECTATION_ONLY_NOT_SOLVER_RESULT');mutations.append({'file':name,'expected_status':status,'changes':changes,'category':'source expected wrong-result vs unsupported syntax/body distinction','observed_status':None})
   elif n.name=='test_captured_oot_interface_mutations_never_receive_false_proof':
    for i,(old,new,status)in enumerate(cases):
     assert old in smallT;text=smallT.replace(old,new,1);name=f'oot-case-{i}-{status}.mlir';save(name,text,n.name,'TEST_EXPECTATION_ONLY_NOT_SOLVER_RESULT');mutations.append({'file':name,'expected_status':status,'changes':[[old,new]],'category':'target expected wrong-result vs unsupported grammar/contract distinction','observed_status':None})
(B/'EXAMPLE_INPUTS.json').write_text(json.dumps({'declared_oot_target_sha256':values['_OOT_INTERFACE_SHA256'],'files':refs,'mutations':mutations,'in_tree_interface_serialized_bytes':None,'command_buffer_serialized_bytes':None,'source_xdsl_generic_receipt_sha256':None,'solver_results_observed':False},indent=2)+'\n')
# Exact cited output and known existing verification-product namespaces only.
base=Path('out/artifacts/verification');roots=[base/n for n in ['roundtrip-dev','g1','g1-scan','g1-dce','g2','v1']];direct=[base/'transform-receipt.json',base/'detection_matrix.json',Path('out/runs/verification')];visited=[];candidates=[];matches=[];budget=200
for p in direct:visited.append({'path':str(p),'exists':p.exists(),'scope':'exact cited/known output path'})
for root in roots:
 visited.append({'path':str(root),'exists':root.exists(),'scope':'known verification product namespace, depth<=4'})
 if not root.is_dir():continue
 for directory,dirs,files in os.walk(root,followlinks=False):
  depth=len(Path(directory).relative_to(root).parts)
  if depth>=4:dirs[:]=[]
  dirs[:]=[n for n in dirs if not(Path(directory)/n).is_symlink()]
  for name in sorted(files):
   if len(candidates)>=budget:break
   p=Path(directory)/name
   if p.suffix!='.json'or p.is_symlink()or not any(t in name.lower()for t in ['receipt','report','manifest','detection','matrix']):continue
   if p.stat().st_size>2*1024**2:candidates.append({'path':str(p),'read':False,'reason':'metadata cap2MiB'});continue
   raw=p.read_bytes();item={'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'read':True};candidates.append(item)
   try:d=json.loads(raw)
   except(ValueError,UnicodeError):continue
   if isinstance(d,dict)and(d.get('schema')=='merlin.transform_verification.v1'or values['_OOT_INTERFACE_SHA256']in raw.decode('utf-8')):matches.append({**item,'schema':d.get('schema'),'status':d.get('status')})
  if len(candidates)>=budget:break
(B/'EVIDENCE_SEARCH.json').write_text(json.dumps({'scope':'cited exact paths plus five known verification namespaces; no build/SDK/binary/temp/global scan','max_candidate_json':budget,'max_depth':4,'max_bytes_each':2*1024**2,'roots':visited,'candidate_json':candidates,'matching_receipts':matches,'exact_example_checker_output':'UNAVAILABLE_IN_BOUNDED_RETAINED_SEARCH'if not matches else'NEEDS_SOURCE_TARGET_QUERY_TOOL_JOIN','global_absence_claimed':False,'tests_executed':False},indent=2)+'\n');print(json.dumps({'saved_example_texts':len(refs),'mutation_expectations':len(mutations),'candidate_metadata_read':len(candidates),'matching_receipts':matches}))
