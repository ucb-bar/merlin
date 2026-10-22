from pathlib import Path
import re,json,hashlib
import argparse
p=argparse.ArgumentParser();p.add_argument('selected_gsim',type=Path);p.add_argument('firesim_reference',type=Path);p.add_argument('output',type=Path);args=p.parse_args();a=args.selected_gsim;b=args.firesim_reference
def sha(v):return hashlib.sha256(v).hexdigest()
def modules(p):
 data=p.read_bytes();line_number=1;previous_offset=0;matches=list(re.finditer(rb'^  (?:ext)?module ([A-Za-z0-9_]+)\s*:',data,re.M));result={}
 for i,m in enumerate(matches):
  name=m[1].decode();start=m.start();line_number+=data.count(b'\n',previous_offset,start);previous_offset=start;end=matches[i+1].start() if i+1<len(matches) else len(data);body=data[start:end];assert name not in result;result[name]={'bytes':body,'first_line':line_number,'dependencies':sorted(set(x.decode() for x in re.findall(rb'^\s+inst \S+ of (\S+)',body,re.M)))}
 return {'path':str(p),'sha256':sha(data),'bytes':len(data)},result
pa,ma=modules(a);pb,mb=modules(b)
roots=['StreamReaderCore','StreamReader','LoadController','Scratchpad','ICache'];seen=set();pending=roots[:]
while pending:
 n=pending.pop()
 if n in seen:continue
 assert n in ma and n in mb,n
 seen.add(n);pending.extend(ma[n]['dependencies']);pending.extend(mb[n]['dependencies'])
rows=[]
for n in sorted(seen):
 x,y=ma[n],mb[n];rows.append({'module':n,'selected_gsim_sha256':sha(x['bytes']),'firesim_sha256':sha(y['bytes']),'byte_identical':x['bytes']==y['bytes'],'selected_gsim_first_line':x['first_line'],'firesim_first_line':y['first_line'],'bytes':len(x['bytes']),'dependencies_identical':x['dependencies']==y['dependencies'],'dependencies':x['dependencies']})
j={'schema':'operator.selected_rtl_module_relation.v1','method':'Exact original module byte comparison including declarations, ports, statements and source annotations; no renaming or semantic normalization','selected_gsim':pa,'firesim_reference':pb,'roots':roots,'reachable_module_count':len(rows),'all_reachable_modules_byte_identical':all(r['byte_identical'] for r in rows),'modules':rows,'scope':'Only named modules and their instantiated child closure. Top-level clock/reset/interconnect/system equivalence is NOT inferred; complete FIR hashes remain distinct.','physical_engine_sha256':'76ddbfdc5b3beadef6b6b37768a08522122327057b92d71d3539a9529e4ea651','simulation_runs':0}
args.output.write_text(json.dumps(j,indent=2)+'\n');print(json.dumps({k:j[k] for k in ['reachable_module_count','all_reachable_modules_byte_identical']}));print([r['module'] for r in rows if not r['byte_identical']]);print(sha(args.output.read_bytes()))
