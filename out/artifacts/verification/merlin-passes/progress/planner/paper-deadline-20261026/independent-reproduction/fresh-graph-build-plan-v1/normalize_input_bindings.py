"""Normalize selected metadata refs, without executing provider/controller code."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_plan import Reader, decode, project_records

out = Path(__file__).resolve().parent
reader = Reader(out)
inputs = decode((out / 'INPUTS.json').read_bytes())
inv = decode((out / 'SOURCE_DATA_INVENTORY.json').read_bytes())
refs = {r['ref']['file']: r['ref'] for r in inv['all_read_source_data_associations']}
inputs['packet_data_refs'] = [refs[k] for k in sorted(refs)]
for profile in inputs['profiles']:
    if profile['id'].startswith('Q-'):
        case = profile['cases'][0]
        request = reader.document(case['request_metadata'])
        field = 'sources' if profile['id'] == 'Q-service-CSE' else 'project_sources'
        declared = project_records(request[field])
        profile['catalog'] = [r for r in profile['catalog'] if r['module'] in declared]
        case['project_sources_binding'] = {'path': [field]}
    else:
        packet = inv['packets'][profile['packet']]
        profile['project_inventory'] = packet['metadata_files']['PROJECT-LEAN-SOURCES.json']
(out / 'INPUTS.json').write_text(json.dumps(inputs, indent=2) + '\n')
print(json.dumps({p['id']: len({r['module'] for r in p['catalog']}) for p in inputs['profiles']}))
