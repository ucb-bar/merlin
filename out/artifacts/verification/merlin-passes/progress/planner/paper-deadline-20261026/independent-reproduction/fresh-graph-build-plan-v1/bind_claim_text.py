"""Optional deterministic claim-source annotation after input assembly.

The selected generated queries use one uniquely named final theorem leaf. Text
extraction is descriptive only; the verifier retains a future exact kernel audit.
"""
import json
from pathlib import Path


def bind(packet):
    packet = Path(packet)
    path = packet / 'INPUTS.json'
    inputs = json.loads(path.read_text())
    for profile in inputs['profiles']:
        for case in profile['cases']:
            if case.get('status') == 'UNAVAILABLE':
                continue
            text = (packet / case['query']['file']).read_text()
            marker = 'theorem ' + case['final_root'].rsplit('.', 1)[-1]
            start = text.rfind(marker)
            end = text.find(':=', start)
            if start < 0 or end <= start:
                raise ValueError('final source declaration text unavailable: ' + case['id'])
            case['exact_final_declaration'] = text[start:end].rstrip()
            case['declaration_status'] = 'SOURCE_TEXT_ONLY_NOT_KERNEL_TYPE'
            request = json.loads((packet / case['request_metadata']['file']).read_text())
            prepared = json.loads((packet / case['prepared_metadata']['file']).read_text())
            prepared = prepared.get('prepared', prepared)
            if 'exact_final_statement' in request:
                case['request_exact_final_statement'] = request['exact_final_statement']
            if profile['id'].startswith('W-'):
                statement = case['exact_final_declaration']
                case['domain'] = {'status': 'SOURCE_BINDERS_NOT_KERNEL_CHECKED', 'exact_statement': statement,
                    'scope_description': 'typed kernel entry with CallersValid' if profile['id'] == 'W-whole-v1'
                    else 'every defined function conditioned on its exact EntryDomain; not unconditional all-entry acceptance'}
                case['relation'] = {'source_form': 'Epoch.graphOutcomes iff under same chosen partial interpretation',
                    'status': 'SOURCE_DESCRIPTION_ONLY', 'error_resource_native_divergence_equivalence_claimed': False}
                case['assumptions'] = {'exact_binders_and_relation': statement,
                    'status': 'SOURCE_TEXT_HUMAN_AND_KERNEL_REVIEW_REQUIRED'}
                for key in ('entry_roster', 'execution_witness_roots', 'execution_witness_refusals'):
                    case['model_metadata'][key] = prepared.get(key)

    path.write_text(json.dumps(inputs, indent=2) + '\n')


if __name__ == '__main__':
    bind(Path(__file__).resolve().parent)
