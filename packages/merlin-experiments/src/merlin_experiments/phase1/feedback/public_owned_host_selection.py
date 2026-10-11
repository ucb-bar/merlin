"""Protected, answer-free build inputs for Merlin-owned public host compilation."""
from pathlib import Path
import hashlib,json,os,yaml
from .private_source_freeze import pinned_file

ENV='MERLIN_MODEL_OWNED_HOST_BUILD_SELECTION'

def prepare(spec, declaration, *, target, output):
    spec=Path(spec)
    document=yaml.safe_load(spec.read_bytes())
    if document.get('target') != target or not document.get('models'):
        raise ValueError('owned host selection requires matching private launch configuration')
    rows=document['models']; selected=None; arenas=[]
    for row in rows:
        board=row.get('board'); arena=row.get('arena_mb')
        if not isinstance(board,str) or not board or type(arena) is not int or arena<=0:
            raise ValueError('owned host selection requires explicit board and positive arena_mb')
        current={'board':board}
        for key,source in [('board_catalog','board_catalog'),('dts','host_dts'),('rtl_facts','rtl_facts')]:
            path=pinned_file(spec,row.get(source),row.get(source+'_sha256'))
            current[key]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        if selected is not None and current!=selected:
            raise ValueError('public host inputs require one common selected catalog, DTS, board and RTL extraction')
        selected=current;arenas.append(arena)
    declaration=Path(declaration)
    if declaration.is_symlink() or not declaration.is_file():
        raise ValueError('owned host selection requires exact operator instruction declaration')
    selected['instruction_declaration']={'path':str(declaration.absolute()),'sha256':hashlib.sha256(declaration.read_bytes()).hexdigest()}
    selected['arena_mb']=max(arenas)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(selected,sort_keys=True,indent=2)+'\n');output.chmod(0o600)
    return {'path':str(output.absolute()),'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
