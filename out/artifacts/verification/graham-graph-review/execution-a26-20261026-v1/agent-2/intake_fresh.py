"""One fresh official Lean ZIP data intake; stdlib only, no SDK execution."""
import hashlib
import json
import os
import pathlib
import stat
import sys
import time
import urllib.parse
import urllib.request
import zipfile

ROOT = pathlib.Path('/scratch/grammy/blast/merlin-pass-verification')
EVIDENCE = ROOT / 'out/artifacts/verification/graham-graph-review/execution-a26-20261026-v1/agent-2'
BUILD = ROOT / 'out/build/verification-a26-20261026-v1/fresh-sdk'
SIZE = 925899528
DIGEST = 'c8531faaa4a93d5a4dae0c742cbc1bd42a2993594d02f93a9d629ebf299fb2c8'
TOP = 'lean-4.35.0-linux'
STATE = {'phase': 'metadata', 'attempts': 1, 'bytes': 0, 'native_execution': False}


def save(name, value):
    with (EVIDENCE / name).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


class RestrictedRedirect(urllib.request.HTTPRedirectHandler):
    redirects = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        self.redirects += 1
        if target.scheme != 'https' or target.hostname not in ('github.com', 'release-assets.githubusercontent.com') or self.redirects > 3:
            raise ValueError('redirect outside the selected official delivery hosts')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def main():
    release = json.loads((EVIDENCE / 'release.json').read_bytes())
    tag = json.loads((EVIDENCE / 'tag.json').read_bytes())
    asset, = [a for a in release['assets'] if a['name'] == TOP + '.zip']
    if release['tag_name'] != 'v4.35.0' or tag['object']['sha'] != 'eb6cafb7290e8bd5f2937522d8d804f58f02bafc':
        raise ValueError('release/tag identity mismatch')
    if asset['size'] != SIZE or asset['digest'] != 'sha256:' + DIGEST:
        raise ValueError('fresh publisher metadata mismatch')
    save('OFFICIAL_IDENTITY.json', {'release': {k:release[k] for k in ('id','tag_name','html_url','published_at','immutable')}, 'tag':tag, 'asset':asset, 'checksum_provenance':'Fresh official GitHub release API over HTTPS; publisher/platform digest, not independent signature verification.'})
    destination = BUILD / (TOP + '.zip.part')
    STATE.update(phase='download', archive=str(destination))
    digest = hashlib.sha256()
    opener = urllib.request.build_opener(RestrictedRedirect())
    request = urllib.request.Request(asset['browser_download_url'], headers={'User-Agent':'Merlin-independent-Lean-reproduction','Accept-Encoding':'identity'})
    started = time.monotonic()
    with opener.open(request, timeout=30) as response, destination.open('xb') as output:
        save('DOWNLOAD_RESPONSE.json', {'url':response.geturl(),'status':response.status,'headers':dict(response.headers)})
        while True:
            if time.monotonic() - started > 700:
                raise TimeoutError('download data deadline')
            block = response.read(min(1048576, SIZE + 1 - STATE['bytes']))
            if not block:
                break
            STATE['bytes'] += len(block)
            if STATE['bytes'] > SIZE:
                raise ValueError('download exceeds exact selected size')
            digest.update(block)
            output.write(block)
        output.flush()
        os.fsync(output.fileno())
    STATE['sha256'] = digest.hexdigest()
    if STATE['bytes'] != SIZE or STATE['sha256'] != DIGEST:
        raise ValueError('download identity mismatch before ZIP processing')
    archive_path = BUILD / (TOP + '.zip')
    destination.rename(archive_path)
    save('ARCHIVE_IDENTITY.json', dict(STATE, archive=str(archive_path), identity='MATCH'))
    STATE['phase'] = 'member-roster'
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if not 0 < len(infos) <= 50000:
            raise ValueError('member count bound')
        rows, names, files, directories, total, rejected = [], set(), set(), set(), 0, []
        for info in infos:
            name = info.orig_filename
            clean = name[:-1] if name.endswith('/') else name
            p = pathlib.PurePosixPath(clean)
            mode = (info.external_attr >> 16) & 65535
            kind = stat.S_IFMT(mode)
            directory = name.endswith('/')
            row = {'path':name,'bytes':info.file_size,'compression':info.compress_type,'crc32':info.CRC,'mode':mode}
            rows.append(row)
            valid_path = '\\' not in clean and ':' not in clean and all(32 <= ord(c) != 127 for c in clean) and not p.is_absolute() and str(p) == clean and '..' not in p.parts and p.parts and p.parts[0] == TOP
            if not valid_path or clean in names:
                rejected.append({'path':name,'reason':'unsafe/duplicate member path'})
            names.add(clean)
            if info.flag_bits & (1 | 64) or info.compress_type not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED):
                rejected.append({'path':name,'reason':'encrypted/unsupported compression'})
            if info.create_system != 3 or kind != (stat.S_IFDIR if directory else stat.S_IFREG):
                rejected.append({'path':name,'reason':'nonregular/link/unknown member type'})
            if info.file_size < 0 or info.file_size > 1073741824:
                rejected.append({'path':name,'reason':'member byte cap'})
            if directory:
                directories.add(clean)
                if info.file_size or info.CRC:
                    rejected.append({'path':name,'reason':'directory has payload'})
            else:
                files.add(clean)
                total += info.file_size
            directories.update(str(parent) for parent in p.parents if str(parent) != '.')
        save('ZIP_ROSTER.json',{'archive_sha256':DIGEST,'count':len(rows),'declared_expanded_bytes':total,'members':rows,'rejected':rejected})
        if rejected or files & directories or total > 12 * 1024**3:
            raise ValueError('archive does not meet initial regular-member-only extraction policy; no relaxation')
        tree = BUILD / 'data'
        tree.mkdir(mode=0o700)
        STATE.update(phase='extract', expanded_bytes=0)
        manifests=[]
        for info, row in zip(infos,rows,strict=True):
            STATE['member'] = row['path']
            path = tree / pathlib.PurePosixPath(row['path'])
            if info.is_dir():
                path.mkdir(mode=0o700,parents=True,exist_ok=True)
                continue
            path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            member_hash=hashlib.sha256()
            count=0
            with archive.open(info) as source,path.open('xb') as output:
                while block:=source.read(1048576):
                    count+=len(block)
                    STATE['expanded_bytes']+=len(block)
                    if count>info.file_size or STATE['expanded_bytes']>12*1024**3:
                        raise ValueError('extraction byte cap')
                    member_hash.update(block)
                    output.write(block)
            if count!=info.file_size:
                raise ValueError('extracted size mismatch')
            path.chmod(0o700 if row['mode'] & 0o111 else 0o600)
            manifests.append({'path':str(path.relative_to(BUILD)),'bytes':count,'sha256':member_hash.hexdigest(),'mode':path.stat().st_mode})
        save('SDK_MEMBER_MANIFEST.json',{'archive_sha256':DIGEST,'expanded_bytes':STATE['expanded_bytes'],'regular_members':len(manifests),'members':manifests})
    STATE['phase']='DATA_INTAKE_COMPLETE'
    save('INTAKE_RESULT.json',dict(STATE, sdk=str(BUILD/'data'/TOP), old_runtime_hold='UNCHANGED'))
    print(json.dumps({'status':STATE['phase'],'sdk':str(BUILD/'data'/TOP),'expanded_bytes':STATE['expanded_bytes']}),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        save('FIRST_FAILURE.json',dict(STATE,error_type=type(error).__name__,error=str(error)))
        raise
