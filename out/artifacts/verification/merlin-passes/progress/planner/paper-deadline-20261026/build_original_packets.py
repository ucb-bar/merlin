"""Deliver sealed ORIGINAL repository artifacts; never execute their contents.

No installed SDK is read, no qualification is rerun, no original is modified.
Archives preserve repository-relative names and immutable original bytes. This
does not relocate absolute-path receipts or prove archive/host authenticity.
"""
from pathlib import Path
import hashlib
import io
import json
import os
import stat
import tarfile


ROOT = Path.cwd().resolve()
DEST = ROOT / 'out/build/graph-paper-deadline/original-review-packets-v1'
DEST.mkdir(parents=True, exist_ok=False)
FAMILIES = {
    'shape': ('out/artifacts/verification/graph-shape-normalization/standalone-mode-v1/EVIDENCE_SEAL.json', 1001),
    'integer': ('out/artifacts/verification/graph-state-integer/review/sealed-v2/MANIFEST.json', 2250),
    'pipeline': ('out/artifacts/verification/graph-state-pipeline/review/sealed-v1/MANIFEST.json', 4001),
    'memory': ('out/artifacts/verification/graph-memory-scf/memory-scf-mode-v1/EVIDENCE_INDEX.json', 1061),
    'declared': ('out/artifacts/verification/graph-regions-declared/qualification-v1/SEAL.json', 209),
}


def save(name, value):
    (DEST / name).write_text(json.dumps(value, indent=2) + '\n')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(row):
    path = Path(row['path'])
    path = path if path.is_absolute() else ROOT / path
    relative = path.relative_to(ROOT)
    if '..' in relative.parts:
        raise ValueError('path traverses outside repository')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        a = os.fstat(stream.fileno())
        if not stat.S_ISREG(a.st_mode):
            raise ValueError('nonregular original packet file')
        data = stream.read()
        b = os.fstat(stream.fileno())
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    actual = {'path': str(path), 'bytes': len(data), 'sha256': sha(data)}
    if (any(getattr(a, k) != getattr(b, k) for k in fields)
            or actual['bytes'] != row['bytes'] or actual['sha256'] != row['sha256']):
        save('FIRST-ERROR.json', {'expected': row, 'actual': actual,
                                'reason': 'original byte/stat association differs', 'retry': False})
        raise ValueError('original packet association failure')
    return str(relative), data


def add(archive, name, data):
    member = tarfile.TarInfo(name)
    member.size = len(data)
    member.mode = 0o444
    archive.addfile(member, io.BytesIO(data))


reports = []
save('INTENT.json', {'schema': 'merlin.original-review-delivery.v1', 'families': FAMILIES,
                     'source_bits': 'original existing sealed/indexed refs',
                     'SDK_reads': 0, 'native_commands': 0, 'originals_modified': False})
try:
    for family, (index_name, expected_count) in FAMILIES.items():
        index_data = (ROOT / index_name).read_bytes()
        index_ref = {'path': index_name, 'sha256': sha(index_data), 'bytes': len(index_data)}
        document = json.loads(index_data)
        rows = document if isinstance(document, list) else document['files']
        rows = rows if isinstance(rows, list) else list(rows.values())
        if len(rows) != expected_count:
            raise ValueError('unexpected fixed original packet population')
        seen, expected = set(), {}
        archive_path = DEST / (family + '-originals.tar.gz')
        with tarfile.open(archive_path, 'w:gz', compresslevel=1) as archive:
            for row in rows:
                name, data = read(row)
                if name in seen:
                    raise ValueError('duplicate original artifact path')
                seen.add(name)
                expected[name] = (len(data), sha(data))
                add(archive, name, data)
            if index_name not in seen:
                expected[index_name] = (len(index_data), sha(index_data))
                add(archive, index_name, index_data)
        observed = {}
        # Independent archive readback; never extract over the shared checkout.
        with tarfile.open(archive_path, 'r:gz') as archive:
            for member in archive:
                if not member.isfile() or member.name in observed:
                    raise ValueError('nonregular or duplicate archive member')
                data = archive.extractfile(member).read()
                observed[member.name] = (len(data), sha(data))
        if observed != expected:
            raise ValueError('archive differs from exactly verified original members')
        report = {'family': family, 'original_index': index_ref,
                  'original_indexed_files': len(rows), 'archive_members': len(expected),
                  'original_indexed_bytes': sum(row['bytes'] for row in rows),
                  'archive': {'path': str(archive_path.relative_to(ROOT)),
                              'bytes': archive_path.stat().st_size,
                              'sha256': sha(archive_path.read_bytes())},
                  'all_original_pins_matched': True, 'archive_readback_exact': True,
                  'native_commands': 0, 'qualification_rerun': False}
        reports.append(report)
        save(family + '-DELIVERY.json', report)
        print(json.dumps(report), flush=True)
    save('RESULT.json', {'status': 'ORIGINAL_REPOSITORY_PACKETS_DELIVERED', 'families': reports,
                        'SDK_reads': 0, 'native_commands': 0, 'originals_modified': False,
                        'independent_reviewer_received': False,
                        'absolute_path_receipts_relocated': False,
                        'external_dependencies_delivered': False,
                        'claims_extended': False})
except Exception as error:
    if not (DEST / 'FIRST-ERROR.json').exists():
        save('FIRST-ERROR.json', {'error': type(error).__name__, 'reason': str(error), 'retry': False})
    raise
