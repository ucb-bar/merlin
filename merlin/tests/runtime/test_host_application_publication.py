"""An explicitly selected host application never invents an accelerator image."""
from hashlib import sha256
import json

import pytest

from merlin.semantic_compiler.target_binding import verify_native_publication


def publication(tmp_path):
    (tmp_path / 'segments').mkdir()
    (tmp_path / 'host.c').write_text('int host(void) { return 0; }\n')
    return {'engine': 'merlin_native', 'request_digest': 'source', 'target_identity': 'target',
            'artifact_kind': 'host_application', 'segments': [],
            'regions': [{'placement': 'host', 'native_programs': []}],
            'stage_artifacts': {'host.c': sha256((tmp_path / 'host.c').read_bytes()).hexdigest()}}


def check(root, manifest):
    (root / 'manifest.json').write_text(json.dumps(manifest))
    verify_native_publication(root, manifest, engine='merlin_native', request_digest='source', target_identity='target')


def test_host_publication_authenticates_generated_source_without_fake_program(tmp_path):
    manifest = publication(tmp_path)
    check(tmp_path, manifest)
    (tmp_path / 'host.c').write_text('changed')
    with pytest.raises(ValueError, match='changed or is missing'):
        check(tmp_path, manifest)


@pytest.mark.parametrize('mutation', ['accelerator', 'image', 'extra_segment', 'root_image', 'empty_regions'])
def test_host_publication_rejects_device_and_empty_claims(tmp_path, mutation):
    manifest = publication(tmp_path)
    if mutation == 'accelerator':
        manifest['regions'][0]['placement'] = 'accelerator'
    elif mutation == 'image':
        manifest['segments'] = [{'path': 'segments/fake'}]
    elif mutation == 'extra_segment':
        (tmp_path / 'segments/unaccounted').mkdir()
    elif mutation == 'root_image':
        (tmp_path / 'program.bin').write_bytes(b'fake')
    elif mutation == 'empty_regions':
        manifest['regions'] = []
    with pytest.raises(ValueError, match='host-only application'):
        check(tmp_path, manifest)
