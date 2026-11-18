"""Platform artifacts bind explicit invocation bytes without linking a host ELF."""
from copy import deepcopy
from hashlib import sha256
import unittest

from merlin.compile.application_deployment import platform_files
from merlin.compile.application_handoff import bind_deployment_payloads


class PayloadTest(unittest.TestCase):
    def setUp(self):
        host = dict(address=0x08000000, bytes=65536, usable_bytes=61440,
                    measurement='declared_reservations', reserved_bytes=65536,
                    declared_reservations=[dict(name=n, bytes=b) for n, b in
                        [('code',36864),('data',8192),('context',4096),('stack',12288),('heap',4096)]])
        self.deployment = dict(provider=dict(host_allocation=host, sys_clk_freq_hz=30000000),
            host_allocation=host, window=dict(address=4096, bytes=64), window_required_bytes=16,
            inputs=[dict(path='inputs/input_000.bin', address=4100, bytes=2)],
            parameters=dict(path='handoff/parameters.bin',address=4108,bytes=4,
                            sha256=sha256(b'parm').hexdigest()),
            outputs=[dict(address=4104,bytes=2)])
        self.payloads={'inputs/input_000.bin':b'in','handoff/parameters.bin':b'parm'}

    def test_binding_padding_canary_parameters_and_clock(self):
        files=bind_deployment_payloads(self.deployment,self.payloads)
        self.assertEqual(files['handoff/window.bin'],b'\0'*4+b'in\0\0\xa5\xa5\0\0parm')
        self.assertEqual(files['handoff/platform-cflags.txt'],b'-DSYS_CLK_FREQ=30000000\n')
        self.assertIn(b'LENGTH = 61440',files['handoff/chip.ld'])
        self.assertIn(b'__stack_end <= 0x0800f000',files['handoff/chip.ld'])
        self.assertNotIn('handoff/window.bin',platform_files(self.deployment))

    def test_paged_payload_binding_refuses(self):
        self.deployment['paging'] = {'segments': [{'source_offset': 0, 'bytes': 4}]}
        self.assertNotIn('handoff/window.bin', platform_files(self.deployment))
        with self.assertRaisesRegex(RuntimeError,
                'deployment_paging_contract.*paged deployments bind through refill segments, not window.bin'):
            bind_deployment_payloads(self.deployment, self.payloads)

    def test_missing_inputs_bad_lengths_and_parameters_refuse(self):
        for mutation,reason in [({},'deployment_payload_missing'),
                ({**self.payloads,'inputs/input_000.bin':b'x'},'deployment_payload_contract'),
                ({**self.payloads,'handoff/parameters.bin':b'fail'},'deployment_payload_contract')]:
            with self.assertRaisesRegex(RuntimeError,reason):
                platform_files(self.deployment,mutation)

    def test_binding_forwards_only_explicit_platform_map(self):
        self.assertNotIn('handoff/vcs.ld',
                         bind_deployment_payloads(self.deployment, self.payloads))
        vcs_map = dict(host_address=0x80000000, window_address=0x81000000)
        files = bind_deployment_payloads(self.deployment, self.payloads,
                                         vcs_linker_map=vcs_map)
        self.assertEqual(files['handoff/vcs.ld'],
                         platform_files(self.deployment, vcs_linker_map=vcs_map)['handoff/vcs.ld'])
        self.assertEqual(files['handoff/window.bin'],
                         bind_deployment_payloads(self.deployment, self.payloads)['handoff/window.bin'])

    def test_host_over_limit_and_clock_are_named(self):
        self.deployment['provider']['host_allocation']['declared_reservations'][0]['bytes']=61441
        with self.assertRaisesRegex(RuntimeError,'deployment_host_overflow'):
            platform_files(self.deployment,self.payloads)
        self.setUp()
        self.deployment['provider']['sys_clk_freq_hz']=True
        with self.assertRaisesRegex(RuntimeError,'deployment_clock_contract'):
            platform_files(self.deployment,self.payloads)

    def test_window_overflow_and_aliases_refuse(self):
        self.deployment['window_required_bytes']=65
        with self.assertRaisesRegex(RuntimeError,'deployment_window_overflow'):
            platform_files(self.deployment,self.payloads)
        self.setUp()
        self.deployment['outputs'][0]['address']=4100
        with self.assertRaisesRegex(RuntimeError,'deployment_payload_contract'):
            platform_files(self.deployment,self.payloads)
