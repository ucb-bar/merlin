"""Static admission must reject every immutable check omitted by the fast host."""
import struct
import sys
from pathlib import Path
import unittest

sys.path[:0] = [str(Path(__file__).absolute().parents[1] / 'tools/tiny_full')]
from build_fast import validate_attention_initials


class InitialAdmissionTest(unittest.TestCase):
    def image(self, change=None):
        initial = bytearray(b'\xa5' * 256 + bytes(64) + b'\x5a' * 256)
        panel = bytearray(176)
        struct.pack_into('<QQQ', panel, 120, 256, 64, len(initial))
        struct.pack_into('<Q', panel, 160, 0x2000)
        segments = [dict(flags=4, address=0x1000, file_bytes=0x2000)]
        if change:
            change(panel, initial, segments)

        class Image:
            symbols = {'tiny_full_attention_panels': dict(address=0x1000, size=176),
                       'tiny_full_attention_initial_0': dict(address=0x2000, size=len(initial))}

            def bytes_at(self, address, size):
                data = panel if address == 0x1000 else initial
                return bytes(data[:size])

        image = Image()
        image.segments = segments
        return image, initial

    def check(self, change=None):
        image, initial = self.image(change)
        item = dict(attention_version=2, output_offset=256, output_extent_bytes=64,
                    total_dram_bytes=576, initial='initial')
        return validate_attention_initials(image, [item], lambda _: bytes(initial))

    def test_admitted(self):
        self.assertEqual(self.check(), 1)

    def test_corruptions(self):
        changes = {
            'output': lambda p, a, s: a.__setitem__(256, 1),
            'left_guard': lambda p, a, s: a.__setitem__(255, 0),
            'right_guard': lambda p, a, s: a.__setitem__(320, 0),
            'output_offset': lambda p, a, s: struct.pack_into('<Q', p, 120, 255),
            'extent': lambda p, a, s: struct.pack_into('<Q', p, 128, 63),
            'pointer': lambda p, a, s: struct.pack_into('<Q', p, 160, 0x2001),
            'writable': lambda p, a, s: s[0].__setitem__('flags', 6),
        }
        for field, change in changes.items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(change)

    def test_linked_payload_mismatch(self):
        image, initial = self.image()
        item = dict(attention_version=2, output_offset=256, output_extent_bytes=64,
                    total_dram_bytes=576, initial='initial')
        with self.assertRaises(ValueError):
            validate_attention_initials(image, [item], lambda _: b'\0' * len(initial))


if __name__ == '__main__':
    unittest.main()
