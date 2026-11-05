"""Explicit producer metadata and typed FP8 views for serialized Merlin capture."""
import ctypes as ct

import kernel_library  # noqa: E402,F401
from smol_fusion_program import stage_panel
from smol_fusion_fp8 import stage_fc1_fp8
import numpy as np

SETS = ('none', 'vision-bias', 'vision-bias-gelu', 'fc1-bias-gelu', 'connector', 'vision-bias-gelu-connector', 'fc1-bias-gelu-fp8')


class FusionPanel(ct.Structure):
    _fields_ = [('extra', ct.c_void_p), ('extra_offset', ct.c_size_t),
                ('extra_bytes', ct.c_size_t), ('operations', ct.c_uint), ('output_scale', ct.c_int)]


def operations_for_call(mode, call):
    if mode not in SETS:
        raise ValueError('unsupported Smol fusion set')
    if call.site != 7:
        return ()
    if call.role == 7:
        return ('connector_scale',) if mode in ('connector', 'vision-bias-gelu-connector') else ()
    if not 0 <= call.role <= 6:
        return ()
    if mode in ('fc1-bias-gelu', 'fc1-bias-gelu-fp8'):
        return ('bias', 'gelu_tanh') if call.role == 5 else ()
    if mode in ('vision-bias', 'vision-bias-gelu', 'vision-bias-gelu-connector'):
        return ('bias', 'gelu_tanh') if call.role == 5 and mode != 'vision-bias' else ('bias',)
    return ()


def baseline_geometry(row, scale):
    m, k, n = row['physical_mkn']
    result = dict(row, a_scale=scale, output_rescale_exponent=scale + row['b_scale'],
                  recipe=row['program_policy'])
    old_size = row['output_offset'] + 2 * m * n + 256
    result.update(dram_bytes=old_size, total_dram_bytes=old_size)
    return result


class FusionCapture:
    """Configure immutable output geometry and constants before serialized execution."""
    def __init__(self, library, raw, panels, kernels, calls, mode):
        if mode not in SETS or mode == 'none' or ct.sizeof(FusionPanel) != 32:
            raise ValueError('explicit enabled fusion profile and current ABI required')
        self.library, self.raw, self.mode = library, raw, mode
        self.panels = (FusionPanel * len(panels))()
        self.calls = calls
        self.static = {}
        self.buffers, self.prepared = [], {}
        library.smol_capture_fusion_size.restype = ct.c_size_t
        if library.smol_capture_fusion_size() != ct.sizeof(FusionPanel):
            raise ValueError('fusion C/ctypes ABI differs')
        library.smol_fusion_configure.argtypes = [ct.c_void_p, ct.POINTER(FusionPanel), ct.c_size_t]
        library.smol_fusion_configure.restype = ct.c_int
        library.smol_fusion_input_view.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.POINTER(ct.c_int)]
        library.smol_fusion_input_view.restype = ct.c_int
        # Constants, output geometry, and captured consumer scales are fixed before
        # configuration. Only canonical program/rescale fields change at callbacks.
        for index, row in enumerate(kernels):
            panel = panels[index]
            call = calls[panel.call]
            if operations_for_call(mode, call):
                self.prepare(index, row, call.activation_scale, call, panel)
        if library.smol_fusion_configure(ct.addressof(raw), self.panels, len(self.panels)):
            raise ValueError('initial complete fusion metadata rejected')

    def prepare(self, index, row, scale, call, panel):
        operations = operations_for_call(self.mode, call)
        if not operations:
            return None
        normalized = baseline_geometry(row, scale)
        m, k, n = normalized['physical_mkn']
        bias = None
        if 'bias' in operations:
            if call.input_dtype != 0 or call.output_dtype != 0 or call.bias_dtype != 0 or not call.bias:
                raise ValueError('fusion bias requires actual BF16 vision operands')
            bias = ct.string_at(call.bias, n * 2)
        offset = normalized['output_offset']
        # Program and constants do not depend on A/B contents; actual C staging is
        # independently rechecked with real bytes in verify_stage before execution.
        initial = bytes(offset - 256) + b'\xa5' * 256 + b'\xcd' * (2 * m * n) + b'Z' * 256
        fp8_scale = None
        if self.mode == 'fc1-bias-gelu-fp8':
            consumer_index = int(panel.call) + 1
            if consumer_index >= len(self.calls):
                raise ValueError('missing fc2 consumer')
            consumer = self.calls[consumer_index]
            if (consumer.site, consumer.layer, consumer.role, consumer.image, consumer.rows, consumer.input_width) != (
                    7, call.layer, 6, call.image, call.rows, call.output_width):
                raise ValueError('FP8 producer consumer geometry differs')
            fp8_scale = int(consumer.activation_scale)
        staged = (stage_fc1_fp8(normalized, initial, bias_bf16=bias, output_scale_exponent=fp8_scale)
                  if fp8_scale is not None else stage_panel(normalized, initial, operations=operations, bias_bf16=bias))
        meta = dict(staged.metadata)
        for key in ('initial_sha256', 'baseline_initial_sha256'):
            meta.pop(key)
        extra = staged.arena[meta['extra_offset']:meta['extra_offset'] + meta['extra_bytes']]
        flags = sum({'bias': 1, 'gelu_tanh': 2, 'connector_scale': 4}[op] for op in operations)
        if fp8_scale is not None:
            flags |= 8
        identity = (extra, meta['extra_offset'] if extra else 0, flags, fp8_scale or 0)
        if index not in self.static:
            storage = ct.create_string_buffer(extra) if extra else None
            if storage is not None:
                self.buffers.append(storage)
            self.panels[index] = FusionPanel(ct.addressof(storage) if storage else None,
                                            identity[1], len(extra), flags, fp8_scale or 0)
            self.static[index] = identity
        elif self.static[index] != identity:
            raise ValueError('immutable configured producer constants or output scale changed')
        panel.dram_bytes = meta['total_dram_bytes']
        panel.load_bytes = meta['requested_dma']['load_bytes']
        panel.store_bytes = meta['requested_dma']['store_bytes']
        self.prepared[index] = (normalized, operations, bias, fp8_scale, meta)
        return staged.source, meta, extra

    def input_view(self, pointer, rows, width):
        scale = ct.c_int()
        code = self.library.smol_fusion_input_view(ct.addressof(self.raw), pointer, rows, width, ct.byref(scale))
        if code < 0:
            raise ValueError('actual next FP8 consumer view rejected')
        return int(scale.value) if code else None

    def output_view(self, call_index):
        call = self.calls[call_index]
        first = int(call.first_panel)
        prepared = self.prepared.get(first)
        return prepared[3] if prepared else None

    def verify_stage(self, index, initial, source):
        if index not in self.prepared:
            return None
        row, operations, bias, fp8_scale, _ = self.prepared[index]
        m, _, n = row['physical_mkn']
        old_size = row['output_offset'] + 2 * m * n + 256
        baseline = (initial[:row['output_offset']] + b'\xcd'*(2*m*n) + b'Z'*256
                    if fp8_scale is not None else initial[:old_size])
        staged = (stage_fc1_fp8(row, baseline, bias_bf16=bias, output_scale_exponent=fp8_scale)
                  if fp8_scale is not None else stage_panel(row, baseline, operations=operations, bias_bf16=bias))
        if staged.arena != initial or staged.source != source:
            raise ValueError('actual C producer staging or source differs from independent fusion stage')
        return staged.metadata


def fp8_diagnostic(raw, rows, width, scale):
    """Decode a natural FP8 tile view for logical BF16 capture diagnostics only."""
    if len(raw) != rows*width or rows%32 or width%32:
        raise ValueError('whole natural FP8 tile extent required')
    values = np.frombuffer(raw, dtype=np.uint8).reshape(rows//32, width//32, 32, 32)
    values = values.transpose(0, 2, 1, 3).reshape(rows, width)
    exponent, mantissa = (values >> 3) & 15, values & 7
    if np.any((values & 127) == 127) or np.any((exponent == 0) & (mantissa != 0)):
        raise ValueError('FP8 producer emitted NaN or unsupported subnormal encoding')
    decoded = np.where(exponent != 0, np.ldexp(1 + mantissa.astype(np.float32)/8, exponent.astype(np.int32)-7), 0)
    decoded = np.where(values & 128, -decoded, decoded).astype(np.float32) * np.float32(2.0**scale)
    words = decoded.astype('<f4').view('<u4')
    return ((words + np.uint32(0x7fff) + ((words >> 16) & 1)) >> 16).astype('<u2').tobytes()
