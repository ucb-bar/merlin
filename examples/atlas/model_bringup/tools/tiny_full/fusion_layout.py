"""Pure tile codecs shared by frozen graph capture and Atlas export."""

def pack_bf16_tiles(value):
    """Pack uint16 BF16 words as row tiles, column tiles, and pair halves."""
    import numpy as np
    value = np.asarray(value, dtype='<u2')
    m, n = value.shape
    if m % 32 or n % 32:
        raise ValueError('BF16 tile geometry must be padded to 32')
    return value.reshape(m // 32, 32, n // 32, 2, 16).transpose(0, 2, 3, 1, 4).copy().tobytes()


def unpack_bf16_tiles(data, m, n):
    import numpy as np
    return np.frombuffer(data, dtype='<u2').reshape(m // 32, n // 32, 2, 32, 16).transpose(0, 3, 1, 2, 4).reshape(m, n).copy()


def pack_fp8_tiles(value):
    import numpy as np
    value = np.asarray(value, dtype=np.uint8)
    m, n = value.shape
    return value.reshape(m // 32, 32, n // 32, 32).transpose(0, 2, 1, 3).copy().tobytes()


def unpack_fp8_tiles(data, m, n):
    import numpy as np
    return np.frombuffer(data, dtype=np.uint8).reshape(m // 32, n // 32, 32, 32).transpose(0, 2, 1, 3).reshape(m, n).copy()


def pack_fp8_weights(value):
    import numpy as np
    value = np.asarray(value, dtype=np.uint8)
    k, n = value.shape
    return value.reshape(k // 32, 32, n // 32, 32).transpose(2, 0, 1, 3).copy().tobytes()
