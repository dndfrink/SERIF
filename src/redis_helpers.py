import numpy as np
import struct
import traceback


def encode_numpy(inp: np.ndarray) -> bytes:
    if inp.dtype != np.float64:
        raise ValueError("Numpy encoding assumes floating point data")

    shape_dim = len(inp.shape)
    encoded_data = struct.pack('>I', shape_dim)
    for d in inp.shape:
        encoded_data += struct.pack('>I', d)
    return encoded_data + inp.tobytes()


def decode_numpy(byte_arr: bytes) -> np.ndarray:
    try:
        shape_list = []
        offset = 0

        shape_dim = struct.unpack('>I', byte_arr[:(offset + 4)])[0]
        offset += 4

        for i in range(shape_dim):
            shape_list.append(
                struct.unpack('>I', byte_arr[offset:(offset + 4)])[0]
            )
            offset += 4

        arr = np.frombuffer(bytearray(byte_arr[offset:])).reshape(shape_list)

        return arr
    except BaseException:
        print(traceback.format_exc())
        return None
