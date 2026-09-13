"""The self-describing weights file: write it, and read it back for tests."""
import struct
from pathlib import Path

import numpy as np

MAGIC = b"ROSENNA1"
VERSION = 1
DTYPE_CODE = {"f32": 0, "f64": 1}
_CODE_DTYPE = {v: k for k, v in DTYPE_CODE.items()}
_NUMPY = {"f32": "<f4", "f64": "<f8"}
ENDIAN_MARKER = 0x01020304


def write_weights(plan, graph, path) -> None:
    toc, body = b"", b""
    for spec in plan.weights:
        array = np.asarray(graph.initializers[spec.name], dtype=_NUMPY[plan.dtype])
        if array.nbytes != spec.nbytes:
            raise ValueError(f"{spec.name}: plan says {spec.nbytes} bytes, array has {array.nbytes}")
        name = spec.name.encode("ascii")
        toc += struct.pack("<i", len(name)) + name + struct.pack("<i", array.ndim)
        toc += b"".join(struct.pack("<q", int(d)) for d in array.shape)
        toc += struct.pack("<qq", spec.offset, array.nbytes)
        body += array.tobytes(order="C")
    head = MAGIC + struct.pack("<iiii", VERSION, DTYPE_CODE[plan.dtype], ENDIAN_MARKER,
                               len(plan.weights)) + bytes.fromhex(plan.hash())
    Path(path).write_bytes(head + struct.pack("<i", len(toc)) + toc + body)


def read_weights(path):
    blob = Path(path).read_bytes()
    if blob[:8] != MAGIC:
        raise ValueError(f"{path}: not a roseNNa weights file")
    version, dtype_code, endian, count = struct.unpack("<iiii", blob[8:24])
    if version != VERSION:
        raise ValueError(f"{path}: version {version}, expected {VERSION}")
    if endian != ENDIAN_MARKER:
        raise ValueError(f"{path}: endian marker {endian:#x}")
    header = {"version": version, "dtype": _CODE_DTYPE[dtype_code],
              "plan_hash": blob[24:56].hex(), "count": count}
    toclen = struct.unpack("<i", blob[56:60])[0]
    data_start = 60 + toclen
    tensors, pos = {}, 60
    for _ in range(count):
        namelen = struct.unpack("<i", blob[pos:pos + 4])[0]
        pos += 4
        name = blob[pos:pos + namelen].decode("ascii")
        pos += namelen
        rank = struct.unpack("<i", blob[pos:pos + 4])[0]
        pos += 4
        dims = struct.unpack(f"<{rank}q", blob[pos:pos + 8 * rank])
        pos += 8 * rank
        offset, length = struct.unpack("<qq", blob[pos:pos + 16])
        pos += 16
        raw = blob[data_start + offset:data_start + offset + length]
        tensors[name] = np.frombuffer(raw, dtype=_NUMPY[header["dtype"]]).reshape(dims)
    return tensors, header
