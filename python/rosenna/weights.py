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

    # Check header (24 bytes)
    if len(blob) < 24:
        raise ValueError(f"{path}: file truncated; header requires 24 bytes, got {len(blob)}")
    version, dtype_code, endian, count = struct.unpack("<iiii", blob[8:24])
    if version != VERSION:
        raise ValueError(f"{path}: version {version}, expected {VERSION}")
    if endian != ENDIAN_MARKER:
        raise ValueError(f"{path}: endian marker {endian:#x}")

    # Check plan hash and TOC length marker (32 + 4 = 36 bytes after header)
    if len(blob) < 60:
        raise ValueError(f"{path}: file truncated; plan hash and TOC length require 60 bytes, got {len(blob)}")
    header = {"version": version, "dtype": _CODE_DTYPE[dtype_code],
              "plan_hash": blob[24:56].hex(), "count": count}
    toclen = struct.unpack("<i", blob[56:60])[0]
    data_start = 60 + toclen

    # Check that table of contents fits
    if len(blob) < data_start:
        raise ValueError(f"{path}: file truncated; table of contents ({toclen} bytes) extends to byte {data_start}, got {len(blob)} total")

    tensors, pos = {}, 60
    for tensor_idx in range(count):
        try:
            # Read name length
            if pos + 4 > len(blob):
                raise ValueError(f"{path}: tensor {tensor_idx}: file truncated reading name length at byte {pos}, need 4 bytes")
            namelen = struct.unpack("<i", blob[pos:pos + 4])[0]
            pos += 4

            # Read name
            if pos + namelen > len(blob):
                raise ValueError(f"{path}: tensor {tensor_idx}: file truncated reading name ({namelen} bytes) at byte {pos}")
            name = blob[pos:pos + namelen].decode("ascii")
            pos += namelen

            # Read rank
            if pos + 4 > len(blob):
                raise ValueError(f"{path}: tensor '{name}': file truncated reading rank at byte {pos}")
            rank = struct.unpack("<i", blob[pos:pos + 4])[0]
            pos += 4

            # Read dimensions
            if pos + 8 * rank > len(blob):
                raise ValueError(f"{path}: tensor '{name}': file truncated reading {rank} dimensions ({8*rank} bytes) at byte {pos}")
            dims = struct.unpack(f"<{rank}q", blob[pos:pos + 8 * rank])
            pos += 8 * rank

            # Read offset and length
            if pos + 16 > len(blob):
                raise ValueError(f"{path}: tensor '{name}': file truncated reading offset and length at byte {pos}")
            offset, length = struct.unpack("<qq", blob[pos:pos + 16])
            pos += 16

            # Verify data is accessible
            if data_start + offset + length > len(blob):
                raise ValueError(f"{path}: tensor '{name}': data extends to byte {data_start + offset + length}, file has {len(blob)} bytes")

            raw = blob[data_start + offset:data_start + offset + length]
            tensors[name] = np.frombuffer(raw, dtype=_NUMPY[header["dtype"]]).reshape(dims)
        except Exception as e:
            # Re-raise our ValueError as-is, wrap struct.error
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"{path}: tensor {tensor_idx}: {e}") from e

    return tensors, header
