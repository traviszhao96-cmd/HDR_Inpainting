#!/usr/bin/env python3
"""Extract the embedded JPEG gain map and metadata from an Ultra HDR image."""

import argparse
import ctypes as c
from pathlib import Path


class Error(c.Structure):
    _fields_ = [("error_code", c.c_int), ("has_detail", c.c_int), ("detail", c.c_char * 256)]


class CompressedImage(c.Structure):
    _fields_ = [
        ("data", c.c_void_p),
        ("data_sz", c.c_size_t),
        ("capacity", c.c_size_t),
        ("cg", c.c_int),
        ("ct", c.c_int),
        ("range", c.c_int),
    ]


class MemoryBlock(c.Structure):
    _fields_ = [("data", c.c_void_p), ("data_sz", c.c_size_t), ("capacity", c.c_size_t)]


class GainMapMetadata(c.Structure):
    _fields_ = [
        ("max_content_boost", c.c_float * 3),
        ("min_content_boost", c.c_float * 3),
        ("gamma", c.c_float * 3),
        ("offset_sdr", c.c_float * 3),
        ("offset_hdr", c.c_float * 3),
        ("hdr_capacity_min", c.c_float),
        ("hdr_capacity_max", c.c_float),
        ("use_base_cg", c.c_int),
    ]


def check(status, operation):
    if status.error_code:
        detail = bytes(status.detail).split(b"\0", 1)[0].decode(errors="replace")
        raise RuntimeError(f"{operation}: {detail}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("gainmap", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--library", type=Path, default=Path("build/libuhdr.1.4.0.dylib"))
    args = parser.parse_args()

    lib = c.CDLL(str(args.library.resolve()))
    lib.uhdr_create_decoder.restype = c.c_void_p
    lib.uhdr_dec_set_image.argtypes = [c.c_void_p, c.POINTER(CompressedImage)]
    lib.uhdr_dec_set_image.restype = Error
    lib.uhdr_dec_probe.argtypes = [c.c_void_p]
    lib.uhdr_dec_probe.restype = Error
    lib.uhdr_dec_get_gainmap_width.argtypes = [c.c_void_p]
    lib.uhdr_dec_get_gainmap_width.restype = c.c_int
    lib.uhdr_dec_get_gainmap_height.argtypes = [c.c_void_p]
    lib.uhdr_dec_get_gainmap_height.restype = c.c_int
    lib.uhdr_dec_get_gainmap_image.argtypes = [c.c_void_p]
    lib.uhdr_dec_get_gainmap_image.restype = c.POINTER(MemoryBlock)
    lib.uhdr_dec_get_gainmap_metadata.argtypes = [c.c_void_p]
    lib.uhdr_dec_get_gainmap_metadata.restype = c.POINTER(GainMapMetadata)
    lib.uhdr_release_decoder.argtypes = [c.c_void_p]

    encoded = args.input.read_bytes()
    buffer = c.create_string_buffer(encoded)
    image = CompressedImage(c.cast(buffer, c.c_void_p), len(encoded), len(encoded), -1, -1, -1)
    decoder = lib.uhdr_create_decoder()
    try:
        check(lib.uhdr_dec_set_image(decoder, c.byref(image)), "set image")
        check(lib.uhdr_dec_probe(decoder), "probe")
        width = lib.uhdr_dec_get_gainmap_width(decoder)
        height = lib.uhdr_dec_get_gainmap_height(decoder)
        block = lib.uhdr_dec_get_gainmap_image(decoder).contents
        gainmap_bytes = c.string_at(block.data, block.data_sz)
        metadata = lib.uhdr_dec_get_gainmap_metadata(decoder).contents

        args.gainmap.parent.mkdir(parents=True, exist_ok=True)
        args.gainmap.write_bytes(gainmap_bytes)
        lines = [
            f"gainmap_dimensions={width}x{height}",
            f"max_content_boost={list(metadata.max_content_boost)}",
            f"min_content_boost={list(metadata.min_content_boost)}",
            f"gamma={list(metadata.gamma)}",
            f"offset_sdr={list(metadata.offset_sdr)}",
            f"offset_hdr={list(metadata.offset_hdr)}",
            f"hdr_capacity_min={metadata.hdr_capacity_min}",
            f"hdr_capacity_max={metadata.hdr_capacity_max}",
            f"use_base_cg={metadata.use_base_cg}",
        ]
        args.metadata.write_text("\n".join(lines) + "\n")
        print(f"gainmap={args.gainmap} ({width}x{height}, {len(gainmap_bytes)} bytes)")
        print(f"metadata={args.metadata}")
    finally:
        lib.uhdr_release_decoder(decoder)


if __name__ == "__main__":
    main()
