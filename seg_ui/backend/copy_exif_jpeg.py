"""Copy only the source JPEG EXIF APP1 segment, preserving destination XMP/Gainmap."""
from pathlib import Path
import sys

def jpeg_segments(data):
    if data[:2] != b'\xff\xd8': raise ValueError('not JPEG')
    pos=2
    while pos < len(data) and data[pos] == 0xff:
        marker=data[pos+1]; pos += 2
        if marker in (0xd8,0xd9): continue
        if marker == 0xda: break
        if pos+2 > len(data): break
        length=int.from_bytes(data[pos:pos+2],'big')
        yield marker,data[pos+2:pos+length],data[pos-2:pos+length]
        pos += length

def exif_segment(data):
    for marker,payload,raw in jpeg_segments(data):
        if marker == 0xe1 and payload.startswith(b'Exif\x00\x00'):
            return raw
    return None

def copy_exif(src,dst,out):
    source=Path(src).read_bytes(); target=Path(dst).read_bytes()
    seg=exif_segment(source)
    if seg is None: raise ValueError('source has no EXIF APP1')
    # The encoder output has no EXIF. Insert immediately after SOI and leave
    # every destination APP segment (including Ultra HDR XMP) byte-for-byte.
    Path(out).write_bytes(target[:2] + seg + target[2:])

if __name__ == '__main__':
    copy_exif(*sys.argv[1:4])
