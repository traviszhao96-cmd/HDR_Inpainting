#!/usr/bin/env python3
"""Measure the no-edit Ultra HDR encode/decode baseline for the bundled sample."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
APP = ROOT / "build" / "ultrahdr_app"
WIDTH = 2464
HEIGHT = 3280
SOURCE_HDR = ROOT / "libultrahdr/tests/photo/building_hlgRGBA1010102_output.raw"
SOURCE_SDR = ROOT / "libultrahdr/tests/photo/building.jpg"
GAMUTS = {0: "BT.709", 1: "Display P3", 2: "BT.2100"}


def run_app(*args):
    subprocess.run(
        [str(APP), *map(str, args)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def save_sdr_raw(image_path, raw_path):
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    if rgb.shape[:2] != (HEIGHT, WIDTH):
        raise ValueError(f"unexpected SDR size {rgb.shape[:2]}, expected {(HEIGHT, WIDTH)}")
    rgba = np.empty((HEIGHT, WIDTH, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 255
    rgba.tofile(raw_path)


def encode(hdr_raw, sdr_raw, output, hdr_gamut):
    run_app(
        "-m", 0,
        "-p", hdr_raw,
        "-y", sdr_raw,
        "-w", WIDTH,
        "-h", HEIGHT,
        "-a", 5,              # RGBA1010102
        "-b", 3,              # RGBA8888
        "-t", 1,              # HLG
        "-C", hdr_gamut,
        "-c", 0,              # SDR BT.709/sRGB
        "-q", 100,
        "-Q", 100,
        "-z", output,
    )


def decode(uhdr, sdr_raw, hdr_raw):
    run_app("-m", 1, "-j", uhdr, "-o", 3, "-O", 3, "-z", sdr_raw)
    run_app("-m", 1, "-j", uhdr, "-o", 1, "-O", 5, "-z", hdr_raw)


def load_sdr(path):
    return np.fromfile(path, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)[:, :, :3]


def load_hdr(path):
    packed = np.fromfile(path, dtype=np.uint32).reshape(HEIGHT, WIDTH)
    return np.stack(
        [packed & 1023, (packed >> 10) & 1023, (packed >> 20) & 1023], axis=-1
    )


def error_metrics(reference, candidate):
    delta = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    return {
        "mae": float(delta.mean()),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "p99": float(np.percentile(delta, 99)),
        "max": float(delta.max()),
    }


def validate():
    for required in (APP, SOURCE_HDR, SOURCE_SDR):
        if not required.is_file():
            raise FileNotFoundError(required)

    results = []
    with tempfile.TemporaryDirectory(prefix="uhdr-roundtrip-") as temp_dir:
        temp = Path(temp_dir)
        source_sdr_raw = temp / "source_sdr.raw"
        source_uhdr = temp / "source_p3_hlg.jpg"
        decoded_sdr = temp / "decoded_sdr.raw"
        decoded_hdr = temp / "decoded_hdr.raw"

        save_sdr_raw(SOURCE_SDR, source_sdr_raw)
        encode(SOURCE_HDR, source_sdr_raw, source_uhdr, hdr_gamut=1)
        decode(source_uhdr, decoded_sdr, decoded_hdr)
        sdr_reference = load_sdr(decoded_sdr)
        hdr_reference = load_hdr(decoded_hdr)

        for gamut, name in GAMUTS.items():
            output = temp / f"roundtrip_{gamut}.jpg"
            output_sdr = temp / f"roundtrip_{gamut}_sdr.raw"
            output_hdr = temp / f"roundtrip_{gamut}_hdr.raw"
            encode(decoded_hdr, decoded_sdr, output, hdr_gamut=gamut)
            decode(output, output_sdr, output_hdr)
            results.append(
                {
                    "hdr_input_gamut": name,
                    "sdr_error": error_metrics(sdr_reference, load_sdr(output_sdr)),
                    "hdr_error": error_metrics(hdr_reference, load_hdr(output_hdr)),
                    "encoded_size_bytes": output.stat().st_size,
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()
    results = validate()
    if args.json:
        print(json.dumps(results, indent=2))
        return

    print("No-edit decoded -> encoded -> decoded baseline")
    print("HDR input gamut | SDR MAE | HDR MAE | HDR P99 | HDR MAX | Size")
    for result in results:
        print(
            f"{result['hdr_input_gamut']:15} | "
            f"{result['sdr_error']['mae']:7.2f} | "
            f"{result['hdr_error']['mae']:7.2f} | "
            f"{result['hdr_error']['p99']:7.0f} | "
            f"{result['hdr_error']['max']:7.0f} | "
            f"{result['encoded_size_bytes'] / 1e6:4.1f} MB"
        )


if __name__ == "__main__":
    main()
