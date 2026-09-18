"""Offline export regression: no model or network required.

Run: seg_ui/.venv/bin/python -m unittest seg_ui.backend.test_hdr_export
"""
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from .app import decode_uhdr, rebuild_uhdr, linear_to_hlg, srgb_to_linear


class HDRExportTest(unittest.TestCase):
    def test_p3_metadata_and_reediting(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            w, h = 128, 96
            sdr = np.full((h, w, 3), [180, 120, 60], dtype=np.float32)
            hdr = linear_to_hlg(srgb_to_linear(sdr / 255) * 0.7) * 1023
            exif = Image.Exif()
            exif[271] = "Export regression"
            source = directory / "input.jpg"
            Image.fromarray(sdr.astype(np.uint8)).save(source, exif=exif)
            mask = np.zeros((h, w), dtype=bool)
            output = rebuild_uhdr(hdr, sdr, sdr, mask, w, h, directory,
                                  source_path=source)
            data = output.read_bytes()
            self.assertIn(b"urn:iso:std:iso:ts:21496:-1", data)
            self.assertEqual(data.count(b"http://ns.adobe.com/xap/1.0/"), 2)
            with Image.open(output) as image:
                self.assertEqual(image.getexif()[271], "Export regression")
                profile = ImageCms.ImageCmsProfile(io.BytesIO(image.info["icc_profile"]))
                self.assertIn("Display P3", ImageCms.getProfileDescription(profile))
                # Pillow intentionally exposes XMP Ultra HDR as a single JPEG.
                # Follow the MPF directory to validate the auxiliary image.
                mp = image._getmp()
                self.assertEqual(mp[45057], 2)
                entries = mp[45058]
                gain_start = data.index(b"MPF\x00") + 4 + entries[1]["DataOffset"]
                self.assertEqual(entries[0]["Size"], gain_start)
                self.assertEqual(data[gain_start:gain_start + 2], b"\xff\xd8")
                gain_bytes = data[gain_start:gain_start + entries[1]["Size"]]
                self.assertEqual(gain_start + len(gain_bytes), len(data))
                gain = Image.open(io.BytesIO(gain_bytes))
                gain.load()
                self.assertEqual(gain.mode, "L")
            decoded_sdr, decoded_hdr = decode_uhdr(output, w, h, directory)
            self.assertLess(float(np.abs(decoded_sdr - sdr).mean()), 2)
            self.assertTrue(np.isfinite(decoded_hdr).all())
            second = directory / "second"
            second.mkdir()
            again = rebuild_uhdr(decoded_hdr, decoded_sdr, decoded_sdr, mask,
                                  w, h, second, source_path=output)
            sdr2, hdr2 = decode_uhdr(again, w, h, second)
            self.assertLess(float(np.abs(sdr2 - decoded_sdr).mean()), 2)
            with Image.open(again) as image:
                self.assertEqual(image.getexif()[271], "Export regression")


if __name__ == "__main__":
    unittest.main()
