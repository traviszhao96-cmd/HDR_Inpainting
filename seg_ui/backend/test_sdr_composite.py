import unittest
import numpy as np
from .sdr_composite import composite_sdr, expand_generation_mask


class SDRCompositeTest(unittest.TestCase):
    def setUp(self):
        self.source = np.full((128, 128, 3), 100, np.uint8)
        self.mask = np.zeros((128, 128), bool)
        self.mask[32:96, 32:96] = True

    def test_empty_and_identity(self):
        np.testing.assert_array_equal(composite_sdr(self.source, self.source, self.mask), self.source)
        np.testing.assert_array_equal(composite_sdr(self.source, self.source + 20, self.mask & False), self.source)

    def test_boundary_and_outside(self):
        out = composite_sdr(self.source, self.source + 20, self.mask)
        np.testing.assert_array_equal(out[~self.mask], self.source[~self.mask])
        np.testing.assert_array_equal(out[32, 32:96], self.source[32, 32:96])
        self.assertGreater(int(out[64, 64, 0]), 110)
        self.assertLess(int(out[36, 64, 0]), 110)

    def test_expansion_and_full_mask(self):
        expanded = expand_generation_mask(self.mask)
        self.assertTrue(expanded[self.mask].all())
        self.assertFalse(expanded[21, 64])
        self.assertTrue(expanded[22, 64])
        out = composite_sdr(self.source, self.source + 20, np.ones_like(self.mask))
        self.assertTrue(np.isfinite(out).all())


if __name__ == '__main__':
    unittest.main()
