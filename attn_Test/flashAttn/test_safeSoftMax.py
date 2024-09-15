import unittest

import numpy as np
import torch

from attn_Test.flashAttn.safeSoftMax import SoftMaxWithTiling


class SoftMaxTest(unittest.TestCase):
    def test_softmax(self):

        n_test = 10
        for _ in range(n_test):
            n_elem = np.random.randint(1, 11)
            x = np.random.randn(n_elem).tolist()
            expected = torch.nn.functional.softmax(torch.tensor(x), dim=-1).tolist()

            # out = SoftMax()(x)
            # self.assertTrue(np.allclose(expected, out, atol=1e-4))

            out_with_tiling = SoftMaxWithTiling().forward(x)
            self.assertTrue(np.allclose(expected, out_with_tiling, atol=1e-4))


if __name__  == "__main__":
    unittest.main()