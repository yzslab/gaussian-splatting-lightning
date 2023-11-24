import unittest

import torch

from internal.encodings.positional_encoding import PositionalEncoding
from internal.models.vanilla_deform_model import get_embedder


class PositionalEncodingTestCase(unittest.TestCase):
    def test_positional_encoding(self):
        pe1 = PositionalEncoding(3, 10, True)
        pe2, _ = get_embedder(10, 3)

        input = torch.arange(3 * 100).reshape((-1, 3))
        self.assertTrue(torch.all(pe1(input) == pe2(input)))
        input = torch.randn((100, 3))
        self.assertTrue(torch.all(pe1(input) == pe2(input)))


if __name__ == '__main__':
    unittest.main()
