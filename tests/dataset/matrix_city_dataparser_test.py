import os.path
import unittest
from internal.configs.dataset import MatrixCityParams
from internal.dataparsers.matrix_city_dataparser import MatrixCityDataParser


class MatrixCityDataparserTestCase(unittest.TestCase):
    def test_dataparser(self):
        dataparser = MatrixCityDataParser(
            os.path.expanduser("~/data/matrixcity/aerial/"),
            ".",
            0,
            MatrixCityParams(
                train=["aerial_train/transforms.json"],
                test=["aerial_test/transforms.json"],
            )
        )
        dataparser.get_outputs()
        dataparser = MatrixCityDataParser(
            os.path.expanduser("~/data/matrixcity/street/"),
            ".",
            0,
            MatrixCityParams(
                train=["small_city_road_vertical/transforms.json"],
                test=["small_city_road_vertical_test/transforms.json"],
                depth_read_step=16,
            )
        )
        dataparser.get_outputs()


if __name__ == '__main__':
    unittest.main()
