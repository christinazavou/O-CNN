from config import parse_args
import sys
import pytest


try:
    from unittest.mock import patch
except ImportError:
    from mock import patch


def test_parse_args_same_depth():
    with pytest.raises(ValueError) as e:
        testargs = ["python test_config.py",
                    "--config", "../configs/segmentation/seg_hrnet_partnet_pts.yaml",
                    "DATA.train.depth", "5", "DATA.test.depth", "7"]
        with patch.object(sys, 'argv', testargs):
            FLAGS = parse_args()
    assert "Train and test networks must have the same depth" in str(e)


def test_parse_args_depth_size():
    with pytest.raises(ValueError) as e:
        testargs = ["python test_config.py",
                    "--config", "../configs/segmentation/seg_hrnet_partnet_pts.yaml",
                    "DATA.train.depth", "10"]
        with patch.object(sys, 'argv', testargs):
            FLAGS = parse_args()
    assert "Network depth must be lesser or equal to 8" in str(e)
