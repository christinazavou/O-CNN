from config import parse_args, parse_class_weights
import sys, warnings
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


def test_parse_class_weights():
    testargs = ["python test_config.py",
                "--config", "../configs/segmentation/seg_hrnet_partnet_pts.yaml",
                "MODEL.nout", "31"]
    with patch.object(sys, 'argv', testargs):
        FLAGS = parse_args()
    weights = parse_class_weights(FLAGS)
    assert len(weights) == 31 and all(weights) == 1


def test_parse_class_weights_fails():
    with pytest.raises(AssertionError) as e:
        testargs = ["python test_config.py",
                    "--config", "../configs/segmentation/seg_hrnet_partnet_pts.yaml",
                    "SOLVER.class_weights", "../configs/class_weights.json",
                    "MODEL.nout", "31"]
        with patch.object(sys, 'argv', testargs):
            FLAGS = parse_args()
        weights = parse_class_weights(FLAGS)
    assert "Number of weights does not match number of outputs" in str(e)


def test_parse_class_weights_read():
    testargs = ["python test_config.py",
                "--config", "../configs/segmentation/seg_hrnet_partnet_pts.yaml",
                "SOLVER.class_weights", "../configs/class_weights.json",
                "MODEL.nout", "32"]
    with patch.object(sys, 'argv', testargs):
        FLAGS = parse_args()
    weights = parse_class_weights(FLAGS)
    assert len(weights) == FLAGS.MODEL.nout
    assert weights[0] == weights[1] == 0


def test_parse_args_warnings():
    testargs = ["python test_config.py",
                "--config", "../configs/segmentation/seg_hrnet_partnet_pts.yaml",
                "SOLVER.run", "test", "DATA.test.shuffle", "10", "DATA.test.batch_size", "10"]
    with patch.object(sys, 'argv', testargs):
        with warnings.catch_warnings(record=True) as w:
            FLAGS = parse_args()
            assert len(w) == 2
