from src.graph_builders import classification_graph
from src.tf_model_runner import TFRunner
from test.helper import mock_config

FLAGS = mock_config()
tfrunner = TFRunner(FLAGS.DATA.train, FLAGS.DATA.test, FLAGS.MODEL, classification_graph)
tfrunner.train()
