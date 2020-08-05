from src.graph_builders import classification_graph
from src.tf_model_runner import TFRunner
from test.helper import *

# FLAGS = mock_train_config()
FLAGS = mock_test_config()
tfrunner = TFRunner(FLAGS.DATA.train, FLAGS.DATA.test, FLAGS.MODEL, classification_graph)
tfrunner.run()
