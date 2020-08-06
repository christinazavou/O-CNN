# from test.helper import *
from src.config import *
from src.graph_builders import classification_graph
from src.tf_model_runner import TFRunner

# FLAGS = mock_train_config()
# FLAGS = mock_test_config()
save_config("config.{}.yaml".format(FLAGS.MODEL.run))
tfrunner = TFRunner(FLAGS.DATA.train, FLAGS.DATA.test, FLAGS.MODEL, classification_graph)
tfrunner.run()
