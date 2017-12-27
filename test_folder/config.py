import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

#########################
#   parameters setting  #
#########################

tf.app.flags.DEFINE_string('input_dir', 'E:\PROJECT\Foot_Height\data_Foot_Height\\barefoot_standard\V1.4.0.3', 'input_dir')
tf.app.flags.DEFINE_string('output_dir', 'E:\PROJECT\capsule_net\Barefoot_metric_learning\\test_folder\\feature', 'output_dir')
tf.app.flags.DEFINE_string('checkpoint', 'E:\PROJECT\capsule_net\Barefoot_metric_learning\\test_folder\checkpoint\\fisher_loss2175.ckpt', 'model path')

tf.app.flags.DEFINE_string('save_test', 'E:\PROJECT\capsule_net\Barefoot_metric_learning\\test_folder\\feature\\test_data.mat', 'save test data dir')
tf.app.flags.DEFINE_string('save_center', 'E:\PROJECT\capsule_net\Barefoot_metric_learning\\test_folder\\feature\\center_data.mat', 'save test data dir')
tf.app.flags.DEFINE_string('save_label', 'E:\PROJECT\capsule_net\Barefoot_metric_learning\\test_folder\\feature\\label_data.mat', 'save label dir')

tf.app.flags.DEFINE_string('load_label', 'E:\PROJECT\Fisher_loss\\feature\V1.4.0.3_footnet_v9_metric_v5\\label_data.mat', 'data input dir')
tf.app.flags.DEFINE_string('center_input', 'E:\PROJECT\Fisher_loss\\feature\V1.4.0.3_footnet_v9_metric_v5\\train\\', 'center input dir')


tf.app.flags.DEFINE_integer('height', 128, 'img_height')
tf.app.flags.DEFINE_integer('width', 59, 'img_width')
tf.app.flags.DEFINE_integer('channel', 3, 'img_channel')

