import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

#########################
#   parameters setting  #
#########################

tf.app.flags.DEFINE_string('input_dir', 'E:\PROJECT\Fisher_loss\\feature\\footnet_v8_256_metric_v4\img', 'input_dir')
tf.app.flags.DEFINE_string('output_dir', 'E:\PROJECT\Fisher_loss\\feature\\footnet_v8_256_metric_v4\output', 'output_dir')
tf.app.flags.DEFINE_string('checkpoint', 'E:\PROJECT\capsule_net\Barefoot_metric_learning\\test_folder\checkpoint\\fisher_loss2378.ckpt', 'model path')

tf.app.flags.DEFINE_string('model', 'footnet_v9', 'model_file_name')
tf.app.flags.DEFINE_integer('height', 128, 'img_height')
tf.app.flags.DEFINE_integer('width', 59, 'img_width')
tf.app.flags.DEFINE_integer('channel', 3, 'img_channel')

