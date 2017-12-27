from utils import *
from footnet_v9 import my_alex
import tensorflow as tf
from config import FLAGS
import cv2
import scipy.io as sio
import numpy as np
def img2vec(input_dir, output_dir):
   file_list = []
   output_list = []
   IOfilelist(input_dir, file_list, output_dir, output_list)
   mkdir(input_dir, output_dir)
   x = tf.placeholder(tf.float32, [1, FLAGS.height, FLAGS.width, FLAGS.channel])
   model = my_alex(x, 1, 1000)     #下一版本要改进，用参数搜集
   output, output2 = model.predict()
   saver = tf.train.Saver()
   with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())
       saver.restore(sess, FLAGS.checkpoint)
       print('strat img2vec')
       for index in range(len(file_list)):
          img = np.ndarray([1, FLAGS.height, FLAGS.width, FLAGS.channel])
          img1 = cv2.imread(file_list[index])
          img[0, :, :, :] = cv2.resize(img1, (59, 128))
          out = sess.run(output, feed_dict={x: img})
          feature = {'features': out}
          sio.savemat(output_list[index], feature)
          print('img2vec completion rate=', index/len(file_list))
       print('img2vec is done')
def main():
    img2vec(FLAGS.input_dir, FLAGS.output_dir)
if __name__ == '__main__':
   main()