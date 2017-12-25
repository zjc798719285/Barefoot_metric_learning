import cv2
import os
import tensorflow as tf
import numpy as np
from config import FLAGS
import scipy.io as sio
from footnet_v9 import my_alex
def getfilelist(input_dir, file_path, output_dir, output_path):
    for sub_path in os.listdir(input_dir):
        filename = os.path.join(input_dir, sub_path)
        outfile = os.path.join(output_dir, sub_path)
        if os.path.isdir(filename):
            getfilelist(filename, file_path, outfile, output_path)
        else:
            file_path.append(filename)
            output_path.append(outfile)
    return
def mkdir(path, output):
    for sub_path in os.listdir(path):
        filepath = os.path.join(path, sub_path)
        outpath = os.path.join(output, sub_path)
        if os.path.isdir(filepath):
            if not os.path.exists(outpath):
                os.mkdir(outpath)
            mkdir(filepath, outpath)

def img2vec(filelist, output_list):
   x = tf.placeholder(tf.float32, [1, FLAGS.height, FLAGS.width, FLAGS.channel])
   model = my_alex(x, 1, 1000)  #下一版本要改进，用参数搜集
   output, output2 = model.predict()
   saver = tf.train.Saver()
   with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())
       saver.restore(sess, FLAGS.checkpoint)
       for index in range(len(filelist)):
          img = np.ndarray([1, FLAGS.height, FLAGS.width, FLAGS.channel])
          img[0, :, :, :] = cv2.imread(filelist[index])
          out = sess.run(output, feed_dict={x: img})
          feature = {'features': out}
          sio.savemat(output_list[index], feature)











