import cv2
import os
import tensorflow as tf
import numpy as np
from config import FLAGS
import scipy.io as sio
from footnet_v9 import my_alex
def readlist(path, output):
    file_list = []
    output_list = []
    folder = os.listdir(path)
    for folder_i in folder:
        folder_path = os.path.join(path, folder_i)
        output_path = os.path.join(output, folder_i)
        filename = os.listdir(folder_path)
        for filename_i in filename:
            file_path = os.path.join(folder_path, filename_i)
            out_file_path = os.path.join(output_path, filename_i)
            file_list.append(file_path)
            output_list.append(out_file_path)
    return file_list, output_list
def getfilelist(input_dir, file_path):
    for sub_path in os.listdir(input_dir):
        filename = os.path.join(input_dir, sub_path)
        if os.path.isdir(filename):
            getfilelist(filename, file_path)
        else:
            file_path.append(filename)
    return file_path
def mkdir(path, output):
   folder = os.listdir(path)
   for folder_i in folder:
     folder_path_out = os.path.join(output, folder_i)
     if not os.path.exists(folder_path_out):
        os.mkdir(folder_path_out)

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











