import cv2
import tensorflow as tf
from footnet_v9 import my_alex
import scipy.io as sio
import numpy as np
import os
checkpoint = 'E:\PROJECT\Barefoot_metric_learning\checkpoints\checkpoints\checkpoint2400\\fisher_loss7.ckpt'
file_path = 'E:\PROJECT\Barefoot_metric_learning\data_txt\\V1.4.0.7_2400_test_.txt'
output_folder = 'E:\PROJECT\Barefoot_metric_learning\checkpoints\checkpoints\\features\\'
def person_folder_read(file_path_test):
   f = open(file_path_test, "r")
   folder_name = f.readlines()
   return folder_name
def person_file_read(folder_name, index,output_folder):
   person_folder = folder_name[index]
   person_folder = person_folder[0:len(person_folder)-1]
   person_id = person_folder.split('\\', 8)[-1]
   person_file_name = os.listdir(person_folder)
   for file_name_i in person_file_name:
       if file_name_i[-1] == 't':
           person_file_name.remove(file_name_i)
   person_file_path = []
   person_save_folder = []
   person_save_path = []
   for file_index, item in enumerate(person_file_name):
      file_id = item[0:-4] + '.mat'
      person_file_path.append(os.path.join(person_folder, item))
      person_save_folder.append(os.path.join(output_folder, person_id))
      person_save_path.append(os.path.join(output_folder, person_id, file_id))
   return person_file_path, person_save_folder, person_save_path
x = tf.placeholder(tf.float32, [2, 128, 59, 3])
y = tf.placeholder(tf.float32, [2, 128])
keep_prob = tf.placeholder(tf.float32, shape=None)
model = my_alex(x, keep_prob, num_class=1000)
output, output2 = model.predict()
saver = tf.train.Saver()
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   saver.restore(sess, checkpoint)
   person_folder = person_folder_read(file_path_test=file_path)
   img = np.ndarray([2, 128, 59, 3])
   for i in range(len(person_folder)):
       file_path, file_save_folder, file_save_path = person_file_read(person_folder, i, output_folder)
       if not os.path.exists(file_save_folder[0]):
        os.mkdir(file_save_folder[0])
       print(i)
       for j in range(len(file_path)):
           Image = cv2.resize(cv2.imread(file_path[j]), (59, 128))
           img[0, :, :, :] = Image
           img[1,:, :, :] = np.fliplr(Image)
           out = sess.run(output, feed_dict={x: img, keep_prob: 1})
          # os.mkdir(file_save_folder[j])
           feature = {'features': out}
           sio.savemat(file_save_path[j], feature)






