from Data_generator_triple_v3 import Datagenerator_txt
from footnet_v9 import my_alex
import tensorflow as tf
from metric_learning_v5 import fisher_loss
import os
import time
import numpy as np
#from triple_loss_v2 import triplet_loss
filepath_train = 'E:\PROJECT\Barefoot_Recognition\standard_data_txt\\V1.2.0.1_train_triple.txt'
filepath_test = 'E:\PROJECT\Barefoot_Recognition\standard_data_txt\\V1.2.0.1_test_triple.txt'
checkpoint_path = 'E:\PROJECT\Fisher_loss\checkpoint\\'
checkpoint = 'E:\PROJECT\Barefoot_Recognition\checkpoint\\best_checkpoint\my_net_inception_epoch759.ckpt'
filewriter_path = os.path.join(checkpoint_path, 'writer')
batch_person = 20
learning_rate = 0.01
person_file_num = 5
num_class = 1000
keep_prob1 = 0.7
epoch = 5000
batch_size = batch_person*person_file_num
x = tf.placeholder(tf.float32, [batch_size, 128, 59, 3])
y = tf.placeholder(tf.float32, [batch_size, num_class])
keep_prob = tf.placeholder(tf.float32, shape=None)
model = my_alex(x, keep_prob, num_class=num_class)
output, output2 = model.predict()
centerloss, crossloss = fisher_loss(output, batch_person, person_file_num)
with tf.name_scope('loss'):
  loss = centerloss/crossloss
tf.summary.scalar('loss', loss)
opt1 = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
opt = opt1.minimize(loss)
train_generator = Datagenerator_txt(file_path_train=filepath_train,
                            batch_person=batch_person, person_file_num=person_file_num, num_class=num_class)
test_generator = Datagenerator_txt(file_path_train=filepath_test,
                              batch_person=batch_person, person_file_num=person_file_num, num_class=num_class)
varlist = tf.trainable_variables()
# for var in varlist:
#     tf.summary.histogram(var.name, var)
# gradients = opt1.compute_gradients(loss, var_list=varlist[0:23])
# for gradient, var in gradients:
#   tf.summary.histogram(var.name + '/gradient', gradient)
saver = tf.train.Saver()
min_test_loss = 100000
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(filewriter_path)
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
  # saver.restore(sess, checkpoint_path)
   train_step = train_generator.num_step()
   test_step = test_generator.num_step()
   for i in range(epoch):
       sum_test_loss = 0
       print('training', i, 'epoch')
       train_generator.shuffle_folder()
       test_generator.shuffle_folder()
       for step_train in range(train_step):
         batch_x, label_train = train_generator.next_batch(step_train)
         optt, train_loss, cl, cr = sess.run([opt, loss, centerloss, crossloss], feed_dict={x: batch_x, y: label_train, keep_prob: keep_prob1})
         print('training', step_train, 'step', 'train_loss=', train_loss,
                'center_loss=', cl, 'cross_loss=', cr)
       for step_test in range(test_step):
         batch_x, label = test_generator.next_batch(step_test)
         [test_loss, test_cl, test_cr] = sess.run([loss, centerloss, crossloss], feed_dict={x: batch_x, y: label, keep_prob: 1})
         sum_test_loss = sum_test_loss + test_loss
         print('testing', step_test, 'step', 'test_loss=', test_loss, 'test_loss_mean=', sum_test_loss/(step_test+1),
               'center_loss=', test_cl, 'cross_loss=', test_cr, 'min_test_loss=', min_test_loss)
       # s = sess.run(merged_summary, feed_dict={x: batch_x, y: label, keep_prob1: 1})
       # writer.add_summary(s, i)
       if sum_test_loss/test_step < min_test_loss:
           min_test_loss = sum_test_loss/test_step
           print('**************************save model*******************************')
           checkpoint_name = os.path.join(checkpoint_path, 'fisher_loss' + str(i) + '.ckpt')
           saver.save(sess, checkpoint_name)


