import torch
import time
import resnet_pytorch
from footnet_pytorch import FootNet
from DataGenerator import Datagenerator
import torch.optim as optim
from metric_learning_pytorch import metric_loss3
import numpy as np
batch_person = 10
learning_rate = 0.01
person_file_num = 5
filepath_train = 'E:\PROJECT\Barefoot_metric_learning\data_txt\\V1.4.0.7_700_train.txt'
filepath_test = 'E:\PROJECT\Barefoot_metric_learning\data_txt\\V1.4.0.7_700_test.txt'
save_path = 'E:\PROJECT\Barefoot_metric_learning\checkpoints\\model.pkl'

train_generator = Datagenerator(file_path=filepath_train, batch_person=batch_person, person_file_num=person_file_num)
test_generator = Datagenerator(file_path=filepath_test, batch_person=batch_person, person_file_num=person_file_num)
model = resnet_pytorch.resnet34(num_person=batch_person, num_file=person_file_num).to('cuda')
optimizer = optim.Adadelta(model.parameters(), lr=0.001, rho=0.7)
test_loss_min = 1e6
for _ in range(10000):
   t1 = time.time()
   batch_x, is_next_epoch = train_generator.next_batch()
   fc = model.forward(torch.cuda.FloatTensor(batch_x))
   center, cross, loss = metric_loss3(fc, batch_person=batch_person, num_file=person_file_num, fcs=128)
   loss.backward()
   optimizer.step()
   t2 = time.time()
   print('Training:center_loss=', center, 'cross_loss=', cross, 'loss=', loss, 'times=', t2 - t1)
   if is_next_epoch:
       test_loss = 0; test_num = 0
       print('******************begin testing***********************')
       for _ in range(100000):
           t3 = time.time()
           batch_, is_next_epoch = test_generator.next_batch()
           fc = model.forward(torch.cuda.FloatTensor(batch_))
           center, cross, loss = metric_loss3(fc, batch_person=batch_person, num_file=person_file_num, fcs=128)
           t4 = time.time()
           test_loss += float(loss); test_num += 1
           print('Testing:center_loss=', center,  'cross_loss=', cross, 'loss=', loss, 'times=', t4 - t3)
           if is_next_epoch:
               print('test_loss_mean=', test_loss / test_num)
               if test_loss / test_num < test_loss_min:
                   test_loss_min = test_loss / test_num
                   print('************saving model*******************')
                   torch.save(model, save_path)
               print('***************stop testing********************')
               break











