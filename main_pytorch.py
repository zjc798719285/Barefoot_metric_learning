import torch
import time
import resnet_pytorch
from footnet_pytorch import FootNet
from DataGenerator import  Datagenerator
import torch.optim as optim
from metric_learning_pytorch import metric_loss
import numpy as np
batch_person = 15
learning_rate = 0.01
person_file_num = 10
num_class = 1000
filepath_train = 'E:\PROJECT\Barefoot_metric_learning\data_txt\\V1.4.0.7_700_train.txt'
filepath_test = 'E:\PROJECT\Barefoot_metric_learning\data_txt\\V1.4.0.7_700_test.txt'

# train_generator = Datagenerator_txt(file_path_train=filepath_train,
#                                     batch_person=batch_person,
#                                     person_file_num=person_file_num,
#                                     num_class=num_class)
# test_generator = Datagenerator_txt(file_path_train=filepath_test,
#                                    batch_person=batch_person,
#                                    person_file_num=person_file_num,
#                                    num_class=num_class)
train_generator = Datagenerator(file_path=filepath_train, batch_person=batch_person, person_file_num=person_file_num)

# model = FootNet(batch_person, person_file_num).to('cuda')
model2 = resnet_pytorch.resnet18(pretrained=False, num_person=batch_person,num_file=person_file_num).to('cuda')
optimizer = optim.Adadelta(model2.parameters(), lr=0.001, rho=0.7)

for _ in range(10000):
   for step in range(20):
      t1 = time.time()
      batch_x = train_generator.next_batch()
      t2 = time.time()
      fc = model2.forward(torch.cuda.FloatTensor(batch_x))
      center, cross, loss = metric_loss(fc, batch_person, 128)
      loss.backward()
      optimizer.step()
      t3 = time.time()
      print(t2-t1, t3-t2, t3-t1)
      print('loss=', loss, 'center', center, 'cross', cross)






