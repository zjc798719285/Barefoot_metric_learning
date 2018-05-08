import torch
import time
from footnet_pytorch import FootNet
from Data_generator_pytorch import Datagenerator_txt
import torch.optim as optim
from metric_learning_pytorch import metric_loss
import numpy as np
batch_person = 15
learning_rate = 0.01
person_file_num = 10
num_class = 1000
filepath_train = 'E:\PROJECT\Barefoot_metric_learning\data_txt\\V1.4.0.7_700_train.txt'
filepath_test = 'E:\PROJECT\Barefoot_metric_learning\data_txt\\V1.4.0.7_700_test.txt'

train_generator = Datagenerator_txt(file_path_train=filepath_train,
                                    batch_person=batch_person,
                                    person_file_num=person_file_num,
                                    num_class=num_class)
test_generator = Datagenerator_txt(file_path_train=filepath_test,
                                   batch_person=batch_person,
                                   person_file_num=person_file_num,
                                   num_class=num_class)

model = FootNet(batch_person, person_file_num).to('cuda')
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)


for step in range(100):
    t1 = time.time()
    batch_x, label_train = train_generator.next_batch(step)
    t2 = time.time()
    fc = model.forward(torch.cuda.FloatTensor(batch_x))
    loss = metric_loss(fc, batch_person)
    loss.backward()
    optimizer.step()
    t3 = time.time()
    print(t2-t1, t3-t2, t3-t1)
    print('loss=', loss)






