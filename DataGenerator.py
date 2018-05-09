import torch
import cv2
import os
import math
import numpy as np
import random
import time
class Datagenerator:
    def __init__(self, file_path, batch_person, person_file_num):
        print('Start Initialize Datagenerator')
        self.file_path_train = file_path
        self.batch_person = batch_person
        self.person_file_num = person_file_num
        self.imgList = []
        self.step = 1
        f = open(file_path, "r")
        for person in f.readlines():
            person_list = []
            for file_i in os.listdir(person[0:-1]):
                if file_i[-1] == 'g':
                    img_path = os.path.join(person[0:-1], file_i)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (59, 128))
                    img.reshape(3, 128, 59)
                    person_list.append(img)
                else:
                    continue
            self.imgList.append(person_list)
        print('Datagenerator initialization compelete')
    def shuffle(self):
        random.shuffle(self.imgList)
        self.step = 1


    def next_batch(self):
        batch_x = []
        is_next_epoch = False
        if self.step > int(math.floor(len(self.imgList)/self.batch_person)):
            self.shuffle()
            is_next_epoch = True
        start = (self.step - 1)*self.batch_person
        stop = self.step * self.batch_person
        batch_person = self.imgList[start:stop]
        for person_i in batch_person:
           idx = np.random.randint(low=1, high=len(person_i), size=self.person_file_num)
           imgs = [person_i[i] for i in idx]; imgs = np.array(imgs)
           batch_x.append(imgs)
        batch_x = np.array(batch_x)
        batch_x = batch_x.reshape(-1, 3, 128, 59)
        self.step += 1
        return batch_x, is_next_epoch









if __name__ == '__main__':
    file_path = 'E:\PROJECT\Barefoot_metric_learning\data_txt\\V1.4.0.7_700_train.txt'
    t1 = time.time()
    generator = Datagenerator(file_path=file_path, batch_person=15, person_file_num=10)
    t2 = time.time()
    print('init', t2-t1)
    for _ in range(10):
        t3 = time.time()
        generator.next_batch()
        t4 = time.time()
        print('run', t4-t3)
