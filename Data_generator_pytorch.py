import torch
import cv2
import os
import math
import numpy as np
import time
class Datagenerator_txt:
    def __init__(self, file_path_train, batch_person, person_file_num, num_class):
        self.file_path_train = file_path_train
        self.file_folder = self.read_folder()
        self.shuffle_folder()
        self.batch_person = batch_person
        self.person_file_num = person_file_num
        self.num_class = num_class
        self.batch_size = self.batch_person*self.person_file_num
    def num_step(self):
        length = len(self.file_folder)
        num_step = math.floor(length/self.batch_person)
        return num_step
    def label_erxtract(self, path):
        label = np.zeros([1, self.num_class])
        lab = path.split('\\', 10)[-2]
        for i in range(self.num_class):
            if str(i) == lab:
                label[:, i] = 1
        return label
    def read_folder(self):
        f = open(self.file_path_train, "r")
        folder_name = f.readlines()
        return folder_name
    def shuffle_folder(self):
        self.file_folder = list_shuffle(self.file_folder)
    def person_file(self, folder_path):
        person = []
        folder_path = folder_path[0:len(folder_path)-1]
        file_name = os.listdir(folder_path)
        for file_name_i in file_name:
            if file_name_i[-1] == 't':
                file_name.remove(file_name_i)
        for i in range(self.person_file_num):
            file_name = list_shuffle(file_name)
            file_path = os.path.join(folder_path, file_name[1])
            person.append(file_path)
        return person
    def next_batch(self, step):
        batch_x = np.ndarray([self.batch_size, 128, 59, 3])
        label = np.zeros([self.batch_size, self.num_class])
        file_path = []
        index_k = 0
        for i in range(self.batch_person):
           person_folder = self.file_folder[i+step*self.batch_person]
           person = self.person_file(person_folder)
           file_path.append(person)
        for index_person, person_list in enumerate(file_path):
            for index, path in enumerate(person_list):
               t1 = time.time()
               img = cv2.imread(path)
               t2 = time.time()
               img = cv2.resize(img, (59, 128))
               img = img_augment(img)
               t3 = time.time()
               batch_x[index_k] = img
               label[index_k] = self.label_erxtract(path)
               index_k = index_k+1
        batch_reshape = np.reshape(batch_x, [self.batch_person*self.person_file_num, 3, 128, 59])
        return batch_reshape, label
def list_shuffle(lists):
       for value, list in enumerate(lists):
           rad = np.random.randint(low=0, high=len(lists), size=[1], dtype=np.int16)
           t = list
           lists[value] = lists[rad[0]]
           lists[rad[0]] = t
       return lists
def img_augment(img):
    aug_lr = np.random.rand()
    if aug_lr >= 0.5:
        img = np.fliplr(img)
    return img

if __name__ == '__main__':
    aug_rand = np.random.rand()
    print(aug_rand)
