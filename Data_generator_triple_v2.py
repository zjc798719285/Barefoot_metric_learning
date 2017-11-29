import numpy as np
import cv2
import os
class Datagenerator_txt:
    def __init__(self, file_path_train, batch_person, person_file_num):
        self.file_path_train = file_path_train
        self.file_folder = self.read_folder()
        self.shuffle_folder()
        self.batch_person = batch_person
        self.person_file_num = person_file_num
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
        for i in range(self.person_file_num):
            file_name = list_shuffle(file_name)
            file_path = os.path.join(folder_path, file_name[1])
            person.append(file_path)
        return person
    def next_batch(self):
        self.shuffle_folder()
        batch_x = np.ndarray([self.batch_person*self.person_file_num, 128, 59, 3])
        file_path = []
        for i in range(self.batch_person):
           person_folder = self.file_folder[i]
           person = self.person_file(person_folder)
           file_path.append(person)
        for index_person, person_list in enumerate(file_path):
            for index, path in enumerate(person_list):
               img = cv2.imread(path)
               batch_x[(index_person+1)*(index+1)-1] = img
        return batch_x
def list_shuffle(lists):
       for value, list in enumerate(lists):
           rad = np.random.randint(low=0, high=len(lists), size=[1], dtype=np.int16)
           t = list
           lists[value] = lists[rad[0]]
           lists[rad[0]] = t
       return lists