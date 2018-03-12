import os
import numpy as np
import scipy.io as sio
import cv2
def IOfilelist(input_dir, file_path, output_dir, output_path):
    for sub_path in os.listdir(input_dir):
        filename = os.path.join(input_dir, sub_path)
        outfile = os.path.join(output_dir, sub_path)
        if os.path.isdir(filename):
            IOfilelist(filename, file_path, outfile, output_path)
        else:
            file_path.append(filename)
            output_path.append(outfile)

def getfilelist(input_dir, file_path):
    for sub_path in os.listdir(input_dir):
        filename = os.path.join(input_dir, sub_path)
        if os.path.isdir(filename):
            getfilelist(filename, file_path)
        else:
            file_path.append(filename)

def mkdir(path, output):
    for sub_path in os.listdir(path):
        filepath = os.path.join(path, sub_path)
        outpath = os.path.join(output, sub_path)
        if os.path.isdir(filepath):
            if not os.path.exists(outpath):
                os.mkdir(outpath)
            mkdir(filepath, outpath)

def get_data(input_dir, length):
    feature_list = []
    getfilelist(input_dir, feature_list)
    data = np.ndarray([len(feature_list), length + 1])
    for index, path in enumerate(feature_list):
        label = path.split('\\', len(path))[-2]
        data[index, 0:length] = sio.loadmat(path)['features']
        data[index, length] = float(label)
    return data

def get_center(input_dir, length):
    data = get_data(input_dir, length)
    center = np.zeros([1, length + 1])
    c_sum = np.zeros([1, length + 1])
    c_length = 0
    for index in range(np.shape(data)[0]):
        if index == np.shape(data)[0]-1:
            index_1 = 0
        else:
            index_1 = index + 1
        if data[index, length] == data[index_1, length]:
            c_sum += data[index, :]
            c_length += 1
        elif data[index, length] != data[index_1, length]:
            c_sum += data[index, :]
            c_length += 1
            c_sum = c_sum/c_length
            center = np.row_stack((center, c_sum))
            c_sum = c_sum * 0
            c_length = 0
    center = np.delete(center, 0, 0)
    return center

def search(source, target):
    label_list = []
    i = 0
    for target_i in target:
        print('search', i/len(target))
        i += 1
        min_dis = 1e6; min_label = -1
        for source_i in source:
            dis = np.sqrt(np.sum(np.square((target_i[0:-1] - source_i[0:-1]))))
            if dis < min_dis:
                min_label = source_i[-1]
                min_dis = dis
        label_list.append([target_i[-1], min_label])
    return label_list

def save_data(data, name, save_dir):
    Data = {name: data}
    sio.savemat(save_dir, Data)

def load_data(load_dir):
    data = sio.loadmat(load_dir)
    return data

def save_err_img(label, input_dir, output_dir):
    img_list = []
    getfilelist(input_dir, img_list)
    for index, lab in enumerate(label):
        if lab[0] != lab[1]:
            img = cv2.imread(img_list[index])
            seg_input = input_dir.split('\\', len(input_dir))
            seg_list = img_list[index].split('\\', len(img_list[index]))
            sub_path = [i for i in seg_list if i not in seg_input]
            sub_path.insert(0, output_dir)
            sub_path.pop(-2)
            sub_path.insert(-1, str(int(lab[1])))
            file_output = list2path(sub_path)
            out_dir = list2path(sub_path[0:-1])
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            cv2.imwrite(file_output, img)

def save_ori_img(input_dir, output_dir):
    img_list = []
    getfilelist(input_dir, img_list)




    return 1


def list2path(list):
    path = ''
    for item in list:
        path = os.path.join(path, item)
    return path