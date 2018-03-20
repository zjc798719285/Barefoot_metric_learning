import os
import cv2
read_txt = 'F:\zjc\Barefoot_metric_learning\data_txt\V1.4.0.7_test.txt'
output_dir = 'F:\zjc\Barefoot_metric_learning\checkpoints\zjc_demo\Images\\'
f = open(read_txt, "r")
folder_name = f.readlines()
for name_i in folder_name:
    folder_i = name_i.split('\\', len(name_i))[-1]
    file_name = os.listdir(name_i[0:-1])
    folder_path = os.path.join(output_dir, folder_i)
    os.mkdir(folder_path[0:-1])
    for file_i in file_name:
        out_path = os.path.join(folder_path[0:-1], file_i)
        img_path = os.path.join(name_i[0:-1], file_i)
        img = cv2.imread(img_path)
        try:
         cv2.imwrite(out_path, img)
        except:
            continue







print()