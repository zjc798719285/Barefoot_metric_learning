from utils import getfilelist, get_center, get_data, search, save_data, load_data, save_err_img, list2path
import scipy.io as sio
from config import FLAGS
test_data = get_data('E:\PROJECT\capsule_net\Barefoot_metric_learning\\test_folder\\feature\Test', 256)
center_data = get_center('E:\PROJECT\capsule_net\Barefoot_metric_learning\\test_folder\\feature\Train', 256)
label = search(center_data, test_data)
save_data(test_data, 'test_data', FLAGS.save_test)
save_data(center_data, 'center_data', FLAGS.save_center)
save_data(label, 'label', FLAGS.save_label)
label = load_data(FLAGS.load_label)['label']
label = sio.loadmat('E:\PROJECT\capsule_net\Barefoot_metric_learning\\test_folder\\feature\label_data.mat')['label']
print(label)
save_err_img(label, 'E:\PROJECT\Foot_Height\data_Foot_Height\\barefoot_standard\V1.4.0.3\Test', 'E:\PROJECT\capsule_net\Barefoot_metric_learning\\test_folder\\feature\err_img')