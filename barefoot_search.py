import numpy as np
import scipy.io as sio
import os
import operator
feature_dir = 'F:\zjc\Barefoot_metric_learning\checkpoints\\features'
person = os.listdir(feature_dir)
mean_feature = []
all_feature = []
for person_i in person:
    filePath = os.path.join(feature_dir, person_i)
    fileName = os.listdir(filePath)
    for indFn, fileName_i in enumerate(fileName):
        sum_feature = np.zeros(512)
        feature_i = sio.loadmat(os.path.join(filePath, fileName_i))['features']
        all_feature.append({'feature': feature_i, 'person': person_i})
        sum_feature = sum_feature + feature_i
    mean_feature.append({'feature': sum_feature / indFn, 'person': person_i})
search_result = []
for all_i in all_feature:
    person_list = []
    for mean_i in mean_feature:
        diff = all_i['feature'] - mean_i['feature']
        dis = np.dot(diff, np.transpose(diff))
        person_list.append([dis, mean_i['person'], all_i['person']])
    person_list.sort(key=operator.itemgetter(0))
    search_result.append(person_list)

print(mean_feature)







