import numpy as np
import scipy.io as sio
import os
input_dir = 'F:\zjc\Barefoot_metric_learning\checkpoints\\feature\\'
personName = os.listdir(input_dir)
baseData = [];tarData=[]
for personName_i in personName:
    personPath = os.path.join(input_dir, personName_i)
    fileName = os.listdir(personPath)
    fc_path = os.path.join(personPath, fileName[0])
    fc = sio.loadmat(fc_path)['features']
    baseData.append(fc)
baseData = np.array(baseData)
#############################################################
for personName_i in personName:
    personPath = os.path.join(input_dir, personName_i)
    fileName = os.listdir(personPath)
    fc_path = os.path.join(personPath, fileName[0])
    fc = sio.loadmat(fc_path)['features']
    baseData.append(fc)
baseData = np.array(baseData)


print(np.shape(baseData))



