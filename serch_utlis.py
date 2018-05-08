import numpy as np
import scipy.io as sio
import os, operator, cv2

def load_database(feature_dir):
    #用于载入数据库中所有特征
    database = []; person_center = []
    person = os.listdir(feature_dir)
    for person_i in person:
        filePath = os.path.join(feature_dir, person_i)
        fileName = os.listdir(filePath)
        person_sum = []
        for indFn, fileName_i in enumerate(fileName):
            feature_i = sio.loadmat(os.path.join(filePath, fileName_i))['features']
            database.append({'person': person_i, 'features': feature_i})
            person_sum.append(feature_i)
        feature_center = ({'person': person_i, 'person_center': np.mean(np.array(person_sum), axis=0)})
        person_center.append(feature_center)
    return database, person_center


def load_database2(feature_dir):
    #用于载入数据库中所有特征
    database = []
    person = os.listdir(feature_dir)
    for person_i in person:
        filePath = os.path.join(feature_dir, person_i)
        fileName = os.listdir(filePath)
        feature_list = []
        for indFn, fileName_i in enumerate(fileName):
            feature_i = sio.loadmat(os.path.join(filePath, fileName_i))['features']
            feature_list.append(feature_i)
        database.append({'person': person_i, 'features': feature_list})
    return database


def split_database(database):
    center = []; feature_list = []
    for data_i in database:
        person_i = data_i['person']
        features = data_i['features']
        ind = np.random.randint(low=0, high=len(features), size=1)[0]
        center.append({'person': person_i, 'person_center': features[ind]})
        features.pop(ind)
        for feature_i in features:
            feature_list.append({'person': person_i, 'features': feature_i})

    return center, feature_list



def foot_search(center, person):
    #该函数用于检索
    # 参数:
    #     database: 加载到内存中的数据库，这是一个列表，每个单元包含一个字典{'person'：person_id,'feauters':feature_list}
    #     fc: 被检索人的特征向量，一个128维ndarray数组
    # 返回：
    #     result_list：检索结果的排序，这是一个二维列表。[[metric_1, id_1],[metric_2, id_2]...[metric_n, id_n]]
    result_list = []
    fc = person['features']
    id = person['person']
    for center_i in center:
        feature_list = center_i['person_center']
        person_id = center_i['person']
        metric_list = []
        for feature_i in feature_list:
            diff = feature_i - fc
            metric = np.dot(diff, np.transpose(diff))  #欧氏距离矩阵
            metric_reshape = np.reshape(metric, newshape=4)
            metric_list.append(min(metric_reshape))
        metric_list.sort()
        result_list.append([metric_list[0], person_id])
    result_list.sort(key=operator.itemgetter(0))
    for top_i, result_i in enumerate(result_list):
        if result_i[1] == id:
            return result_list, top_i+1




if __name__ == '__main__':
    feature_dir = 'E:\PROJECT\Barefoot_metric_learning\checkpoints\checkpoints\\features'
    # persons, center = load_database(feature_dir=feature_dir)
    database = load_database2(feature_dir=feature_dir)
    center, persons = split_database(database)
    sum_top_1 = 0
    ratio = np.zeros(shape=(len(persons)))
    for person_i in persons:
        result, top_i = foot_search(center, person_i)
        top = np.zeros(shape=(len(persons)))
        top[top_i:-1] = 1
        ratio = ratio + top
        print(top_i)
        if top_i == 1:
            sum_top_1 += 1
    ratio = ratio / len(persons)
    print(ratio[0:20])



    # model_path = 'F:\zjc\Barefoot_metric_learning\checkpoints\zjc_demo\model\\fisher_loss4.ckpt'
    # file_path = 'F:\zjc\Barefoot_metric_learning\checkpoints\zjc_demo\Images\\36\\5636.jpg'
    # img = tf.placeholder(tf.float32, [1, 128, 59, 3])
    # model = my_alex(x=img)
    # output1, output2 = model.predict()
    # saver = tf.train.Saver()
    # with tf.Session() as sess:
    #     Image = cv2.imread(file_path)         #载入图片
    #     Image = cv2.resize(Image, (59, 128))  #resize尺寸
    #     Image = np.expand_dims(Image, axis=0) #增加维度
    #     saver.restore(sess, model_path)
    #     fc = sess.run(output1, feed_dict={img: Image})
    #     print(fc)








