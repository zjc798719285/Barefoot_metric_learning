import numpy as np
import tensorflow as tf
#类间算最小距离，类内计算平均中心距和最大中心距
def fisher_loss(fc, batch_person, person_file_num):
    centerloss = 0
    cross_loss = 0
    for i in range(batch_person):
      index = np.linspace(i*person_file_num,
                          i*person_file_num+person_file_num-1, person_file_num).astype(np.int32)
      person_fc = tf.gather(fc, index)
      person_center = tf.reduce_mean(person_fc, axis=0)
      centerloss = centerloss + center_loss(person_fc, person_center, person_file_num)
      tf.add_to_collection('center', person_center)
    center = tf.get_collection('center')
    for j in range(batch_person):
        cross_loss = cross_loss + nearst_search(index_p=j, center=center, fc=fc,
                                  batch_person=batch_person, person_file_num=person_file_num)
    return centerloss/batch_person, cross_loss/batch_person
def center_loss(person_fc, person_center, person_file_num):
    sum_centerdis = 0
    for i in range(person_file_num):
        centerdis = tf.nn.l2_loss(tf.gather(person_fc, i)-person_center)
        sum_centerdis = sum_centerdis + centerdis
        tf.add_to_collection('maxdis', centerdis)
    maxdis = tf.get_collection('maxdis')
    nn = tf.argmax(maxdis, axis=0)
    loss = tf.gather(maxdis, [nn])
    return (loss+sum_centerdis/person_file_num)/2
def nearst_search(index_p, center, fc, batch_person, person_file_num):
    batch_size = batch_person*person_file_num
    index = np.linspace(0, batch_size-1, batch_size).astype(np.int32)
    del_index = np.linspace(index_p*person_file_num,
                    (index_p+1)*person_file_num-1, person_file_num).astype(np.int32)
    index = np.delete(arr=index, obj=del_index, axis=0)
    pc_center = tf.gather(center, [index_p])
    others_fc = tf.gather(fc, index)
    for i in range((batch_person-1)*person_file_num):
        dis = tf.nn.l2_loss(pc_center - others_fc[i, :])
        tf.add_to_collection('nearst', dis)
    nearst = tf.get_collection('nearst')
    nn = tf.argmin(nearst, axis=0)
    nearst = tf.gather(nearst, [nn])
    return nearst
def pos_neg_batch(batch_x, batch_person, person_file_num):
    shape = np.shape(batch_x)
    anchor_batch = batch_x
    pos_batch = np.ndarray([int(shape[0]), int(shape[1])])
    neg_batch = np.ndarray([int(shape[0]), int(shape[1])])
    for i in range(batch_person):
        for j in range(person_file_num):
            index = i*batch_person+j
            dis_matrix = distance(batch_x[index, :], batch_x)
        pos_batch[index, :] = search_pos_dis(dis_matrix=dis_matrix, person=i,
                                             num_person_file=person_file_num, batch=batch_x)
        # neg_batch[index] = search_neg_dis(dis_matrix=dis_matrix, person=i,
        #                                      num_person_file=person_file_num, batch=batch_x)
    return anchor_batch, pos_batch, neg_batch
def search_pos_dis(dis_matrix, person, num_person_file,batch):
    pos_dis = dis_matrix[person*num_person_file:(person+1)*num_person_file-1]
    pos_index = np.argmax(pos_dis[:, 0])
    pos = batch[int(pos_dis[pos_index, 1]), :]
    return pos
def search_neg_dis(dis_matrix, person, num_person_file, batch):
    for i in range(num_person_file):
        index = person*num_person_file
        dis_matrix = np.delete(arr=dis_matrix, obj=index, axis=0)
    neg_idex = np.argmin(dis_matrix[:, 0])
    neg = batch[int(dis_matrix[neg_idex, 1]), :]
    return neg
def distance(x1, batch_x):
    batch_size = int(np.shape(batch_x)[0])
    dis_matrix = np.ndarray([batch_size, 2])
    for i in range(batch_size):
        s1 = np.sum(np.square(x1 - batch_x[i, :]))
        # s1 = np.subtract(x1, batch_x[i, :])
        # r1 = tf.reshape(s1, [-1, 64])
        # s3 = np.sum(np.square(r1), 1)
        # s3 = tf.reshape(s3, [-1, 1])
        dis_matrix[i, 0] = eval(s1)
        dis_matrix[i, 1] = i
    return dis_matrix


# def select_triplets(embeddings, nrof_images_per_class, people_per_batch, alpha):
#     """ Select the triplets for training
#     """
#     trip_idx = 0
#     emb_start_idx = 0
#     num_trips = 0
#     triplets = []
#
#     # VGG Face: Choosing good triplets is crucial and should strike a balance between
#     #  selecting informative (i.e. challenging) examples and swamping training with examples that
#     #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
#     #  the image n at random, but only between the ones that violate the triplet loss margin. The
#     #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
#     #  choosing the maximally violating example, as often done in structured output learning.
#
#     for i in range(people_per_batch):
#         nrof_images = int(nrof_images_per_class[i])
#         for j in range(1, nrof_images):
#             a_idx = emb_start_idx + j - 1
#             neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
#             for pair in range(j, nrof_images):  # For every possible positive pair.
#                 p_idx = emb_start_idx + pair
#                 pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
#                 neg_dists_sqr[emb_start_idx:emb_start_idx + nrof_images] = np.NaN
#                 #all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
#                 all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]  # VGG Face selecction
#                 nrof_random_negs = all_neg.shape[0]
#                 if nrof_random_negs > 0:
#                     rnd_idx = np.random.randint(nrof_random_negs)
#                     n_idx = all_neg[rnd_idx]
#                     triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
#                     # print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' %
#                     #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
#                     trip_idx += 1
#
#                 num_trips += 1
#
#         emb_start_idx += nrof_images
#
#     np.random.shuffle(triplets)
#     return triplets, num_trips, len(triplets)