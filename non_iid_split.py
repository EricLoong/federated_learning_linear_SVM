import numpy as np
import random




def non_iid_index(size_data, number_users):
    """
    size_dara: size of the train data
    number_users: number of users in the distributed networks

    Returns: non_iid_data, distribution of data at different users.

    This function creates a random distribution for the users,
    i.e. number of images each client have.
    """
    temp_users = list()
    for i in range(number_users - 1):
        rand_temp = random.randint(10, 100)
        temp_users.append(rand_temp)

    weight_users = np.array(temp_users) / sum(temp_users)
    non_iid_data = (weight_users * size_data).astype(int)
    non_iid_data = list(non_iid_data)
    sum_v = sum(non_iid_data)
    non_iid_data.append(size_data - sum_v)
    non_iid_data[0] -= 436
    non_iid_data[-1] += 436
    return non_iid_data


distribution = non_iid_index(size_data=50000, number_users=20)

def m_split_to_n(m,n):
    # divide m samples into n random parts
    init = list(np.ones(n))

    for i in range((m-n)):
        ind = random.randint(0,n-1)
        init[ind] += 1

    return init

def find_index(array_a, array_b):
    result = list()
    for i in range(len(array_b)):
        result.append(int(np.where(array_a==array_b[i])[0]))

    return result




def split_img_noniid(data_x, labels_y, n_users=50):
    '''
    Split the image datasets(with labels) into n_users parts, following
    non-iid distribution. Assume at least each user holds two class.
    :param data: x part of the dataset
    :param labels: target value, y.
    :param n_users: number of users. default value is 50.
    :return: differen users hold different distribution of data, which satisfy the requirement
    of Fedearted Learning.
    '''
    classes = np.unique(labels_y)
    random.seed(1234567)
    number_classes_per_user = list()
    for i in range(n_users):
        random.seed(i)
        number_classes_per_user.append(random.randint(2, len(classes)))
    classes_per_user = dict()
    for i in range(n_users):
        temp_class = random.sample(list(classes), number_classes_per_user[i])
        classes_per_user[str(i)] = temp_class

    class_distribution = dict()
    for j in range(len(classes)):
        temp = list()
        for i in range(n_users):
            if classes_per_user[str(i)].count(j) !=0:
                temp.append(i)
        class_distribution[str(j)] = temp


    n_class_distribution = dict()
    for j in range(len(classes)):
        n_user_class = m_split_to_n(5000, len(class_distribution[str(j)]))
        n_class_distribution[str(j)] = n_user_class

    user_class_samples = dict()
    for i in range(n_users):
        user_class_samples[str(i)] = dict()
        for j in range(len(classes)):
            if i in class_distribution[str(j)]:
                ind = class_distribution[str(j)].index(i)
                v = n_class_distribution[str(j)]
                user_class_samples[str(i)][str(j)] = v[ind]

    class_index_y_train = dict()
    for i in range(len(classes)):
        class_index_y_train [str(i)]= np.where(labels_y==i)[0]
    # now we have the number of samples per class for each user.

    partition_collection = dict()

    for i in range(n_users):
        temp_selected_index = list()
        temp_keys = user_class_samples[str(i)].keys()
        for key in temp_keys:
            temp_index = class_index_y_train[key]
            # find number of samples for this class
            temp_n_samples = user_class_samples[str(i)][key]
            sampled_index = random.sample(list(temp_index), int(temp_n_samples))
            temp_selected_index.extend(sampled_index)
            class_index_y_train[key] = np.delete(temp_index, find_index(temp_index,
                                                                        np.array(sampled_index)))

        partition_collection[str(i)] = temp_selected_index

    # now select image data according to these indexes
    data_collection_user_x = dict()
    data_collection_user_y = dict()

    for i in range(n_users):
        data_collection_user_x[str(i)] = data_x[partition_collection[str(i)]]
        data_collection_user_y[str(i)] = np.array(labels_y[partition_collection[str(i)]])

    return data_collection_user_x, data_collection_user_y