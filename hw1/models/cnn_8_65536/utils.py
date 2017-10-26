import os, sys
import numpy as np
import time

def load_phone_map(f_phone2phone, f_phone2char):
    # 48 to 39
    phone2phone = dict()
    with open(f_phone2phone, 'r') as f:
        for line in f:
            phone1, phone2 = line.strip().split('\t')
            phone2phone[phone1] = phone2
    # phone to char
    phone2char = dict()
    phone2idx = dict()
    with open(f_phone2char, 'r') as f:
        for line in f:
            phone, idx, char = line.strip().split('\t')
            phone2idx[phone] = int(idx)
            phone2char[phone] = char
    return phone2phone, phone2char, phone2idx

def load_data(f_path, delimiter=' ', dtype='float32'):
    print('load:{}  '.format(f_path), end='')
    start_time = time.time()
    instance_ids, datas= [], []
    with open(f_path, 'r') as f:
        for line in f:
            # parse line
            tokens = line.strip().split(delimiter)
            instance_id = '_'.join(tokens[0].split('_')[:-1])
            data = tokens[1:]
            # new sentence
            if len(instance_ids) == 0 or instance_id != instance_ids[-1]:
                if len(instance_ids) > 0:
                    datas[-1] = np.array(datas[-1], dtype=dtype)
                instance_ids.append(instance_id)
                datas.append([])
            # append data
            datas[-1].append(data)
    datas[-1] = np.array(datas[-1], dtype=dtype)
    print('time:{}'.format(time.time()-start_time))
    return np.array(datas), np.array(instance_ids)

def rearrange(datas, datas_instance_ids, base_ids):
    datas_instance_map = dict(zip(datas_instance_ids, list(range(len(datas_instance_ids)))))
    idx = np.vectorize(datas_instance_map.get)(base_ids)
    return datas[idx], datas_instance_ids[idx]

def pad(matrix, shape, value=0.):
    result = np.full(shape, value)
    result[:matrix.shape[0],:matrix.shape[1]] = matrix
    return result

def reverse_map(map):
    reversed_map = dict([(value, key) for key, value in map.items()])
    return reversed_map


if __name__ == '__main__':
    loader = DataLoader()
    train_X, train_Y, label_map, instance_map = loader.load('data/fbank/train.ark', 'data/label/train.lab')
    test_X, _, _, instance_map = loader.load('data/fbank/test.ark')


