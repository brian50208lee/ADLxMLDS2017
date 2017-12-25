import os
import random
import json
from scipy import misc
from scipy.misc import imread
import numpy as np

def load_train_data(imgs_dir, tags_path, feature_set, imresize_shape=[96,96,3], max_data_len=None):
    # load data
    X, Y = [], []
    with open(tags_path, 'r') as f:
        for line in f:
            img_id, tags = line.strip().lower().split(',')
            if int(img_id) % 1000 == 0:
                print('loading train image: {}'.format(img_id))
            # init feature
            feature = dict()
            for key in feature_set.keys():
                feature[key] = []
            # parse feature from tags
            tags = [tuple(tag.strip().split(':')) for tag in tags.split('\t')]
            for tag in tags:
                for feature_type, feature_name in feature_set.items():
                    if tag[0].strip() in feature_name:
                        feature[feature_type].append(tag)
            # select max
            for feature_type, feature_list in feature.items():
                feature[feature_type] = max(feature_list, key=lambda x: int(x[1]))[0] if len(feature_list) > 0 else ''
            sent = ' '.join(feature.values()).strip()
            sent = ' '.join(sent.split())
            if len(sent) == 0:
                continue
            Y.append(sent)
            # parse img
            img_path = os.path.join(imgs_dir, '{}.jpg'.format(img_id))
            img = misc.imread(img_path)
            img = misc.imresize(img, imresize_shape)
            X.append(img)
            # max data length
            if max_data_len and len(X) >= max_data_len:
                break

    return np.array(X), Y

def load_test_data(sents_path):
    test = []
    with open(sents_path, 'r') as f:
        for line in f:
            sent = line.split(',')[1].strip()
            test.append(sent)
    return test

def load_feature_set(feature_path):
    feature_dict = json.load(open(feature_path, 'r'))
    for key in feature_dict.keys():
        feature_dict[key] = set(feature_dict[key])
    return feature_dict

def load_feature_map(feature_path):
    feature_map = json.load(open(feature_path, 'r'))
    return feature_map

def sent2feature(sents, feature_map, max_feature_len=100):
    feature_dict = feature_map['hair_color'] + feature_map['eyes_color'] + feature_map['hair_style']
    feature_dict = dict(zip(feature_dict, range(len(feature_dict))))
    sents_vec = np.zeros([len(sents), max_feature_len], dtype=np.float32)
    for sent_idx, sent in enumerate(sents):
        for feature in feature_dict:
            if feature in sent:
                sents_vec[sent_idx ,feature_dict[feature]] = 1.0
    return sents_vec
