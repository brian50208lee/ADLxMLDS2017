import os, sys
import numpy as np

def reverse_map(map):
    reversed_map = dict([(value, key) for key, value in map.items()])
    return reversed_map

class DataLoader(object):

    def __init__(self):
        pass

    def load_map(f_48_39, f_48phone_char):
        map_48_39 = dict()
        with open(f_48_39, 'r') as f:
            for line in f:
                p1, p2 = line.strip().split('\t')
                map_48_39[p1] = p2

        map_48phone_char = dict()
        with open(file_path, 'r') as f:
            for line in f:
                p1, _, p2 = line.strip().split('\t')
                map_48phone_char[p1] = p2

        return map_48_39, map_48phone_char

    def load_dataset(self, dataset_path):
        print('load_dataset: {}'.format(dataset_path))
        dataset = dict()
        with open(dataset_path) as f:
            for line in f:
                tokens = line.strip().split()
                speker_id, sentence_id, frame_id = tokens[0].split('_')
                features = tokens[1:]
                
                if dataset.get(speker_id) is None:
                    dataset[speker_id] = dict()
                
                if dataset[speker_id].get(sentence_id) is None:
                    dataset[speker_id][sentence_id] = [features]
                else:
                    dataset[speker_id][sentence_id].append(features)
        return dataset

    def load_labels(self, labels_path):
        print('load_labels: {}'.format(labels_path))
        labels =dict()
        label_map = dict()
        label_map.update({'sil': 0})
        with open(labels_path) as f:
            for line in f:
                tokens = line.strip().split(',')
                speker_id, sentence_id, frame_id = tokens[0].split('_')
                label = tokens[1]
                
                if label_map.get(label) is None:
                    label_map[label] = len(label_map)

                label = label_map[label]

                if labels.get(speker_id) is None:
                    labels[speker_id] = dict()

                if labels[speker_id].get(sentence_id) is None:
                    labels[speker_id][sentence_id] = [label]
                else:
                    labels[speker_id][sentence_id].append(label)
        label_map = dict([(value, key) for key, value in label_map.items()])
        return labels, label_map


    def load(self, dataset_path, labels_path=None, num_classes=None):
        X, Y, label_map, instance_map = [], [], None, dict()

        # load
        dataset = self.load_dataset(dataset_path)
        if labels_path is not None:
            labels, label_map = self.load_labels(labels_path)
        
        # gen X, Y
        print('generate X, Y')
        if labels_path is None:
            for name in dataset.keys():
                for sent in dataset[name].keys():
                    x = np.array(dataset[name][sent], dtype='float32')
                    y = []
                    X.append(x)
                    Y.append(y)
                    instance_map[len(instance_map)] = '{}_{}'.format(name, sent)
        else:
            for name in labels.keys():
                for sent in labels[name].keys():
                    x = np.array(dataset[name][sent], dtype='float32')
                    y = np.eye(num_classes)[np.array(labels[name][sent], dtype='int32')]
                    X.append(x)
                    Y.append(y)
                    instance_map[len(instance_map)] = '{}_{}'.format(name, sent)

        return X, Y, label_map, instance_map

if __name__ == '__main__':
    loader = DataLoader()
    train_X, train_Y, label_map, instance_map = loader.load('data/fbank/train.ark', 'data/label/train.lab')
    test_X, _, _, instance_map = loader.load('data/fbank/test.ark')


