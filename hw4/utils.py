import os
import random
from scipy import misc
from scipy.misc import imread
import numpy as np

def load_train_data(imgs_dir, tags_path, imresize_shape=[96,96,3], filter_tag=True, max_data_len=None):
    # define filter tag
    hair_tag_filter = set([
        'orange hair', 'white hair', 'aqua hair', 'gray hair',
        'green hair', 'red hair', 'purple hair', 'pink hair',
        'blue hair', 'black hair', 'brown hair', 'blonde hair'])
    eyes_tag_filter = set([
        'gray eyes', 'black eyes', 'orange eyes',
        'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
        'green eyes', 'brown eyes', 'red eyes', 'blue eyes'])
    UKN_tag = '<UKN>'
    # load data
    X, Y = [], []
    with open(tags_path, 'r') as f:
        for line in f:
            img_id, tags = line.strip().lower().split(',')
            if int(img_id) % 1000 == 0:
                print('loading train image: {}'.format(img_id))
            # parse tag
            tags = [tuple(tag.strip().split(':')) for tag in tags.split('\t')]
            if filter_tag:
                # filter out hair tag and eyes tag
                hair_tags = [(tag_name, tag_num) for tag_name, tag_num in tags if tag_name in hair_tag_filter]
                eyes_tags = [(tag_name, tag_num) for tag_name, tag_num in tags if tag_name in eyes_tag_filter]
                # skip if no hair tag and eyes tag
                if len(hair_tags) == 0 and len(eyes_tags) == 0: 
                    continue
                # select max tag_num 
                hair_tags = max(hair_tags, key=lambda x: int(x[1])) if len(hair_tags) > 0 else (UKN_tag, '0')
                eyes_tags = max(eyes_tags, key=lambda x: int(x[1])) if len(eyes_tags) > 0 else (UKN_tag, '0')
                tags = [hair_tags, eyes_tags]
                random.shuffle(tags)
            sent = [tag_name for tag_name, tag_num in tags]
            sent = ' '.join(sent)
            Y.append(sent)
            # parse img
            img_path = os.path.join(imgs_dir, '{}.jpg'.format(img_id))
            img = misc.imread(img_path)
            img = misc.imresize(img, imresize_shape)
            X.append(img)
            # max data length
            if max_data_len and len(X) >= max_data_len:
                break

    return np.array(X, dtype=np.float32)/255., Y

def load_test_data(sents_path):
	test = []
	with open(sents_path, 'r') as f:
		for line in f:
			sent = line.split(',')[1].strip()
			test.append(sent)
	return test

def sent2vec(sents, word2idx=None, max_seq_len=4):
	if word2idx is None: # build word2idx
		word2idx = dict()
		word2idx.update({'<UKN>': 0})
		for sent in sents:
			for word in sent.split():
				if word not in word2idx:
					word2idx[word] = len(word2idx)
	sents_vec = np.zeros([len(sents), max_seq_len * len(word2idx)], dtype=np.float32)
	for sent_idx, sent in enumerate(sents):
		for word_idx, word in enumerate(sent.split()):
			if word_idx >= max_seq_len:
				break
			if word in word2idx and word != '<UKN>':
				onehot_idx = word_idx * len(word2idx) + word2idx[word]
				sents_vec[sent_idx, onehot_idx] = 1.
	return sents_vec, word2idx


