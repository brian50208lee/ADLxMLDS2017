import os, sys
import numpy as np
import tensorflow as tf
from utils import load_train_data, load_test_data, sent2vec
from GAN import GAN

testing_text_path = sys.argv[1] + os.sep if len(sys.argv) > 1 else './data/data/special_text.txt'
imgs_dir = './data/data/faces/'
tags_path = './data/data/tags_clean.csv'
output_dir = './samples/'
special_text_path = './special_text.txt'

# params
output_resize = (64, 64, 3)
output_gene_num = 5
output_fname_format = 'sample_({testing_text_id})_({sample_id}).jpg'

# load data
imgs, sents = load_train_data(imgs_dir, tags_path, max_data_len=None)
sents, word2idx = sent2vec(sents)

test_sents = load_test_data(special_text_path)
test_sents, _ = sent2vec(test_sents, word2idx)

# data info
print('word2idx:', word2idx)
print('train img:', imgs.shape)
print('train sents:', sents.shape)
print('test sents:', test_sents.shape)

inputs_shape = (64, 64, 3)
seq_vec_len = 64
model = GAN(
	        inputs_shape,
        	seq_vec_len,
	        noise_len=100,
	        optimizer=tf.train.AdamOptimizer,
	        learning_rate=0.0001,
	        output_graph_path='tb'
		)
model.train(train=[imgs, sents], valid_sents=test_sents)








