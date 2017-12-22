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

# load data
train_imgs, train_sents = load_train_data(imgs_dir, tags_path, max_data_len=None)
train_sents, word2idx = sent2vec(train_sents)

test_sents = load_test_data(special_text_path)
test_sents, _ = sent2vec(test_sents, word2idx)

# data info
print('word2idx:', word2idx)
print('train img:', train_imgs.shape)
print('train sents:', train_sents.shape)
print('test sents:', test_sents.shape)

inputs_shape = (96, 96, 3)
seq_vec_len = 96
model = GAN(
	        inputs_shape,
        	seq_vec_len,
	        summary_path='tb'
		)
model.train(train=[train_imgs, train_sents], valid_seqs=test_sents)
model.save('./model/finish')







