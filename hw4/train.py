from utils import load_feature_set, load_feature_map
from utils import load_train_data, load_test_data, sent2feature
from GAN import GAN

imgs_dir = './data/data/faces/'
tags_path = './data/data/tags_clean.csv'
feature_path = './feature.txt'
exp_text_path = './exp_text.txt'

# params
inputs_shape = (96, 96, 3)
seq_vec_len = 100

# feature
feature_set = load_feature_set(feature_path)
feature_map = load_feature_map(feature_path)

# load data
train_imgs, train_sents = load_train_data(imgs_dir, tags_path, feature_set, imresize_shape=inputs_shape, max_data_len=1000)
train_sents = sent2feature(train_sents, feature_map, max_feature_len=seq_vec_len)
train_imgs = train_imgs.astype('float32') / 172.5 - 1.0 # normalize to [-1, 1]

test_sents = load_test_data(exp_text_path)
test_sents = sent2feature(test_sents, feature_map, max_feature_len=seq_vec_len)

# data info
print('feature_map:', feature_map)
print('train img:', train_imgs.shape)
print('train sents:', train_sents.shape)
print('test sents:', test_sents.shape)

model = GAN(
	        inputs_shape,
        	seq_vec_len,
	        summary_path='./models/tb'
		)
model.train(train=[train_imgs, train_sents], valid_seqs=test_sents)
model.save('./models/finish/finish')







