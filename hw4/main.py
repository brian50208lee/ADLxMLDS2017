import sys
from utils import load_feature_set, load_feature_map
from utils import load_train_data, load_test_data, sent2feature
from utils import dump_img
from GAN import GAN

# params
feature_path = './feature.txt'
inputs_shape = (96, 96, 3)
seq_vec_len = 100
output_shape = (64, 64, 3)

# train setting
train = True
imgs_dir = './data/data/faces/'
tags_path = './data/data/tags_clean.csv'
exp_text_path = './exp_text.txt'

# test settubg
test = False
best_model_path = './models/0/cgan'
test_text_path = sys.argv[1] if len(sys.argv) > 1 else './data/data/sample_testing_text.txt'
test_output_dir = './samples/'
samples_num = 5

# feature
feature_set = load_feature_set(feature_path)
feature_map = load_feature_map(feature_path)

if train:
    # load data
    train_imgs, train_sents = load_train_data(imgs_dir, tags_path, feature_set, imresize_shape=inputs_shape, max_data_len=None)
    train_sents = sent2feature(train_sents, feature_map, max_feature_len=seq_vec_len)

    exp_sents, _ = load_test_data(exp_text_path)
    exp_sents = sent2feature(exp_sents, feature_map, max_feature_len=seq_vec_len)

    # data info
    print('feature map:', feature_map)
    print('train img:', train_imgs.shape)
    print('train sents:', train_sents.shape)
    print('exp sents:', exp_sents.shape)

    model = GAN(
                inputs_shape,
                seq_vec_len,
                output_shape,
                summary_path='./models/log'
            )
    model.train(train=[train_imgs, train_sents], valid_seqs=exp_sents)
    model.save('./models/finish/finish')

if test:
    test_sents, indices = load_test_data(test_text_path)
    test_sents = sent2feature(test_sents, feature_map, max_feature_len=seq_vec_len)

    model = GAN(
                inputs_shape,
                seq_vec_len,
                output_shape,
            )
    model.load(best_model_path)
    for sample_id in range(1, samples_num + 1):
        imgs = model.generate_image(test_sents)
        dump_img(test_output_dir, indices, imgs, sample_id)


