import argparse
from utils import load_feature_set, load_feature_map
from utils import load_train_data, load_test_data, sent2feature
from utils import dump_img
from GAN import GAN

# basic setting
feature_path = './feature.txt'
inputs_shape = (96, 96, 3)
seq_vec_len = 100
output_shape = (64, 64, 3)

# train setting
imgs_dir = './data/data/faces/'
tags_path = './data/data/tags_clean.csv'
exp_text_path = './exp_text.txt'

# test setting
best_model_path = './models/22000/cgan'
test_output_dir = './samples/'
samples_num = 5

# arg
def parse():
    parser = argparse.ArgumentParser(description='Anime GAN')
    parser.add_argument('--train', action='store_true', help="run train step")
    parser.add_argument('--test', action='store_true', help="run test step")
    parser.add_argument('--test_text', default='./data/data/sample_testing_text.txt', help="test condition file path")
    args, unknown = parser.parse_known_args()
    for arg in vars(args):
        print('{} = {}'.format(arg, getattr(args, arg)))
    if len(unknown) > 0:
        print('unknown args:', unknown)
    return args

def train():
    # feature
    feature_set = load_feature_set(feature_path)
    feature_map = load_feature_map(feature_path)

    # load data
    train_imgs, train_sents = load_train_data(imgs_dir, tags_path, feature_set, imresize_shape=inputs_shape, max_data_len=None)
    train_sents = sent2feature(train_sents, feature_map, max_feature_len=seq_vec_len)

    exp_sents, _ = load_test_data(exp_text_path)
    exp_sents = sent2feature(exp_sents, feature_map, max_feature_len=seq_vec_len)

    # data info
    print('feature map:', feature_map)
    print('train img shape:', train_imgs.shape)
    print('train sents shape:', train_sents.shape)
    print('exp sents:', exp_sents.shape)

    # model
    model = GAN(
                inputs_shape,
                seq_vec_len,
                output_shape,
                summary_path='./models/log'
            )

    # train
    model.train(train=[train_imgs, train_sents], valid_seqs=exp_sents)
    model.save('./models/finish/finish')

def test(test_text_path):
    # feature
    feature_set = load_feature_set(feature_path)
    feature_map = load_feature_map(feature_path)

    # load data
    test_sents, indices = load_test_data(test_text_path)
    test_sents = sent2feature(test_sents, feature_map, max_feature_len=seq_vec_len)

    # model
    model = GAN(
                inputs_shape,
                seq_vec_len,
                output_shape,
                seed=1034
            )
    model.load(best_model_path)

    # save result
    for sample_id in range(1, samples_num + 1):
        imgs = model.generate_image(test_sents)
        dump_img(test_output_dir, indices, imgs, sample_id)

if __name__ == '__main__':
    args = parse()
    if args.train:
        train()
    if args.test:
        test(args.test_text)

