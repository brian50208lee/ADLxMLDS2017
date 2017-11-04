import os, sys
import numpy as np
import tensorflow as tf

# argv
data_dir = sys.argv[1] + os.sep if len(sys.argv) > 1 else './data/MLDS_hw2_data/'
test_output = sys.argv[2] if len(sys.argv) > 2 else 'test.csv'
peer_output = sys.argv[3] if len(sys.argv) > 3 else 'peer.csv'

# file path
f_fbank_train = data_dir + 'fbank/train.ark'