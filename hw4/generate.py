import os, sys

import numpy as np
import tensorflow as tf

# argv
fpath_testing_text = sys.argv[1] + os.sep if len(sys.argv) > 1 else './data/data/sample_testing_text.txt'
fpath_output_dir = './samples/'
fpath_train_img = './data/data/faces/'
fpath_special_text = './special_text.txt'

# params
output_resize = (64, 64)
output_gene_num = 5
output_fname_format = 'sample_({testing_text_id})_({sample_id}).jpg'

# filter
filter_hairs = [
    'orange hair', 'white hair', 'aqua hair', 'gray hair',
    'green hair', 'red hair', 'purple hair', 'pink hair',
    'blue hair', 'black hair', 'brown hair', 'blonde hair'
] + ['long hair', 'short hair']
filter_eyes = [
    'gray eyes', 'black eyes', 'orange eyes',
    'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
    'green eyes', 'brown eyes', 'red eyes', 'blue eyes'
] + ['bicolored eyes']



