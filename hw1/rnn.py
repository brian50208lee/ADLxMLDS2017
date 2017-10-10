import os, sys
import numpy as np
import tensorflow as tf
from utils import DataLoader

# argv
data_dir = sys.argv[1] + os.sep if len(sys.argv) > 1 else './data/'
output_file = sys.argv[2] if len(sys.argv) > 2 else 'predict.csv'

# file path
fbank_train = data_dir + 'fbank/train.ark'
fbank_test = data_dir + 'fbank/test.ark'
mfcc_train = data_dir + 'mfcc/train.ark'
mfcc_test = data_dir + 'mfcc/test.ark'
train_label = data_dir + 'label/train.lab'

# load data
loader = DataLoader()
train_X, train_Y, label_map, instance_map = loader.load(fbank_train, labels_path=train_label, num_classes=48)
test_X, _, _, instance_map = loader.load(fbank_test)

# padding
def pad(x, shape):
    pad_x = np.zeros(shape, dtype='float32')
    if len(shape) == 2:
        pad_x[:x.shape[0], :x.shape[1]] = x
    return pad_x

MAX_SQU_LEN = 800
train_X = [pad(x, (MAX_SQU_LEN, x.shape[1])) for x in train_X]
train_Y = [pad(y, (MAX_SQU_LEN, y.shape[1])) for y in train_Y]
test_X = [pad(x, (MAX_SQU_LEN, x.shape[1])) for x in test_X]

# model
input_dim = train_X[0].shape[-1]
num_classes = train_Y[0].shape[-1]
num_hidden = 128
num_layer = 2
tf.reset_default_graph()
# Placeholders
input_x = tf.placeholder(tf.float32, [None, MAX_SQU_LEN, input_dim])
label_y = tf.placeholder(tf.float32, [None, MAX_SQU_LEN, num_classes])
# rnn
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=0.0, state_is_tuple=True)
outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, input_x, dtype=tf.float32, time_major=False)
#init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
#outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, input_x, initial_state=init_state, time_major=False)

# softmax
outputs = tf.reshape(outputs, [-1, num_hidden])
weight = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
bias = tf.Variable(tf.zeros([1, num_classes]) + 0.1)
predict = tf.nn.softmax(tf.matmul(outputs, weight) + bias)
predict = tf.reshape(predict, [-1, MAX_SQU_LEN, num_classes])

# loss
mask = tf.sign(tf.reduce_max(tf.abs(label_y), axis=2))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=label_y)
cross_entropy *= mask
loss = tf.reduce_mean(tf.reduce_mean(cross_entropy))

# accuracy
correct = tf.equal(tf.argmax(predict, axis=2), tf.argmax(label_y, axis=2))
correct = tf.cast(correct, tf.float32) 
norm_mask = mask / tf.reduce_mean(mask)
correct *= norm_mask
accuracy = tf.reduce_mean(tf.reduce_mean(correct))

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
train_step = optimizer.minimize(loss)

# train
num_epochs = 10
batch_size = 1

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(num_epochs):
    num_steps = (len(train_X)-1)//batch_size + 1
    for step in range(num_steps):
        batch_x = train_X[step*batch_size : step*batch_size+batch_size]
        batch_y = train_Y[step*batch_size : step*batch_size+batch_size]
        print('epoch:{:>3}/{:<3}  step:{:>5}/{:<5}  '.format(epoch+1, num_epochs, step+1, num_steps), end='')
        _ = sess.run(train_step, {input_x: batch_x, label_y: batch_y})
        loss_value, acc_value = sess.run([loss, accuracy], {input_x: batch_x, label_y: batch_y})
        print('loss:{:<5f}  acc:{:<5f}'.format(loss_value, acc_value))
        break

# predict
preds = sess.run(predict, {input_x: train_X[:1]})
preds = np.argmax(preds, axis=2)
print(preds[0])




