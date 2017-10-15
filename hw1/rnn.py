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
data_X, data_Y, label_map, instance_map = loader.load(fbank_train, labels_path=train_label, num_classes=48)
test_X, _, _, instance_map = loader.load(fbank_test)

# padding
def pad(x, shape):
    pad_x = np.zeros(shape, dtype='float32')
    if len(shape) == 2:
        pad_x[:x.shape[0], :x.shape[1]] = x
    return pad_x

MAX_SQU_LEN = 800
data_X = [pad(x, (MAX_SQU_LEN, x.shape[1])) for x in data_X]
data_Y = [pad(y, (MAX_SQU_LEN, y.shape[1])) for y in data_Y]
test_X = [pad(x, (MAX_SQU_LEN, x.shape[1])) for x in test_X]


class SequenceLabelling(object):
    def __init__(self, input_dim, num_classes, max_squ_len, num_hidden=128, num_layers=3):
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._max_squ_len = max_squ_len
        #build
        tf.reset_default_graph()
        self.data = tf.placeholder(tf.float32, [None, self._max_squ_len, input_dim])
        self.target = tf.placeholder(tf.float32, [None, self._max_squ_len, num_classes])
        self.dropout = tf.placeholder(tf.float32)
        self.prediction = self._build_prediction()
        self.loss = self._build_loss()
        self.optimize = self._build_optimize()
        self.accuracy = self._build_accuracy()
        # vars
        self.vars = {var.name: var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
        # init session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def fit(self, train, valid=None, dropout=0., num_epochs=10, batch_size=16, eval_every=1):
        train_X, train_Y = train
        for epoch in range(num_epochs):
            num_steps = (len(train_X)-1)//batch_size + 1
            for step in range(num_steps):
                batch_x = train_X[step*batch_size : step*batch_size+batch_size]
                batch_y = train_Y[step*batch_size : step*batch_size+batch_size]
                # run
                self.sess.run(self.optimize, feed_dict={self.data: batch_x, self.target: batch_y, self.dropout: dropout})
                loss, acc = self.evaluate(batch_x, batch_y)
                print('epoch:{:>2d}/{:<2d}  batch:{:>4d}/{:<4d}  '.format(epoch+1, num_epochs, step*batch_size, num_steps*batch_size), end='')
                print('loss:{:<3.5f}  acc:{:>3.1f}%  '.format(loss, 100*acc), end='')
                # evaluation
                if step % eval_every == 0 and valid is not None:
                    valid_x, valid_y = valid
                    val_loss, val_acc = self.evaluate(valid_x, valid_y)
                    print('val_loss:{:<3.5f}  va_acc:{:>3.1f}%  '.format(val_loss, 100*val_acc), end='')
                print()
    
    def evaluate(self, x, y):
        loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict={self.data: x, self.target: y, self.dropout: 0.})
        return loss, acc

    def predict(self, x):
        preds = self.sess.run(self.prediction, feed_dict={self.data: x, self.dropout: 0.})
        preds = np.argmax(preds, axis=2)
        return preds
    
    def _build_prediction(self):
        # Recurrent network.
        cells = [tf.contrib.rnn.GRUCell(self._num_hidden, reuse=False) for _ in range(self._num_layers)]
        dropcells = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1-self.dropout) for cell in cells]
        multicell = tf.contrib.rnn.MultiRNNCell(dropcells, state_is_tuple=True)
        multicell = tf.contrib.rnn.DropoutWrapper(multicell, output_keep_prob=1-self.dropout)
        output, _ = tf.nn.dynamic_rnn(multicell, self.data, dtype=tf.float32)
        # Softmax layer.
        max_length = int(self.target.get_shape()[1])
        num_classes = int(self.target.get_shape()[2])
        weight, bias = self._weight_and_bias(self._num_hidden, num_classes)
        # Flatten to apply same weights to all time steps.
        output = tf.reshape(output, [-1, self._num_hidden])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        prediction = tf.reshape(prediction, [-1, max_length, num_classes])
        return prediction
    
    def _build_loss(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.target)
        #cross_entropy = self.target * tf.log(self.prediction)
        #cross_entropy = -tf.reduce_mean(cross_entropy, axis=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), axis=2))
        mask /= tf.reduce_mean(tf.reduce_mean(mask))
        cross_entropy *= mask
        cross_entropy = tf.reduce_mean(cross_entropy, axis=1)
        return tf.reduce_mean(cross_entropy)
    
    def _build_optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        return optimizer.minimize(self.loss)
    
    def _build_accuracy(self):
        correct = tf.equal(tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
        correct = tf.cast(correct, tf.float32)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), axis=2))
        mask /= tf.reduce_mean(tf.reduce_mean(mask))
        correct *= mask
        return tf.reduce_mean(tf.reduce_mean(correct))
    
    def _weight_and_bias(self, in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=1.0)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)
    
    def save(self, checkpoint_filename):
        save_path = '{}.ckpt'.format(checkpoint_filename)
        tf.train.Saver(self.vars).save(self.sess, save_path)
        print('Model saved to: {}'.format(save_path))
    
    def load(self, checkpoint_filename):
        load_path = '{}.ckpt'.format(checkpoint_filename)
        tf.train.Saver(self.vars).restore(self.sess, load_path)
        print('Model restored from: {}'.format(load_path))
    
    def summary(self):
        print('='*50)
        print('Summary:')
        variables = [variable for variable in tf.trainable_variables()]
        total_parms = 0
        for variable in variables:
            name = variable.name
            shape = variable.shape
            parms = np.array(list(variable.shape), dtype='int32').prod()
            print('Var: {}    shape: {}    parms: {:,}'.format(name, shape, parms))
            total_parms += parms
        print('='*50)
        print('Total Parameters: {:,}'.format(total_parms))

# model
input_dim = data_X[0].shape[-1]
num_classes = data_Y[0].shape[-1]
max_squ_len = MAX_SQU_LEN
model = SequenceLabelling(input_dim, num_classes, max_squ_len, num_hidden=128, num_layers=1)
model.summary()

valid_size = 500
valid_X, valid_Y = data_X[:valid_size], data_Y[:valid_size]
train_X, train_Y = data_X[valid_size:], data_Y[valid_size:]
model.fit(train=[train_X, train_Y], valid=[valid_X, valid_Y], dropout=0., num_epochs=10, batch_size=16, eval_every=1)

# predict
preds = sess.run(predict, {input_x: train_X[:1]})
preds = np.argmax(preds, axis=2)
print(preds[0])




