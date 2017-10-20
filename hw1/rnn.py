import os, sys
import numpy as np
import tensorflow as tf
from utils import load_phone_map, load_data, rearrange, pad, reverse_map

# argv
data_dir = sys.argv[1] + os.sep if len(sys.argv) > 1 else './data/'
f_output = sys.argv[2] if len(sys.argv) > 2 else 'predict.csv'

# file path
f_fbank_train = data_dir + 'fbank/train.ark'
f_fbank_test = data_dir + 'fbank/test.ark'
f_mfcc_train = data_dir + 'mfcc/train.ark'
f_mfcc_test = data_dir + 'mfcc/test.ark'
f_train_label = data_dir + 'label/train.lab'
f_phone2phone = data_dir + 'phones/48_39.map'
f_phone2char = data_dir + '48phone_char.map'

# load map
phone2phone, phone2char, phone2idx = load_phone_map(f_phone2phone, f_phone2char)

# load train
data_X, data_X_id = load_data(f_fbank_train, delimiter=' ', dtype='float32')
data_Y, data_Y_id = load_data(f_train_label, delimiter=',', dtype='str')
data_X, data_X_id = rearrange(data_X, data_X_id, data_Y_id)

# load test
test_X, test_X_id = load_data(f_fbank_test, delimiter=' ', dtype='float32')

# to 39 phone to idx to one-hot
for idx in range(len(data_Y)):
    data_Y[idx] = np.vectorize(phone2phone.get)(data_Y[idx])
    data_Y[idx] = np.vectorize(phone2idx.get)(data_Y[idx])
    data_Y[idx] = np.eye(48)[data_Y[idx].reshape(-1)]

# padding
max_squ_len = np.array([len(d) for d in data_X] + [len(t) for t in test_X]).max()
print('max_squ_len:{}'.format(max_squ_len))
data_X = np.array([pad(x, (max_squ_len, x.shape[1])) for x in data_X])
data_Y = np.array([pad(y, (max_squ_len, y.shape[1])) for y in data_Y])
test_X = np.array([pad(x, (max_squ_len, x.shape[1])) for x in test_X])

class SequenceLabelling(object):
    def __init__(self, input_dim, num_classes, max_squ_len, num_hidden=128, num_layers=3):
        self._input_dim = input_dim
        self._num_classes = num_classes
        self._max_squ_len = max_squ_len
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        #build
        tf.reset_default_graph()
        self.data = tf.placeholder(tf.float32, [None, self._max_squ_len, self._input_dim])
        self.target = tf.placeholder(tf.float32, [None, self._max_squ_len, self._num_classes])
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
    
    def fit(self, train, valid=None, dropout=0., num_epochs=10, batch_size=16, eval_every=1, shuffle=False):
        train_X = np.array(train[0], dtype='float32')
        train_Y = np.array(train[1], dtype='float32')
        for epoch in range(num_epochs):
            # shuffle
            if shuffle:
                shuffle_idx = np.random.permutation(len(train_X))
                train_X = train_X[shuffle_idx]
                train_Y = train_Y[shuffle_idx]
            # epoch
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
        output = self.data
        # Convolution network
        '''
        kernel_size = 5
        num_filters = 32
        output = tf.pad(output, [[0,0],[kernel_size//2,kernel_size//2],[0,0]])
        output = tf.expand_dims(output, -1)
        output = tf.layers.conv2d(inputs=output,
                                  filters=num_filters,
                                  kernel_size=[kernel_size, self._input_dim],
                                  padding="VALID",
                                  activation=tf.nn.relu)
        output = tf.reshape(output, [-1, self._max_squ_len, num_filters])
        '''
        # Recurrent network.
        def bidirectional_lstm(inputs, num_units, num_layers):
            bi_lstms = inputs
            for _ in range(num_layers):
                with tf.variable_scope(None, default_name="bidirectional-rnn"):
                    lstm_cell_fw = tf.contrib.rnn.LSTMCell(num_units, reuse=False)
                    lstm_cell_bw = tf.contrib.rnn.LSTMCell(num_units, reuse=False)
                    drop_cell_fw = tf.contrib.rnn.DropoutWrapper(lstm_cell_fw, input_keep_prob=1-self.dropout)
                    drop_cell_bw = tf.contrib.rnn.DropoutWrapper(lstm_cell_bw, input_keep_prob=1-self.dropout)
                    output, state = tf.nn.bidirectional_dynamic_rnn(drop_cell_fw, drop_cell_bw, bi_lstms,  dtype=tf.float32)
                    bi_lstms = output[0] + output[1]
            return bi_lstms
        '''
        cells = [tf.contrib.rnn.GRUCell(self._num_hidden, reuse=False) for _ in range(self._num_layers)]
        dropcells = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1-self.dropout) for cell in cells]
        multicell = tf.contrib.rnn.MultiRNNCell(dropcells, state_is_tuple=True)
        multicell = tf.contrib.rnn.DropoutWrapper(multicell, output_keep_prob=1-self.dropout)
        output, state = tf.nn.dynamic_rnn(multicell, self.data, dtype=tf.float32)
        '''
        output = bidirectional_lstm(output, self._num_hidden, self._num_layers)
        # Flatten to apply same weights to all time steps.
        output = tf.reshape(output, [-1, self._num_hidden])
        # Dense
        def dense(inputs, num_units, act=lambda x: x):
            weight, bias = self._weight_and_bias(inputs.get_shape()[-1].velue, num_units)
            return tf.matmul(inputs, weight) + bias
        # Softmax layer.
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        prediction = tf.reshape(prediction, [-1, self._max_squ_len, self._num_classes])
        return prediction
    
    def _build_loss(self):
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.target)
        cross_entropy = self.target * tf.log(self.prediction)
        cross_entropy = -tf.reduce_mean(cross_entropy, axis=2)
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
        weight = tf.truncated_normal([in_size, out_size], stddev=0.1)
        bias = tf.constant(1., shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)
    
    def save(self, checkpoint_filename):
        save_path = './{}.ckpt'.format(checkpoint_filename)
        tf.train.Saver(self.vars).save(self.sess, save_path)
        print('Model saved to: {}'.format(save_path))
    
    def load(self, checkpoint_filename):
        load_path = './{}.ckpt'.format(checkpoint_filename)
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
input_dim = data_X.shape[-1]
num_classes = data_Y.shape[-1]
model = SequenceLabelling(input_dim, num_classes, max_squ_len, num_hidden=128, num_layers=1)
model.summary()

valid_size = 200
valid_X, valid_Y = data_X[:valid_size], data_Y[:valid_size]
train_X, train_Y = data_X[valid_size:], data_Y[valid_size:]

model.fit(train=[train_X, train_Y], valid=[valid_X, valid_Y], dropout=0., num_epochs=50, batch_size=32, eval_every=1, shuffle=True)

def output_result(f_output, model, datas, instanse_id, frame_wise=False):
    print('predict size:{}'.format(len(datas)))
    # mask of sentence len
    sents_len = []
    for data in datas:
        for frame_idx, vector in enumerate(data):
            if all(vector == 0.):
                sents_len.append(frame_idx)
                continue
    # predict
    preds = model.predict(datas)
    # transform
    preds = np.vectorize(reverse_map(phone2idx).get)(preds)
    preds = np.vectorize(phone2char.get)(preds)
    # output prediction
    import re
    print('output:{}'.format(f_output))
    with open(f_output, 'w') as out:
        _ = out.write('id,phone_sequence\n')
        for data_idx, pred in enumerate(preds):
            result_str = pred[:sents_len[data_idx]]
            result_str = ''.join(result_str)
            if not frame_wise:
                result_str = result_str.strip(phone2char['sil']) # trim sil
                result_str = re.sub(r'([a-zA-Z0-9])\1+', r'\1', result_str) # trim
            _ = out.write('{},{}\n'.format(instanse_id[data_idx], result_str))

output_result('train_out_frame_wise.csv', model, data_X, data_X_id, frame_wise=True)
output_result('train_out.csv', model, data_X, data_X_id)
output_result(f_output, model, test_X, test_X_id)

# save
model.save('test')





