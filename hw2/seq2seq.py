
# load dataimport os, sys, json, time
import numpy as np
import tensorflow as tf

# argv
fpath_data_dir = sys.argv[1] + os.sep if len(sys.argv) > 1 else './data/MLDS_hw2_data/'
fpath_test_output = sys.argv[2] if len(sys.argv) > 2 else 'test.csv'
fpath_peer_output = sys.argv[3] if len(sys.argv) > 3 else 'peer.csv'

# file path
fpath_train_data = fpath_data_dir + 'training_data/feat/'
fpath_train_label = fpath_data_dir + 'training_label.json'
fpath_test_data = fpath_data_dir + 'testing_data/feat/'
fpath_test_ids = fpath_data_dir + 'testing_id.txt'


def load_train_data(fpath_data, fpath_label, word2id=None):
    """Load Training Data"""
    print('Load Training Data ...')
    start_time = time.time()
    # if no word2id
    if word2id is None:
        word2id = dict()
        word2id['<pad>'] = len(word2id) # padding
        word2id['<bos>'] = len(word2id) # begin of sentence
        word2id['<eos>'] = len(word2id) # end of sentence
        word2id['<ukn>'] = len(word2id) # unknown
    # load data
    X, Ys, video_ids = [], [], []
    for video in json.load(open(fpath_label, 'r')): # each vedio
        vedio_id = video['id']
        vedio_labels = video['caption']
        vedio_feature = np.load(fpath_data + vedio_id + '.npy')
        # word2id
        for caption_idx, caption in enumerate(vedio_labels): # each caption
            caption = caption.lower().strip().strip('.').split()
            caption = ['<bos>'] + caption + ['<eos>']
            for word_idx, word in enumerate(caption): # each word
                if word not in word2id:
                    word2id[word] = len(word2id)
                caption[word_idx] = word2id[word]
            vedio_labels[caption_idx] = caption
        # append
        X.append(vedio_feature)
        Ys.append(vedio_labels)
        video_ids.append(vedio_id)
    # return
    print('Time: {:.2f}s'.format(time.time()-start_time))
    return X, Ys, video_ids, word2id


def load_test_data(fpath_data, fpath_test_ids):
    """Load Testing Data"""
    print('Load Testing Data ...')
    start_time = time.time()
    # load data
    X, video_ids = [], []
    for line in open(fpath_test_ids, 'r'): # each vedio
        vedio_id = line.strip()
        vedio_feature = np.load(fpath_data + vedio_id + '.npy')
        # append
        X.append(vedio_feature)
        video_ids.append(vedio_id)
    # return
    print('Time: {:.2f}s'.format(time.time()-start_time))
    return X, video_ids


train_X, train_Ys, train_video_ids, word2id = load_train_data(fpath_train_data, fpath_train_label)

# params
feature_dim = train_X[0].shape[1]
vocab_size = len(word2id)
max_frame_len = train_X[0].shape[0]
max_sent_len = np.array([len(caption) for vedio in train_Ys for caption in vedio]).max()
print('feature_dim:', feature_dim)
print('vocab_size:', vocab_size)
print('max_frame_len:', max_frame_len)
print('max_sent_len:', max_sent_len)

# pading Ys
for vedio_idx, vedio in enumerate(train_Ys):
    for caption_idx, caption in enumerate(vedio):
        if len(caption) < max_sent_len:
            vedio[caption_idx] += [0]*(max_sent_len-len(caption))
    train_Ys[vedio_idx] = np.array(vedio)


class Seq2seq(object):
    def __init__(self, input_dim, vocab_size, hidden_dim, encode_steps, decode_steps, load_model_path=None):
        # params
        self._input_dim = input_dim
        self._vocab_size = vocab_size
        self._hidden_dim = hidden_dim
        self._encode_steps = encode_steps # video
        self._decode_steps = decode_steps # caption
        # placeholders
        tf.reset_default_graph()
        self.inputs = tf.placeholder(tf.float32, [None, self._encode_steps, self._input_dim]) # (batch_size, video_steps, features)
        self.caption = tf.placeholder(tf.int32, [None, self._decode_steps]) # (batch_size, caption_steps)
        self.batch_size = tf.placeholder(tf.int32)
        # variables
        self.encode_image_W = tf.Variable(tf.truncated_normal([self._input_dim, self._hidden_dim], stddev=0.1), name='encode_image_W')
        self.encode_image_B = tf.Variable(tf.zeros([self._hidden_dim]), name='encode_image_B')
        self.word_emb = tf.Variable(tf.truncated_normal([self._vocab_size, self._hidden_dim], stddev=0.1), name='word_emb')
        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(self._hidden_dim, state_is_tuple=True)
        self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(self._hidden_dim, state_is_tuple=True)
        self.decode_word_W = tf.Variable(tf.truncated_normal([self._hidden_dim, self._vocab_size], stddev=0.1), name='decode_word_W')
        self.decode_word_B = tf.Variable(tf.zeros([self._vocab_size]), name='decode_word_B')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='Adam')
        # models
        self.predict = self._build_predict()
        self.loss = self._build_loss()
        self.optimize = self._build_optimize()
        self.accuracy = self._build_accuracy()
        # vars
        self.vars = {var.name: var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
        self.saver = tf.train.Saver(self.vars)
        # init session
        self.sess = tf.Session()
        # restore model
        if load_model_path is not None:
            self.load(load_model_path)
        else:
            self.sess.run(tf.global_variables_initializer())
    
    def _build_predict(self):
        print('build predict')
        # dense (batch_size, video_steps, features) -> (batch_size, video_steps, hidden)
        video_flat = tf.reshape(self.inputs, [-1, self._input_dim])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_B)
        image_emb = tf.nn.relu(image_emb)
        image_emb = tf.reshape(image_emb, [-1, self._encode_steps, self._hidden_dim])
        # lstm encode
        state1_c = tf.zeros([self.batch_size, self.lstm1.state_size[0]])
        state1_h = tf.zeros([self.batch_size, self.lstm1.state_size[1]])
        state2_c = tf.zeros([self.batch_size, self.lstm2.state_size[0]])
        state2_h = tf.zeros([self.batch_size, self.lstm2.state_size[1]])
        padding = tf.zeros([self.batch_size, self._hidden_dim])
        for step in range(0, self._encode_steps):
            with tf.variable_scope(tf.get_variable_scope()):
                if step > 0:
                    tf.get_variable_scope().reuse_variables()
                output1, (state1_c, state1_h) = self.lstm1(image_emb[:,step,:], (state1_c, state1_h), scope='lstm1')
                output2, (state2_c, state2_h) = self.lstm2(tf.concat([padding, output1], 1), (state2_c, state2_h), scope='lstm2')
        # lstm decode
        caption = tf.pad(self.caption, [[0,0],[0,1]]) # padding one more step
        output_captions = []
        for step in range(0, self._decode_steps+1):
            with tf.variable_scope(tf.get_variable_scope()):
                with tf.device("/cpu:0"):
                    current_word_embed = tf.nn.embedding_lookup(self.word_emb, caption[:, step])
                tf.get_variable_scope().reuse_variables()
                output1, (state1_c, state1_h) = self.lstm1(padding, (state1_c, state1_h), scope='lstm1')
                output2, (state2_c, state2_h) = self.lstm2(tf.concat([current_word_embed, output1], 1), (state2_c, state2_h), scope='lstm2')
            output_captions.append(output2)
        output = tf.stack(output_captions[:-1], axis=1) # stack with step, ignore last padding step
        # dense 
        output = tf.reshape(output, [-1, self._hidden_dim])
        output = tf.nn.xw_plus_b(output, self.decode_word_W, self.decode_word_B)
        output = tf.nn.softmax(output)  
        output = tf.reshape(output, [-1, self._decode_steps, self._vocab_size]) 
        return output
    
    def _build_loss(self):
        print('build loss')
        caption = tf.one_hot(self.caption, depth=self._vocab_size, axis=2)
        cross_entropy = caption * tf.log(self.predict)
        cross_entropy = -tf.reduce_mean(cross_entropy, axis=2)
        mask = tf.cast(tf.not_equal(self.caption, 0), tf.float32)
        mask /= tf.reduce_mean(tf.reduce_mean(mask))
        cross_entropy *= mask
        cross_entropy = tf.reduce_mean(cross_entropy, axis=1)
        return tf.reduce_mean(cross_entropy)
    
    def _build_optimize(self):
        print('build optimize')
        clip_value = 1.
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables), clip_value)
        return self.optimizer.apply_gradients(zip(grads, trainable_variables))
    
    def _build_accuracy(self):
        print('build accuracy')
        correct = tf.equal(self.caption, tf.cast(tf.argmax(self.predict, 2), tf.int32))
        correct = tf.cast(correct, tf.float32)
        mask = tf.cast(tf.not_equal(self.caption, 0), tf.float32)
        mask /= tf.reduce_mean(tf.reduce_mean(mask))
        correct *= mask
        return tf.reduce_mean(tf.reduce_mean(correct))
    
    def fit(self, train, valid=None, num_epochs=10, batch_size=32, eval_every=1, shuffle=False, save_min_loss=False):
        train_X = np.array(train[0], dtype='float32')
        train_Y = np.array(train[1])
        min_loss = 0.
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
                batch_y = np.array([y[np.random.randint(len(y), size=1)[0]] for y in batch_y], dtype='int32')
                # run
                self.sess.run(self.optimize, feed_dict={self.inputs: batch_x, self.caption: batch_y, self.batch_size: len(batch_x)})
                loss, acc = self.evaluate(batch_x, batch_y, batch_size=batch_size)
                print('epoch:{:>2d}/{:<2d}  batch:{:>4d}/{:<4d}  '.format(epoch+1, num_epochs, step*batch_size, num_steps*batch_size), end='')
                print('loss:{:<3.5f}  acc:{:>3.1f}%  '.format(loss, 100*acc), end='')
                # evaluation
                if step % eval_every == 0 and valid is not None:
                    valid_x, valid_y = valid
                    val_loss, val_acc = self.evaluate(valid_x, valid_y, batch_size=batch_size)
                    print('val_loss:{:<3.5f}  val_acc:{:>3.1f}%  '.format(val_loss, 100*val_acc), end='')
                    # save_min_loss
                    if save_min_loss and (min_loss == 0. or val_loss < min_loss):
                        min_loss = val_loss
                        self.save('./models/best.ckpt', verbose=False)
                        print('save min loss model  '.format(min_loss), end='')
                print()
    
    def evaluate(self, x, y, batch_size=32):
        losses, accs = [], []
        offset = 0
        while offset < len(x):
            batch_x = x[offset : offset+batch_size]
            batch_y = y[offset : offset+batch_size]
            loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict={self.inputs: batch_x, self.caption: batch_y, self.batch_size: len(batch_x)})
            losses += [loss] * len(batch_x)
            accs += [acc] * len(batch_x)
            offset += batch_size
        return np.array(losses).mean(), np.array(accs).mean()
    
    def predict(self, x, batch_size=32):
        preds = []
        offset = 0
        while offset < len(x):
            batch_x = x[offset : offset+batch_size]
            pred = self.sess.run(self.prediction, feed_dict={self.inputs: batch_x, self.dropout: 0.})
            preds.append(np.argmax(pred, axis=2))
            offset += batch_size
        return np.vstack(preds)
    
    def save(self, checkpoint_file_path, verbose=True):
        if not os.path.exists(os.path.dirname(checkpoint_file_path)):
            os.makedirs(os.path.dirname(checkpoint_file_path))
        self.saver.save(self.sess, checkpoint_file_path)
        if verbose: print('Model saved to: {}'.format(checkpoint_file_path))
    
    def load(self, checkpoint_file_path, verbose=True):
        self.saver.restore(self.sess, checkpoint_file_path)
        if verbose: print('Model restored from: {}'.format(checkpoint_file_path))
    
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

model = Seq2seq(feature_dim, vocab_size, 100, max_frame_len, max_sent_len)
model.summary()
model.fit(train=[train_X, train_Ys], valid=None, num_epochs=1000, batch_size=32, shuffle=True, save_min_loss=True)



