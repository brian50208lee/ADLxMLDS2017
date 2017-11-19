# load data
import os, sys, json, time
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

# train
run_train = True

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
id2word = dict(zip(word2id.values(), word2id))

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
        self.batch_size = tf.placeholder(tf.int32) # number of videos
        self.ground_truth_prob = tf.placeholder(tf.float32) # feed truth/predict word. 0.~1. if train, 0. if test
        # variables
        self.encode_image_W = tf.Variable(tf.truncated_normal([self._input_dim, self._hidden_dim], stddev=0.1), name='encode_image_W')
        self.encode_image_B = tf.Variable(tf.zeros([self._hidden_dim]), name='encode_image_B')
        self.word_emb = tf.Variable(tf.truncated_normal([self._vocab_size, self._hidden_dim], stddev=0.1), name='word_emb')
        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(self._hidden_dim, state_is_tuple=True) # encode
        self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(self._hidden_dim, state_is_tuple=True) # decode
        self.decode_word_W = tf.Variable(tf.truncated_normal([self._hidden_dim, self._vocab_size], stddev=0.1), name='decode_word_W')
        self.decode_word_B = tf.Variable(tf.zeros([self._vocab_size]), name='decode_word_B')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='Adam')
        # models
        self.pred, att_alphas = self._build_predict()
        self.loss = self._build_loss(att_alphas)
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
        encode_hs = []
        for step in range(0, self._encode_steps):
            with tf.variable_scope(tf.get_variable_scope()):
                if step > 0:
                    tf.get_variable_scope().reuse_variables()
                output1, (state1_c, state1_h) = self.lstm1(tf.concat([padding, image_emb[:,step,:]], 1), (state1_c, state1_h), scope='lstm1')
                output2, (state2_c, state2_h) = self.lstm2(tf.concat([padding, output1], 1), (state2_c, state2_h), scope='lstm2')
                encode_hs.append(output1)
        # attension
        encode_hs = tf.stack(encode_hs, axis=0) # t x b x h
        encode_hs = tf.transpose(encode_hs, [1,0,2]) # b x t x h 
        def attention_context(encode_hs, decode_h):
            # encode_hs -> b x t x h
            # alpha
            alphas = tf.multiply(encode_hs, tf.expand_dims(decode_h, 1))
            alphas = tf.reduce_sum(alphas, 2, keep_dims=True)
            # weighted sum
            alphas = tf.nn.softmax(alphas, 1)
            contex = tf.multiply(encode_hs, alphas)
            contex = tf.reduce_sum(contex, axis=1)
            return contex, alphas
        # lstm decode
        caption = tf.pad(self.caption, [[0,0],[0,1]]) # padding one more step
        output_captions = []
        att_alphas = []
        for step in range(0, self._decode_steps+1):
            with tf.variable_scope(tf.get_variable_scope()):
                # random select from ground truth or predict by self.ground_truth_prob
                previous_word_gt = caption[:, step] # ground truth
                previous_word_pred = tf.cast(tf.argmax(output2, axis=1), tf.int32) # predict
                indice = tf.multinomial(tf.log([[self.ground_truth_prob, 1-self.ground_truth_prob]]), 1)
                indice = tf.squeeze(indice, [0])
                previous_word = tf.gather(tf.stack([previous_word_gt, previous_word_pred]), indice)
                previous_word = tf.squeeze(previous_word, [0])
                # word embedding
                with tf.device("/cpu:0"):
                    previous_word_embed = tf.nn.embedding_lookup(self.word_emb, previous_word)
                # feed
                tf.get_variable_scope().reuse_variables()
                output1, (state1_c, state1_h) = self.lstm1(tf.concat([previous_word_embed, padding], 1), (state1_c, state1_h), scope='lstm1')
                # attention
                context, alphas = attention_context(encode_hs, output1)
                att_alphas.append(alphas)
                output2, (state2_c, state2_h) = self.lstm2(tf.concat([context, output1], 1), (state2_c, state2_h), scope='lstm2')
            output_captions.append(output2)
        output = tf.stack(output_captions[:-1], axis=1) # stack with step, ignore last padding step
        # dense 
        output = tf.reshape(output, [-1, self._hidden_dim])
        output = tf.nn.xw_plus_b(output, self.decode_word_W, self.decode_word_B)
        output = tf.nn.softmax(output)  
        output = tf.reshape(output, [-1, self._decode_steps, self._vocab_size]) 
        # alphas
        att_alphas = tf.stack(att_alphas, axis=0) # t, b, v, 1
        return output, att_alphas
    
    def _build_loss(self, att_alphas):
        print('build loss')
        caption = tf.pad(self.caption, [[0,0],[0,1]]) # padding one more step
        caption = caption[:,1:]
        mask = tf.cast(tf.not_equal(caption, 0), tf.float32)
        mask /= tf.reduce_mean(tf.reduce_mean(mask))
        caption = tf.one_hot(caption, depth=self._vocab_size, axis=2)
        cross_entropy = caption * tf.log(self.pred)
        cross_entropy = -tf.reduce_mean(cross_entropy, axis=2)
        cross_entropy *= mask
        cross_entropy = tf.reduce_mean(cross_entropy, axis=1)
        loss = tf.reduce_mean(cross_entropy)
        # attention reg.
        reg = 1. - tf.reduce_sum(att_alphas, axis=0)
        reg = tf.pow(reg, 2)
        reg = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(reg)))
        loss += reg
        return loss
    
    def _build_optimize(self):
        print('build optimize')
        clip_value = 1.
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables), clip_value)
        return self.optimizer.apply_gradients(zip(grads, trainable_variables))
    
    def _build_accuracy(self):
        print('build accuracy')
        caption = tf.pad(self.caption, [[0,0],[0,1]]) # padding one more step
        caption = caption[:,1:]
        correct = tf.equal(caption, tf.cast(tf.argmax(self.pred, 2), tf.int32))
        correct = tf.cast(correct, tf.float32)
        mask = tf.cast(tf.not_equal(caption, 0), tf.float32)
        mask /= tf.reduce_mean(tf.reduce_mean(mask))
        correct *= mask
        return tf.reduce_mean(tf.reduce_mean(correct))
    
    def fit(self, train, valid=None, ground_truth_prob=1., ground_truth_prob_decay=0.99 ,num_epochs=10, batch_size=32, eval_every=1, shuffle=False, save_min_loss=False, id2word=None):
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
                self.sess.run(self.optimize, feed_dict={self.inputs: batch_x, 
                                                        self.caption: batch_y, 
                                                        self.batch_size: len(batch_x), 
                                                        self.ground_truth_prob: ground_truth_prob})
                loss, acc = self.evaluate(batch_x, batch_y, batch_size=batch_size)
                print('epoch:{:>2d}/{:<2d}  batch:{:>4d}/{:<4d}  gt_prob:{:<.3f}  '.format(epoch+1, 
                                                                                      num_epochs, 
                                                                                      step*batch_size, 
                                                                                      num_steps*batch_size, 
                                                                                      ground_truth_prob), 
                                                                                      end='')
                print('loss:{:<3.5f}  acc:{:>3.1f}%  '.format(loss, 100*acc), end='')
                # evaluation
                if step % eval_every == 0 and valid is not None:
                    valid_x = np.array(valid[0], dtype='float32')
                    valid_y = np.array(valid[1])
                    valid_y = np.array([y[np.random.randint(len(y), size=1)[0]] for y in valid_y], dtype='int32')
                    val_loss, val_acc = self.evaluate(valid_x, valid_y, batch_size=batch_size)
                    print('val_loss:{:<3.5f}  val_acc:{:>3.1f}%  '.format(val_loss, 100*val_acc), end='')
                    # save_min_loss
                    if save_min_loss and (min_loss == 0. or val_loss < min_loss):
                        min_loss = val_loss
                        self.save('./models/att_reg_best', verbose=False)
                        print('save min loss model  '.format(min_loss), end='')
                    # visaul
                    if id2word is not None:
                        sample_idx = np.random.randint(len(valid_x))
                        visual_x = self.predict(valid_x[sample_idx:sample_idx+1])
                        visual_y = valid_y[sample_idx:sample_idx+1]
                        print()
                        print('    visual_model: {}'.format(self.visual(visual_x, id2word)[0]))
                        print('    visual_truth: {}  '.format(self.visual(visual_y, id2word)[0]), end='')
                print()
            # update prob
            ground_truth_prob = max(ground_truth_prob*ground_truth_prob_decay, 0.5)
    
    def evaluate(self, x, y, batch_size=32):
        losses, accs = [], []
        offset = 0
        while offset < len(x):
            batch_x = x[offset : offset+batch_size]
            batch_y = y[offset : offset+batch_size]
            loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict={self.inputs: batch_x, 
                                                                             self.caption: batch_y, 
                                                                             self.batch_size: len(batch_x), 
                                                                             self.ground_truth_prob: 0.})
            losses += [loss] * len(batch_x)
            accs += [acc] * len(batch_x)
            offset += batch_size
        return np.array(losses).mean(), np.array(accs).mean()
    
    def predict(self, x, batch_size=32):
        preds = []
        offset = 0
        while offset < len(x):
            batch_x = x[offset : offset+batch_size]
            batch_y = np.zeros([len(batch_x), self._decode_steps], dtype='int32')
            pred = self.sess.run(self.pred, feed_dict={self.inputs: batch_x, 
                                                          self.caption: batch_y, 
                                                          self.batch_size: len(batch_x), 
                                                          self.ground_truth_prob: 0.})
            preds.append(np.argmax(pred, axis=2))
            offset += batch_size
        return np.vstack(preds)
    
    def visual(self, ys, id2word):
        results = []
        for pred in ys:
            pred_words = np.vectorize(id2word.get)(pred)
            statr_idx = np.where(pred_words=='<bos>')[0][0] if '<bos>' in pred_words else 0
            end_idx = np.where(pred_words=='<eos>')[0][0] if '<eos>' in pred_words else len(pred_words)
            sentence = ' '.join(pred_words[statr_idx:end_idx+1])
            sentence = sentence.replace('<bos>','').replace('<eos>','').strip()
            results.append(sentence)
        return results
    
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

model = Seq2seq(feature_dim, vocab_size, 256, max_frame_len, max_sent_len, load_model_path=None)
model.summary()

if run_train:
    try:
        model.fit(train=[train_X[:], train_Ys[:]], 
                  valid=[train_X[-50:], train_Ys[-50:]], 
                  num_epochs=200, 
                  batch_size=64,
                  ground_truth_prob=1., 
                  ground_truth_prob_decay=0.997,
                  shuffle=True,
                  eval_every=1,
                  save_min_loss=False,
                  id2word=id2word)
    except KeyboardInterrupt: # ctrl + c
        pass
    model.save('./tmp/att_reg_finish')
 
# output test
test_X, test_video_ids = load_test_data(fpath_test_data, fpath_test_ids)
preds = model.predict(test_X)
preds = model.visual(preds, id2word)
with open(fpath_test_output, 'w') as o:
    for sent, video_id in zip(preds, test_video_ids):
        _ = o.write('{},{}\n'.format(video_id, sent))

