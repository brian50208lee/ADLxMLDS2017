import os
import numpy as np
import tensorflow as tf
from scipy import stats

class BasicGAN(object):
    def __init__(
        self,
        inputs_shape,
        seq_vec_len,
        noise_len=100,
        optimizer=tf.train.RMSPropOptimizer,
        learning_rate=0.0001,
        output_graph_path=None
    ):  
        # params
        self.inputs_shape = inputs_shape
        self.seq_vec_len = seq_vec_len
        self.noise_len = noise_len
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.output_graph_path = output_graph_path

        # model
        self._build_placeholder()
        self._build_model()
        self._build_loss()
        self._build_optimize()

        # noise sampler
        self.noise_sampler = stats.truncnorm(0., 1., loc=0.5, scale=0.5)

        # saver
        self.saver = tf.train.Saver(tf.global_variables())

        # session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        # log
        self._build_summary()

    def _build_placeholder(self):
        self.g_noise = tf.placeholder(tf.float32, [None, self.noise_len], name='generative_noise')
        
        self.r_seq = tf.placeholder(tf.float32, [None, self.seq_vec_len], name='real_sequence')
        self.w_seq = tf.placeholder(tf.float32, [None, self.seq_vec_len], name='wrong_sequence')

        self.r_img = tf.placeholder(tf.float32, [None] + list(self.inputs_shape), name='real_image')
        self.w_img = tf.placeholder(tf.float32, [None] + list(self.inputs_shape), name='wrong_image')

    def _net_generative(self, seq, noise):
        raise NotImplementedError()    

    def _net_discriminative(self, seq, img, reuse=False):
        raise NotImplementedError()

    def _build_model(self):
        with tf.variable_scope('generative_net'):
            self.g_net = self._net_generative(self.r_seq, self.g_noise)
            self.f_img = self.g_net
        with tf.variable_scope('discriminative_net'):
            self.d_net_rr = self._net_discriminative(self.r_seq, self.r_img)
            self.d_net_rf = self._net_discriminative(self.r_seq, self.f_img, reuse=True)
            self.d_net_wr = self._net_discriminative(self.w_seq, self.r_img, reuse=True)
            self.d_net_rw = self._net_discriminative(self.r_seq, self.w_img, reuse=True)

    def _build_loss(self):
        def cross_entropy_with_logits(logits, labels):
            epsilon = tf.constant(value=1e-08)
            logits += epsilon
            cross_entropy = -(labels * tf.log(logits))
            return cross_entropy

        with tf.variable_scope('loss'):
            self.g_loss = tf.reduce_mean(cross_entropy_with_logits(logits=self.d_net_rf, labels=tf.ones_like(self.d_net_rf))) 
            self.d_loss = tf.reduce_mean(cross_entropy_with_logits(logits=self.d_net_rr, labels=tf.ones_like(self.d_net_rr))) \
                        + (tf.reduce_mean(cross_entropy_with_logits(logits=self.d_net_rf, labels=tf.zeros_like(self.d_net_rf))) + \
                           tf.reduce_mean(cross_entropy_with_logits(logits=self.d_net_wr, labels=tf.zeros_like(self.d_net_wr))) + \
                           tf.reduce_mean(cross_entropy_with_logits(logits=self.d_net_rw, labels=tf.zeros_like(self.d_net_rw)))) / 3 

    def _build_optimize(self):
        with tf.variable_scope('train_op'):
            g_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generative_net')
            d_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminative_net')
            self.g_train_op = self.optimizer(self.learning_rate).minimize(self.g_loss, var_list=g_params)
            self.d_train_op = self.optimizer(self.learning_rate).minimize(self.d_loss, var_list=d_params)

    def _build_summary(self):
        if self.output_graph_path:
            #self.reward_hist = tf.placeholder(tf.float32, [None], name='reward_hist')
            tf.summary.image('f_img', self.f_img, max_outputs=100)
            self.summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.output_graph_path, self.sess.graph)

    def train(self, train, max_batch_num=300000, valid_sents=None, batch_size=128, summary_every=1000):
        imgs, seqs = train
        for batch in range(max_batch_num):
            r_idx = np.random.choice(len(imgs), size=batch_size, replace=False)
            w_idx = np.random.choice(len(imgs), size=batch_size, replace=False)
            _, _, _, d_loss, g_loss = self.sess.run([self.d_train_op, self.g_train_op, self.g_train_op, self.d_loss, self.g_loss],
                                                  feed_dict={
                                                        self.g_noise: self.noise_sampler.rvs([batch_size, self.noise_len]),
                                                        self.r_seq: seqs[r_idx],
                                                        self.r_img: imgs[r_idx],
                                                        self.w_seq: seqs[w_idx],
                                                        self.w_img: imgs[w_idx]
                                                  })
            print('batch:{} d_loss: {} g_loss: {}'.format(batch, d_loss, g_loss))
            if valid_sents is not None and batch % summary_every == 0:
                test_seqs = valid_sents
                feed_dict = {
                    self.g_noise: self.noise_sampler.rvs([len(test_seqs), self.noise_len]),
                    self.r_seq: test_seqs
                }
                self.summary(batch, feed_dict)

    def summary(self, step, feed_dict):
        if self.output_graph_path:
            result = self.sess.run(self.summary_op, feed_dict=feed_dict)
            self.summary_writer.add_summary(result, step)

    def save(self, checkpoint_file_path):
        if not os.path.exists(os.path.dirname(checkpoint_file_path)):
            os.makedirs(os.path.dirname(checkpoint_file_path))
        self.saver.save(self.sess, checkpoint_file_path)
        print('Model saved to: {}'.format(checkpoint_file_path))
    
    def load(self, checkpoint_file_path):
        self.saver.restore(self.sess, checkpoint_file_path)
        print('Model restored from: {}'.format(checkpoint_file_path))


class GAN(BasicGAN):
    def leaky_relu(self, x, alpha=0.2):
        return tf.maximum(tf.minimum(0.0, alpha * x), x)

    def _net_generative(self, seq, noise):
        net = tf.concat([seq, noise], axis=1, name='noise_vector')
        print(net.name, net.shape)
        net = tf.layers.dense(
            inputs=net, 
            units=4*4*256,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='fc1'
        )
        net = tf.reshape(net, [-1, 4, 4, 256])
        print(net.name, net.shape)
        net = tf.layers.conv2d_transpose(
            inputs=net, 
            filters=128, 
            kernel_size=(5, 5), 
            strides=(2, 2), 
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name='conv2'
        )
        print(net.name, net.shape)
        net = tf.layers.conv2d_transpose(
            inputs=net, 
            filters=64, 
            kernel_size=(5, 5), 
            strides=(2, 2), 
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name='conv3'
        )
        print(net.name, net.shape)
        net = tf.layers.conv2d_transpose(
            inputs=net, 
            filters=32, 
            kernel_size=(5, 5), 
            strides=(2, 2), 
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name='conv4'
        )
        print(net.name, net.shape)
        net = tf.layers.conv2d_transpose(
            inputs=net, 
            filters=3, 
            kernel_size=(5, 5), 
            strides=(2, 2), 
            padding='same',
            activation=tf.nn.sigmoid,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name='conv5'
        )
        print(net.name, net.shape)
        return net

    def _net_discriminative(self, seq, img, reuse=False):
        if reuse == True:
           tf.get_variable_scope().reuse_variables()

        net = img
        print(net.name, net.shape)
        net = tf.layers.conv2d(
            inputs=net, 
            filters=32, 
            kernel_size=(5, 5), 
            strides=(2, 2), 
            padding='same',
            activation=self.leaky_relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name='conv1'
        )
        print(net.name, net.shape)
        net = tf.layers.conv2d(
            inputs=net, 
            filters=64, 
            kernel_size=(5, 5), 
            strides=(2, 2), 
            padding='same',
            activation=self.leaky_relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name='conv2'
        )
        print(net.name, net.shape)
        net = tf.layers.conv2d(
            inputs=net, 
            filters=128, 
            kernel_size=(5, 5), 
            strides=(2, 2), 
            padding='same',
            activation=self.leaky_relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name='conv3'
        )
        print(net.name, net.shape)
        seq_vectors = tf.expand_dims(tf.expand_dims(seq, 1), 2)
        seq_vectors = tf.tile(seq_vectors, [1, 8, 8, 1])
        net = tf.concat([net, seq_vectors], axis=-1, name='concat_condition')
        print(net.name, net.shape)
        net = tf.layers.conv2d(
            inputs=net, 
            filters=128, 
            kernel_size=(1, 1), 
            strides=(1, 1), 
            padding='same',
            activation=self.leaky_relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name='conv4'
        )
        print(net.name, net.shape)
        net = tf.layers.conv2d(
            inputs=net, 
            filters=1, 
            kernel_size=(8, 8), 
            strides=(1, 1), 
            padding='valid',
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name='conv5'
        )
        print(net.name, net.shape)
        net = tf.squeeze(net, [1, 2, 3], name='squeeze')
        print(net.name, net.shape)

        return net

