import os
import numpy as np
import tensorflow as tf
from scipy import stats

class BasicGAN(object):
    def __init__(
        self,
        inputs_shape,
        seq_vec_len,
        noise_len=50,
        g_optimizer=tf.train.RMSPropOptimizer(learning_rate=0.0001),
        d_optimizer=tf.train.RMSPropOptimizer(learning_rate=0.0001),
        summary_path=None
    ):  
        # params
        self.inputs_shape = inputs_shape
        self.seq_vec_len = seq_vec_len
        self.noise_len = noise_len
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.summary_path = summary_path

        # model
        tf.reset_default_graph()
        self._build_placeholder()
        self._build_model()
        self._build_loss()
        self._build_optimize()
        self._build_clip_parms() # WGAN

        # noise sampler
        self.noise_sampler = stats.truncnorm(0.0, 1.0, loc=0.5, scale=1.0)

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
            self.f_img = self.g_net # fake image
        with tf.variable_scope('discriminative_net'):
            self.d_net_rr = self._net_discriminative(self.r_seq, self.r_img) # real sequence, real image -> 1
            self.d_net_rf = self._net_discriminative(self.r_seq, self.f_img, reuse=True) # real sequence, fake image -> 0
            self.d_net_wr = self._net_discriminative(self.w_seq, self.r_img, reuse=True) # wrong sequence, real image -> 0
            self.d_net_rw = self._net_discriminative(self.r_seq, self.w_img, reuse=True) # real sequence, wrong image -> 0

    def _build_loss(self):
        with tf.variable_scope('loss'):
            # GAN loss
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_net_rf, labels=tf.ones_like(self.d_net_rf))) 
            self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_net_rr, labels=tf.ones_like(self.d_net_rr))) \
                        + (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_net_rf, labels=tf.zeros_like(self.d_net_rf))) + \
                           tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_net_wr, labels=tf.zeros_like(self.d_net_wr))) + \
                           tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_net_rw, labels=tf.zeros_like(self.d_net_rw)))) / 3 
            '''
            # WGAN loss
            self.g_loss = -tf.reduce_mean(self.d_net_rf)
            self.d_loss = -tf.reduce_mean(self.d_net_rr) \
                        + (tf.reduce_mean(self.d_net_rf) + \
                           tf.reduce_mean(self.d_net_wr) + \
                           tf.reduce_mean(self.d_net_rw)) / 3
            '''
    
    def _build_optimize(self):
        with tf.variable_scope('train_op'):
            self.g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generative_net')
            self.d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminative_net')
            self.g_train_op = self.g_optimizer.minimize(self.g_loss, var_list=self.g_vars)
            self.d_train_op = self.d_optimizer.minimize(self.d_loss, var_list=self.d_vars)
            
    def _build_clip_parms(self):
        # WGAN clip
        with tf.variable_scope('d_clip_op'):
            self.d_clip_op = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d_vars]

    def _build_summary(self):
        if self.summary_path:
            fake_img = tf.image.resize_images(self.f_img, [64,64])
            tf.summary.image('fake_img', fake_img, max_outputs=100)
            self.summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.summary_path, self.sess.graph)

    def train(self, train, max_batch_num=300000, valid_seqs=None, batch_size=64, summary_every=100):
        imgs, seqs = train
        for batch in range(max_batch_num):
            r_idx = np.random.choice(len(imgs), size=batch_size, replace=False) # real
            w_idx = np.random.choice(len(imgs), size=batch_size, replace=False) # wrong
            # train d_net : g_net = 1 : 2
            #print(self.noise_sampler.rvs([batch_size, self.noise_len][0]))
            _, _, _, d_loss, g_loss = self.sess.run([self.d_train_op, self.g_train_op, self.g_train_op, self.d_loss, self.g_loss],
                                                    feed_dict={
                                                        self.g_noise: self.noise_sampler.rvs([batch_size, self.noise_len]),
                                                        self.r_seq: seqs[r_idx],
                                                        self.r_img: imgs[r_idx],
                                                        self.w_seq: seqs[w_idx],
                                                        self.w_img: imgs[w_idx]
                                                    })
            print('batch:{} d_loss: {} g_loss: {}'.format(batch, d_loss, g_loss))
            if valid_seqs is not None and batch % summary_every == 0:
                self.summary(batch, valid_seqs)

    def summary(self, step, seqs):
        if self.summary_path:
            result = self.sess.run(self.summary_op, 
                                   feed_dict={
                                        self.g_noise: self.noise_sampler.rvs([len(seqs), self.noise_len]),
                                        self.r_seq: seqs
                                   })
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
            units=3*3*256,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='fc1'
        )
        net = tf.reshape(net, [-1, 3, 3, 256])
        print(net.name, net.shape)
        net = tf.layers.conv2d_transpose(
            inputs=net, 
            filters=256, 
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
            filters=128, 
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
            filters=64, 
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
            filters=32, 
            kernel_size=(5, 5), 
            strides=(2, 2), 
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name='conv5'
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
            name='conv6'
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
        net = tf.layers.conv2d(
            inputs=net, 
            filters=256, 
            kernel_size=(5, 5), 
            strides=(2, 2), 
            padding='same',
            activation=self.leaky_relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name='conv4'
        )
        print(net.name, net.shape)
        seq_vectors = tf.expand_dims(tf.expand_dims(seq, 1), 2)
        seq_vectors = tf.tile(seq_vectors, [1, 6, 6, 1])
        net = tf.concat([net, seq_vectors], axis=-1, name='concat_condition')
        print(net.name, net.shape)
        net = tf.layers.conv2d(
            inputs=net, 
            filters=256, 
            kernel_size=(1, 1), 
            strides=(1, 1), 
            padding='same',
            activation=self.leaky_relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name='conv5'
        )
        print(net.name, net.shape)
        net = tf.layers.conv2d(
            inputs=net, 
            filters=1, 
            kernel_size=(6, 6), 
            strides=(1, 1), 
            padding='valid',
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name='conv6'
        )
        print(net.name, net.shape)
        net = tf.squeeze(net, [1, 2, 3], name='squeeze')
        print(net.name, net.shape)

        return net

