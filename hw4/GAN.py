import os
import numpy as np
import tensorflow as tf

class BasicGAN(object):
    def __init__(
        self,
        inputs_shape,
        seq_vec_len,
        noise_len=50,
        g_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
        d_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
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

        # noise sampler
        self.noise_sampler = lambda size: np.random.normal(loc=0.0, scale=1.0, size=size)
        #self.noise_sampler = lambda size: np.random.uniform(low=0.0, high=1.0, size=size)

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
        self.training = tf.placeholder(tf.bool)
        
        self.g_noise = tf.placeholder(tf.float32, [None, self.noise_len], name='generative_noise')
        
        self.r_seq = tf.placeholder(tf.float32, [None, self.seq_vec_len], name='real_sequence')
        self.w_seq = tf.placeholder(tf.float32, [None, self.seq_vec_len], name='wrong_sequence')

        self.r_img = tf.placeholder(tf.float32, [None] + list(self.inputs_shape), name='real_image')
        self.w_img = tf.placeholder(tf.float32, [None] + list(self.inputs_shape), name='wrong_image')

    def _net_generative(self, seq, noise, training):
        raise NotImplementedError()    

    def _net_discriminative(self, seq, img, training):
        raise NotImplementedError()

    def _build_model(self):
        with tf.variable_scope('generative_net'):
            self.g_net = self._net_generative(self.r_seq, self.g_noise, self.training)
            self.f_img = self.g_net # fake image
        with tf.variable_scope('discriminative_net'):
            self.d_net_rr = self._net_discriminative(self.r_seq, self.r_img, self.training) # real sequence, real image -> 1
        with tf.variable_scope('discriminative_net', reuse=True):
            self.d_net_rf = self._net_discriminative(self.r_seq, self.f_img, self.training) # real sequence, fake image -> 0
        with tf.variable_scope('discriminative_net', reuse=True):
            self.d_net_wr = self._net_discriminative(self.w_seq, self.r_img, self.training) # wrong sequence, real image -> 0
        with tf.variable_scope('discriminative_net', reuse=True):
            self.d_net_rw = self._net_discriminative(self.r_seq, self.w_img, self.training) # real sequence, wrong image -> 0

    def _build_loss(self):
        with tf.variable_scope('loss'):
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_net_rf, labels=tf.ones_like(self.d_net_rf))) 
            self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_net_rr, labels=tf.ones_like(self.d_net_rr))) \
                        + (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_net_rf, labels=tf.zeros_like(self.d_net_rf))) + \
                           tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_net_wr, labels=tf.zeros_like(self.d_net_wr))) + \
                           tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_net_rw, labels=tf.zeros_like(self.d_net_rw)))) / 3 
    
    def _build_optimize(self):
        with tf.variable_scope('train_op'):
            self.g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generative_net')
            self.d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminative_net')
            self.g_train_op = self.g_optimizer.minimize(self.g_loss, var_list=self.g_vars)
            self.d_train_op = self.d_optimizer.minimize(self.d_loss, var_list=self.d_vars)
            
    def _build_summary(self):
        if self.summary_path:
            fake_img = tf.image.resize_images(self.f_img, [64,64])/2 + 0.5
            tf.summary.image('fake_img', fake_img, max_outputs=100)
            self.summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.summary_path, self.sess.graph)

    def train(self, train, valid_seqs=None, max_batch_num=150000, batch_size=64, summary_every=100):
        imgs, seqs = train
        d_iter, g_iter = 1, 1
        for batch in range(max_batch_num):
            g_iter = max(min(g_iter, 5), 1)
            for _ in range(d_iter): # discimenator iter
                r_idx = np.random.choice(len(imgs), size=batch_size, replace=False) # real
                w_idx = np.random.choice(len(imgs), size=batch_size, replace=False) # wrong
                g_noise = self.noise_sampler([batch_size, self.noise_len]) # noise
                _, d_loss = self.sess.run([self.d_train_op, self.d_loss],
                                          feed_dict={
                                                self.training: True,
                                                self.g_noise: g_noise,
                                                self.r_seq: seqs[r_idx],
                                                self.r_img: imgs[r_idx],
                                                self.w_seq: seqs[w_idx],
                                                self.w_img: imgs[w_idx]
                                          })
            for _ in range(g_iter): # generator iter
                r_idx = np.random.choice(len(imgs), size=batch_size, replace=False) # real
                w_idx = np.random.choice(len(imgs), size=batch_size, replace=False) # wrong
                g_noise = self.noise_sampler([batch_size, self.noise_len]) # noise
                _, g_loss = self.sess.run([self.g_train_op, self.g_loss],
                                          feed_dict={
                                                self.training: True,
                                                self.g_noise: g_noise,
                                                self.r_seq: seqs[r_idx],
                                                self.r_img: imgs[r_idx],
                                          })
            print('batch:{} d_loss: {} g_loss: {} g_iter: {}'.format(batch, d_loss, g_loss, g_iter))
            if batch % 100 == 0 and g_loss > 2: g_iter += 1
            if batch % 100 == 0 and g_loss < 1: g_iter -= 1
            if valid_seqs is not None and batch % summary_every == 0: # summary
                self.summary(step=batch, seqs=valid_seqs)

    def summary(self, step, seqs):
        if self.summary_path:
            r_idx = range(len(seqs)) # real
            g_noise = self.noise_sampler([len(seqs), self.noise_len]) # noise
            g_noise[0] = np.zeros([1, g_noise.shape[1]], dtype='float32') # without noise
            result = self.sess.run(self.summary_op, 
                                   feed_dict={
                                        self.training: True,
                                        self.g_noise: g_noise,
                                        self.r_seq: seqs[r_idx]
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
        return tf.maximum(tf.minimum(0.0, alpha*x), x)

    def img_condition_concat(self, tensor_img, tensor_seq):
        """
        tensor_img shape: [batch, height, width, depth]
        tensor_seq shape: [batch, seq_len]
        output shape: [batch, height, width, depth + seq_len]
        """
        img_shape = tensor_img.shape.as_list()[1:-1]
        img = tensor_img
        condition = tf.expand_dims(tf.expand_dims(tensor_seq, 1), 2)
        condition = tf.tile(condition, [1] + img_shape + [1])
        concat = tf.concat([img, condition], axis=3)
        return concat

    def _net_generative(self, seq, noise, training, use_bias=False):
        # --------- input ----------
        net = tf.expand_dims(tf.expand_dims(noise, 1), 2)
        net = tf.identity(net, name='input')
        print(net.name, net.shape)
        # --------- layer1 ----------
        net = tf.layers.conv2d_transpose(net, 512, (3, 3), strides=(1, 1), padding='valid', use_bias=True, name='deconv1')
        print(net.name, net.shape)
        # --------- concat ----------
        net = self.img_condition_concat(net, seq)
        net = tf.identity(net, name='concat_condition')
        print(net.name, net.shape)
        net = tf.nn.relu(net)
        # --------- layer2 ----------
        net = tf.layers.conv2d_transpose(net, 256, (5, 5), strides=(2, 2), padding='same', use_bias=use_bias, name='deconv2')
        print(net.name, net.shape)
        net = tf.nn.relu(net)
        # --------- layer3 ----------
        net = tf.layers.conv2d_transpose(net, 128, (5, 5), strides=(2, 2), padding='same', use_bias=use_bias, name='deconv3')
        print(net.name, net.shape)
        net = tf.nn.relu(net)
        # --------- layer4 ----------
        net = tf.layers.conv2d_transpose(net, 64, (5, 5), strides=(2, 2), padding='same', use_bias=use_bias, name='deconv4')
        print(net.name, net.shape)
        net = tf.nn.relu(net)
        # --------- layer5 ----------
        net = tf.layers.conv2d_transpose(net, 32, (5, 5), strides=(2, 2), padding='same', use_bias=use_bias, name='deconv5')
        print(net.name, net.shape)
        net = tf.nn.relu(net)
        # --------- layer6 ----------
        net = tf.layers.conv2d_transpose(net, 3, (5, 5), strides=(2, 2), padding='same', use_bias=use_bias, name='deconv6')
        print(net.name, net.shape)
        # --------- output ----------
        net = tf.nn.tanh(net)
        net = tf.identity(net, name='output')
        print(net.name, net.shape)

        return net

    def _net_discriminative(self, seq, img, training, use_bias=False):
        # --------- input ----------
        net = tf.identity(img, name='input')
        print(net.name, net.shape)
        # --------- layer1 ----------
        net = tf.layers.conv2d(net, 32, (5, 5), strides=(2, 2), padding='same', use_bias=use_bias, name='conv1')
        print(net.name, net.shape)
        net = tf.layers.batch_normalization(net, training=training)
        net = self.leaky_relu(net)
        # --------- layer2 ----------
        net = tf.layers.conv2d(net, 64, (5, 5), strides=(2, 2), padding='same', use_bias=use_bias, name='conv2')
        print(net.name, net.shape)
        net = tf.layers.batch_normalization(net, training=training)
        net = self.leaky_relu(net)
        # --------- layer3 ----------
        net = tf.layers.conv2d(net, 128, (5, 5), strides=(2, 2), padding='same', use_bias=use_bias, name='conv3')
        print(net.name, net.shape)
        net = tf.layers.batch_normalization(net, training=training)
        net = self.leaky_relu(net)
        # --------- layer4 ----------
        net = tf.layers.conv2d(net, 256, (5, 5), strides=(2, 2), padding='same', use_bias=use_bias, name='conv4')
        print(net.name, net.shape)
        net = tf.layers.batch_normalization(net, training=training)
        net = self.leaky_relu(net)
        # --------- concat ----------
        net = self.img_condition_concat(net, seq)
        net = tf.identity(net, name='concat_condition')
        print(net.name, net.shape)
        # --------- layer5 ----------
        net = tf.layers.conv2d(net, 256, (1, 1), strides=(1, 1), padding='same', use_bias=use_bias, name='conv5')
        print(net.name, net.shape)
        net = tf.layers.batch_normalization(net, training=training)
        net = self.leaky_relu(net)
        # --------- layer6 ----------
        final_shape = net.shape.as_list()[1:-1] # discrimenative
        net = tf.layers.conv2d(net, 1, final_shape, strides=(1, 1), padding='valid', use_bias=use_bias, name='conv6')
        print(net.name, net.shape)
        # --------- output ----------
        net = tf.squeeze(net, [1, 2, 3], name='output')
        print(net.name, net.shape)

        return net
