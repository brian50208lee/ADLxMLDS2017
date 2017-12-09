import os
import numpy as np
import tensorflow as tf


class DeepQNetwork(object):
    def __init__(
        self,
        n_actions,
        inputs_shape,
        learning_rate=0.0001,
        discount=0.99,
        memory_size=10000,
        batch_size=32
    ):  
        # params
        self.n_actions = n_actions
        self.inputs_shape = inputs_shape
        self.lr = learning_rate
        self.discount = discount
        self.batch_size = batch_size

        # initialize memory [s, a, r, s_]
        self.memory_size = memory_size
        self.memory_counter = 0
        self.memory_s = np.zeros((self.memory_size,) + tuple(self.inputs_shape))
        self.memory_a = np.zeros((self.memory_size,))
        self.memory_r = np.zeros((self.memory_size,))
        self.memory_s_ = np.zeros((self.memory_size,) + tuple(self.inputs_shape))

        # consist of [target_net, evaluate_net]
        self._build_model()
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        with tf.variable_scope('soft_replacement'):
            self.update_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # vars
        self.vars = {var.name: var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
        self.saver = tf.train.Saver(self.vars)

        # session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _net(self, inputs, name):
        with tf.variable_scope(name):
            net = inputs
            print(net.name, net.shape)
            net = tf.layers.conv2d(
                inputs=net, 
                filters=32,
                kernel_size=(8, 8), 
                strides=(4, 4), 
                padding='valid', 
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                name=name+'_conv1'
            )
            print(net.name, net.shape)
            net = tf.layers.conv2d(
                inputs=net, 
                filters=32, 
                kernel_size=(4, 4), 
                strides=(2, 2), 
                padding='valid',
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                name=name+'_conv2'
            )
            print(net.name, net.shape)
            shape = net.get_shape().as_list()
            net = tf.reshape(net, shape=[-1, shape[1]*shape[2]*shape[3]], name=name+'_flatten')
            print(net.name, net.shape)
            net = tf.layers.dense(
                inputs=net, 
                units=self.n_actions,
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name=name+'_fc3'
            )
            print(net.name, net.shape)
            return net

    def _build_model(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None] + list(self.inputs_shape), name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None] + list(self.inputs_shape), name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        # ------------------ build net ------------------
        self.q_eval = self._net(self.s, 'eval_net')
        self.q_next = self._net(self.s_, 'target_net')


        with tf.variable_scope('q_eval'):
            action_one_hot = tf.one_hot(self.a, self.n_actions)
            self.q_eval_wrt_a = tf.reduce_sum(self.q_eval * action_one_hot, reduction_indices=1)    # shape=(None, )
        
        with tf.variable_scope('q_target'):
            q_target = self.r + self.discount * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)

        # ------------------ build loss ------------------
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.square(self.q_target - self.q_eval_wrt_a), name='loss')
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        idx = self.memory_counter % self.memory_size
        self.memory_s[idx] = np.array(s)
        self.memory_a[idx] = a
        self.memory_r[idx] = r
        self.memory_s_[idx] = np.array(s_)
        self.memory_counter += 1

        # replace old
        if len(self.memory_s) > self.memory_size:
            self.memory_s[:1] = []
            self.memory_a[:1] = []
            self.memory_r[:1] = []
            self.memory_s_[:1] = []

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)
        return action

    def learn(self):
        sample_index = np.random.choice(min(self.memory_counter, self.memory_size), size=self.batch_size)
        self.sess.run(
            self.train_op,
            feed_dict={
                self.s: self.memory_s[sample_index],
                self.a: self.memory_a[sample_index],
                self.r: self.memory_r[sample_index],
                self.s_: self.memory_s_[sample_index]
            })

    def update_target(self):
        self.sess.run(self.update_target_op)
        print('update_target_op')

    def save(self, checkpoint_file_path):
        if not os.path.exists(os.path.dirname(checkpoint_file_path)):
            os.makedirs(os.path.dirname(checkpoint_file_path))
        self.saver.save(self.sess, checkpoint_file_path)
        print('Model saved to: {}'.format(checkpoint_file_path))
    
    def load(self, checkpoint_file_path):
        self.saver.restore(self.sess, checkpoint_file_path)
        print('Model restored from: {}'.format(checkpoint_file_path))
