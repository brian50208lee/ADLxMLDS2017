import numpy as np
import tensorflow as tf
import os

class DeepQNetwork(object):
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.001,
        reward_decay=0.99,
        epsilon_max=0.9,
        epsilon_increment=None,
        replace_target_iter=1000,
        memory_size=1000,
        batch_size=32,
    ):  
        # params
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.reward_decay = reward_decay
        self.replace_target_iter = replace_target_iter
        self.batch_size = batch_size

        # epsilon
        self.epsilon_max = epsilon_max
        self.epsilon_increment = epsilon_increment
        self.epsilon = 0.0 if epsilon_increment is not None else self.epsilon_max

        # initialize memory [s, a, r, s_]
        self.memory_size = memory_size
        self.memory_counter = 0
        self.memory_s = np.zeros([self.memory_size] + list(self.n_features))
        self.memory_a = np.zeros((self.memory_size,))
        self.memory_r = np.zeros((self.memory_size,))
        self.memory_s_ = np.zeros([self.memory_size] + list(self.n_features))

        # total learning step
        self.learn_step_counter = 0

        # consist of [target_net, evaluate_net]
        self._build_model()
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

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
                inputs=tf.expand_dims(net, -1), 
                filters=16, 
                kernel_size=(3, 3), 
                strides=(2, 2), 
                padding='valid', 
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                name=name+'_conv_1'
            )
            print(net.name, net.shape)
            net = tf.layers.conv2d(
                inputs=net, 
                filters=32, 
                kernel_size=(3, 3), 
                strides=(2, 2), 
                padding='valid',
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                name=name+'_conv_2'
            )
            print(net.name, net.shape)
            shape = net.get_shape().as_list()
            net = tf.reshape(net, shape=[-1, shape[1]*shape[2]*shape[3]], name=name+'_flatten')
            print(net.name, net.shape)
            net = tf.layers.dense(
                inputs=net, 
                units=128,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name=name+'_fc1'
            )
            print(net.name, net.shape)
            net = tf.layers.dense(
                inputs=net, 
                units=self.n_actions, 
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name=name+'_fc2'
            )
            print(net.name, net.shape)
            return net

    def _build_model(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None] + self.n_features, name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None] + self.n_features, name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        # ------------------ build net ------------------
        self.q_eval = self._net(self.s, 'eval_net')
        self.q_next = self._net(self.s_, 'target_net')


        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        
        with tf.variable_scope('q_target'):
            q_target = self.r + self.reward_decay * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)

        # ------------------ build loss ------------------
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_sum(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        idx = self.memory_counter % self.memory_size
        self.memory_s[idx] = np.array(s)
        self.memory_a[idx] = a
        self.memory_r[idx] = r
        self.memory_s_[idx] = np.array(s)
        self.memory_counter += 1

        # replace old
        if len(self.memory_s) > self.memory_size:
            self.memory_s[:1] = []
            self.memory_a[:1] = []
            self.memory_r[:1] = []
            self.memory_s_[:1] = []

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('target_params_replaced')

        # sample batch memory from all memory
        sample_index = np.random.choice(min(self.memory_counter, self.memory_size), size=self.batch_size)
        self.sess.run(
            self.train_op,
            feed_dict={
                self.s: self.memory_s[sample_index],
                self.a: self.memory_a[sample_index],
                self.r: self.memory_r[sample_index],
                self.s_: self.memory_s_[sample_index]
            })

        # increasing epsilon
        self.epsilon = min(self.epsilon+self.epsilon_increment, self.epsilon_max)
        self.learn_step_counter += 1

    def save(self, checkpoint_file_path):
        if not os.path.exists(os.path.dirname(checkpoint_file_path)):
            os.makedirs(os.path.dirname(checkpoint_file_path))
        self.saver.save(self.sess, checkpoint_file_path)
        print('Model saved to: {}'.format(checkpoint_file_path))
    
    def load(self, checkpoint_file_path):
        self.saver.restore(self.sess, checkpoint_file_path)
        print('Model restored from: {}'.format(checkpoint_file_path))
