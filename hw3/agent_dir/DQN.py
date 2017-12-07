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
        replace_target_iter=100,
        memory_size=1000,
        batch_size=32,
        epsilon_increment=None,
        output_graph=False,
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
        self.memory_s = []
        self.memory_a = []
        self.memory_r = []
        self.memory_s_ = []

        # total learning step
        self.learn_step_counter = 0

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # vars
        self.vars = {var.name: var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
        self.saver = tf.train.Saver(self.vars)

        # session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None] + self.n_features, name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None] + self.n_features, name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            q1 = tf.layers.conv2d(
                inputs=tf.expand_dims(self.s, -1), 
                filters=32, 
                kernel_size=(4, 4), 
                strides=(2, 2), 
                padding='valid', 
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                name='q1_conv_1'
            )
            shape = q1.get_shape().as_list()
            q1 = tf.reshape(q1, shape=[-1, shape[1]*shape[2]*shape[3]], name='flatten')
            self.q_eval = tf.layers.dense(
                inputs=q1, 
                units=self.n_actions, 
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='q'
            )

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 =  tf.layers.conv2d(
                inputs=tf.expand_dims(self.s_, -1), 
                filters=32, 
                kernel_size=(4, 4), 
                strides=(2, 2), 
                padding='valid', 
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                name='t1_conv_1'
            )
            shape = t1.get_shape().as_list()
            t1 = tf.reshape(t1, shape=[-1, shape[1]*shape[2]*shape[3]], name='flatten')
            self.q_next = tf.layers.dense(
                inputs=t1, 
                units=self.n_actions, 
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='t2'
            )

        with tf.variable_scope('q_target'):
            q_target = self.r + self.reward_decay * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_sum(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        self.memory_s.append(s)
        self.memory_a.append(a)
        self.memory_r.append(r)
        self.memory_s_.append(s_)

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
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        sample_index = np.random.choice(len(self.memory_s), size=self.batch_size)

        self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: np.array(self.memory_s[sample_index]),
                self.a: np.array(self.memory_a[sample_index]),
                self.r: np.array(self.memory_r[sample_index]),
                self.s_: np.array(self.memory_s_[sample_index]),
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
