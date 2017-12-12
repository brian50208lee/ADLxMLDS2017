import os
import numpy as np
import tensorflow as tf

class BasicDeepQNetwork(object):
    def __init__(
        self,
        inputs_shape,
        n_actions,
        gamma=0.99,
        optimizer=tf.train.AdamOptimizer,
        learning_rate=0.0001,
        batch_size=32,
        memory_size=10000,
        output_graph_path=None
    ):  
        # params
        self.inputs_shape = inputs_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.output_graph_path = output_graph_path

        # initialize memory [s, a, r, s_]
        self.memory_counter = 0
        self.memory_s = np.zeros((self.memory_size,) + tuple(self.inputs_shape))
        self.memory_a = np.zeros((self.memory_size,))
        self.memory_r = np.zeros((self.memory_size,))
        self.memory_s_ = np.zeros((self.memory_size,) + tuple(self.inputs_shape))

        # model
        self._build_placeholder()
        self._build_model()
        self._build_loss()
        self._build_optimize()
        self._build_replacement()

        # saver
        self.vars = {var.name: var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
        self.saver = tf.train.Saver(self.vars)

        # session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # log
        self._build_summary()

    def _build_placeholder(self):
        self.s = tf.placeholder(tf.float32, [None] + list(self.inputs_shape), name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None] + list(self.inputs_shape), name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None], name='a')  # input Action
     
    def _net(self, inputs):
        raise NotImplementedError()

    def _build_model(self):
        with tf.variable_scope('online_net'):
            self.online_net = self._net(self.s)
        with tf.variable_scope('target_net'):
            self.target_net = self._net(self.s_)

    def _build_loss(self):
        with tf.variable_scope('loss'):
            action_one_hot = tf.one_hot(self.a, self.n_actions)
            q_eval = tf.reduce_sum(self.online_net * action_one_hot, axis=1, name='q_eval')
            self.q_target = self.r + self.gamma * tf.reduce_max(self.target_net, axis=1, name='q_target')
            self.q_target = tf.stop_gradient(self.q_target)
            self.loss = tf.reduce_mean(tf.square(self.q_target - q_eval), name='loss_mse')

    def _build_optimize(self):
        with tf.variable_scope('train_op'):            
            clip_value = 1.
            trainable_variables = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables), clip_value)
            self.train_op = self.optimizer(self.learning_rate).minimize(self.loss)

    def _build_replacement(self):
        o_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='online_net')
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        with tf.variable_scope('replacement'):
            self.replace_target_net_op = [tf.assign(t, o) for t, o in zip(t_params, o_params)]

    def _build_summary(self):
        if self.output_graph_path:
            self.reward_hist = tf.placeholder(tf.float32, [None], name='reward_hist')
            tf.summary.scalar('min_reward', tf.reduce_min(self.reward_hist))
            tf.summary.scalar('max_reward', tf.reduce_max(self.reward_hist))
            tf.summary.scalar('avg_reward', tf.reduce_mean(self.reward_hist))
            tf.summary.scalar('max_q', tf.reduce_max(self.q_target))
            tf.summary.scalar('loss_mse', self.loss)
            self.summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.output_graph_path, self.sess.graph)

    def store_transition(self, s, a, r, s_):
        idx = self.memory_counter % self.memory_size
        self.memory_s[idx] = np.array(s)
        self.memory_a[idx] = a
        self.memory_r[idx] = r
        self.memory_s_[idx] = np.array(s_)
        self.memory_counter += 1

    def learn(self):
        sample_index = np.random.choice(min(self.memory_counter, self.memory_size), size=self.batch_size)
        self.sess.run(self.train_op,
                      feed_dict={
                            self.s: self.memory_s[sample_index],
                            self.a: self.memory_a[sample_index],
                            self.r: self.memory_r[sample_index],
                            self.s_: self.memory_s_[sample_index]
                      })

    def summary(self, step, reward_hist):
        if self.output_graph_path:
            sample_index = np.random.choice(min(self.memory_counter, self.memory_size), size=self.batch_size)
            result = self.sess.run(self.summary_op,
                                   feed_dict={
                                        self.reward_hist: np.array(reward_hist[-self.batch_size:]),
                                        self.s: self.memory_s[sample_index],
                                        self.a: self.memory_a[sample_index],
                                        self.r: self.memory_r[sample_index],
                                        self.s_: self.memory_s_[sample_index]
                                   })
            self.summary_writer.add_summary(result, step)

    def replace_target_net(self):
        self.sess.run(self.replace_target_net_op)
        print('replace target net')

    def choose_action(self, observation):
        actions_value = self.sess.run(self.online_net, feed_dict={self.s: observation[np.newaxis, :]})
        action = np.argmax(actions_value)
        return action

    def save(self, checkpoint_file_path):
        if not os.path.exists(os.path.dirname(checkpoint_file_path)):
            os.makedirs(os.path.dirname(checkpoint_file_path))
        self.saver.save(self.sess, checkpoint_file_path)
        print('Model saved to: {}'.format(checkpoint_file_path))
    
    def load(self, checkpoint_file_path):
        self.saver.restore(self.sess, checkpoint_file_path)
        print('Model restored from: {}'.format(checkpoint_file_path))



class DeepQNetwork(BasicDeepQNetwork):
    def _net(self, inputs):
        net = inputs
        print(net.name, net.shape)
        net = tf.layers.conv2d(
            inputs=net, 
            filters=32,
            kernel_size=(8, 8), 
            strides=(4, 4), 
            padding='valid', 
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), 
            name='conv1'
        )
        print(net.name, net.shape)
        net = tf.layers.conv2d(
            inputs=net, 
            filters=64, 
            kernel_size=(4, 4), 
            strides=(2, 2), 
            padding='valid',
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), 
            name='conv2'
        )
        print(net.name, net.shape)
        net = tf.layers.conv2d(
            inputs=net, 
            filters=64, 
            kernel_size=(3, 3), 
            strides=(1, 1), 
            padding='valid',
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), 
            name='conv3'
        )
        print(net.name, net.shape)
        net = tf.contrib.layers.flatten(net, scope='flatten')
        print(net.name, net.shape)
        net = tf.layers.dense(
            inputs=net, 
            units=512,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='fc4'
        )
        print(net.name, net.shape)
        net = tf.layers.dense(
            inputs=net, 
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='fc5'
        )
        print(net.name, net.shape)
        return net

class DoubleDeepQNetwork(DeepQNetwork):
    def _build_loss(self):
        with tf.variable_scope('loss'):
            # eval q
            action_one_hot = tf.one_hot(self.a, self.n_actions)
            q_eval = tf.reduce_sum(self.online_net * action_one_hot, axis=1, name='q_eval')
            # target q
            action_eval = tf.argmax(self.online_net, axis=1)
            action_eval_one_hot = tf.one_hot(action_eval, self.n_actions)
            self.q_target = self.r + self.gamma * tf.reduce_sum(self.target_net * action_eval_one_hot, axis=1, name='q_target')
            self.q_target = tf.stop_gradient(self.q_target)
            # loss
            self.loss = tf.reduce_mean(tf.square(self.q_target - q_eval), name='loss_mse') 
