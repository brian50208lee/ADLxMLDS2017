import numpy as np
import tensorflow as tf
import os

class BasicPolicyGradient(object):
    def __init__(
        self, 
        inputs_shape, 
        n_actions,
        gamma=0.99,
        optimizer=tf.train.AdamOptimizer,
        learning_rate=0.0001,
        output_graph_path=None,
    ):
        # params
        self.inputs_shape = inputs_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.output_graph_path = output_graph_path

        # initial memory
        self.memory_s = []
        self.memory_a = []
        self.memory_r = []

        # build model
        self._build_placeholder()
        self._build_model()
        self._build_loss()
        self._build_optimize()

        # saver
        self.vars = {var.name: var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
        self.saver = tf.train.Saver(self.vars)

        # session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        # log
        self._build_summary()

    def _build_placeholder(self):
        with tf.name_scope('inputs'):
            self.s = tf.placeholder(tf.float32, [None] + list(self.inputs_shape), name="s") # State
            self.a = tf.placeholder(tf.int32, [None], name="a") # Action
            self.r = tf.placeholder(tf.float32, [None], name="r") # Reward

    def _net(self, inputs):
        raise NotImplementedError()

    def _build_model(self):
        with tf.variable_scope('network'):
            self.network_without_softmax = self._net(self.s)
            self.network = tf.nn.softmax(self.network_without_softmax, name='softmax')

    def _build_loss(self):
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.a, logits=self.network_without_softmax)
            self.loss = tf.reduce_sum(cross_entropy * self.r, name='loss')
            
    def _build_optimize(self):
        with tf.name_scope('train_op'):
            clip_value = 1.
            trainable_variables = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables), clip_value)
            self.train_op = self.optimizer(self.learning_rate).apply_gradients(zip(grads, trainable_variables))

    def _build_summary(self):
        if self.output_graph_path:
            with tf.name_scope('summary'):
                self.reward_hist = tf.placeholder(tf.float32, [None], name='reward_hist')
                tf.summary.scalar('min_reward', tf.reduce_min(self.reward_hist))
                tf.summary.scalar('max_reward', tf.reduce_max(self.reward_hist))
                tf.summary.scalar('avg_reward', tf.reduce_mean(self.reward_hist))
                tf.summary.scalar('loss', self.loss)
                self.summary_op = tf.summary.merge_all()
                self.summary_writer = tf.summary.FileWriter(self.output_graph_path, self.sess.graph)

    def _discount_rewards(self, memory_r):
        discounted_r = np.zeros_like(memory_r)
        running_add = 0.
        for t in reversed(range(len(memory_r))):
            if memory_r[t] != 0:
                running_add = 0. # reset the sum, since this was a game boundary
            running_add = running_add * self.gamma + memory_r[t]
            discounted_r[t] = running_add
        return discounted_r

    def _normalize_rewards(self, discounted_r):
        normalized_r = np.array(discounted_r)
        normalized_r -= np.mean(normalized_r)
        normalized_r /= np.std(normalized_r)
        return normalized_r

    def store_transition(self, s, a, r):
        self.memory_s.append(s)
        self.memory_a.append(a)
        self.memory_r.append(r)

    def clear_transition(self):
        self.memory_s = []
        self.memory_a = []
        self.memory_r = []

    def learn(self):
        # discounted and normalize reward
        discounted_norm_r = self.memory_r
        discounted_norm_r = self._discount_rewards(discounted_norm_r)
        discounted_norm_r = self._normalize_rewards(discounted_norm_r)
        # train on episode
        self.sess.run(self.train_op, 
                      feed_dict={
                            self.s: np.array(self.memory_s),
                            self.a: np.array(self.memory_a),
                            self.r: discounted_norm_r,
                      })
        # clear memory
        self.clear_transition()

    def summary(self, step, reward_hist):
        if self.output_graph_path:
            reward_hist = np.array(reward_hist[-min(30,len(reward_hist)):])
            # discounted and normalize reward
            discounted_norm_r = self.memory_r
            discounted_norm_r = self._discount_rewards(discounted_norm_r)
            discounted_norm_r = self._normalize_rewards(discounted_norm_r)
            result = self.sess.run(self.summary_op,
                                   feed_dict={
                                        self.reward_hist: reward_hist,
                                        self.s: np.array(self.memory_s),
                                        self.a: np.array(self.memory_a),
                                        self.r: discounted_norm_r,
                                   })
            self.summary_writer.add_summary(result, step)

    def choose_action(self, s):
        action_prob = self.sess.run(self.network, feed_dict={self.s: s[np.newaxis,:]})
        action = np.random.choice(range(self.n_actions), p=action_prob.ravel())
        return action

    def choose_best_action(self, s):
        action_prob = self.sess.run(self.network, feed_dict={self.s: s[np.newaxis,:]})
        action = np.argmax(action_prob)
        return action

    def save(self, checkpoint_file_path):
        if not os.path.exists(os.path.dirname(checkpoint_file_path)):
            os.makedirs(os.path.dirname(checkpoint_file_path))
        self.saver.save(self.sess, checkpoint_file_path)
        print('Model saved to: {}'.format(checkpoint_file_path))
    
    def load(self, checkpoint_file_path):
        self.saver.restore(self.sess, checkpoint_file_path)
        print('Model restored from: {}'.format(checkpoint_file_path))



class PolicyGradient(BasicPolicyGradient):
    def _net(self, inputs):
        net = inputs
        print(net.name, net.shape)
        net = tf.layers.conv2d(
            inputs=net, 
            filters=16,
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
            filters=32, 
            kernel_size=(4, 4), 
            strides=(2, 2), 
            padding='valid',
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name='conv2'
        )
        print(net.name, net.shape)
        net = tf.contrib.layers.flatten(net, scope='flatten')
        print(net.name, net.shape)
        net = tf.layers.dense(
            inputs=net, 
            units=128,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='fc3'
        )
        print(net.name, net.shape)
        net = tf.layers.dense(
            inputs=net, 
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='fc4'
        )
        print(net.name, net.shape)
        return net


        