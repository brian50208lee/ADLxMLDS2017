import time

import numpy as np
import tensorflow as tf

from agent_dir.agent import Agent
from agent_dir.DQN import DeepQNetwork

class Agent_DQN(Agent):
    def __init__(self, env, args):
        super(Agent_DQN,self).__init__(env)
        # enviroment infomation
        self.action_map = range(env.action_space.n)

        # model parameters
        self.n_actions = len(self.action_map)
        self.inputs_shape = [84, 84, 4]

        # learning parameters
        self.learn_start = 10000
        self.learn_freq = 4
        self.replace_target_freq = 1000
        self.summary_freq = 100
        self.max_step = 5e6
        self.explore_rate = 1.0
        self.min_explore_rate = 0.05
        self.decrease_explore_rate = (self.explore_rate - self.min_explore_rate) / (self.max_step * 0.2)
        
        # model
        self.model = DeepQNetwork(
                        inputs_shape=self.inputs_shape,
                        n_actions=self.n_actions,
                        gamma=0.99,
                        optimizer=tf.train.RMSPropOptimizer,
                        learning_rate=0.0001,
                        batch_size=32,
                        memory_size=30000,
                        output_graph_path='models/break/tb{}'.format(time.strftime("%y%m%d_%H%M%S", time.localtime()))
                     )

        # load
        if args.test_dqn or args.load_best:
            self.model.load('models/break/train/best')


    def init_game_setting(self):
        pass

    def train(self):
        best_mean_reward = 0.
        reward_hist = []
        step = 0
        episode = 0
        while step < self.max_step:
            try:
                episode += 1
                episode_reward = 0.0
                observation = self.env.reset()
                while True:
                    # transition
                    action = self.model.choose_action(observation)
                    if np.random.uniform() < self.explore_rate:
                        action = np.random.randint(0, self.n_actions)
                    pre_observation = observation
                    observation, reward, done, _ = self.env.step(self.action_map[action])
                    self.model.store_transition(pre_observation, action, reward, observation)
                    
                    # update params
                    step += 1
                    episode_reward += reward
                    self.explore_rate -= self.decrease_explore_rate
                    self.explore_rate = max(self.explore_rate, self.min_explore_rate)

                    # learn and replace
                    if step > self.learn_start:
                        if step % self.learn_freq == 0:
                            self.model.learn()
                        if step % self.replace_target_freq == 0:
                            self.model.replace_target_net()
                        if step % self.summary_freq == 0:
                            self.model.summary(step=step, reward_hist=reward_hist)

                    if done:
                        # show info
                        info = 'episode: {}  reward: {}  step: {}  explore_rate: {}'.format(
                                episode, episode_reward, step, self.explore_rate)
                        print(info)
                        # history mean, save best
                        reward_hist.append(episode_reward)
                        if len(reward_hist) > 100:
                            mean_reward = np.array(reward_hist[max(len(reward_hist)-100,0):]).astype('float32').mean()
                            if best_mean_reward != 0. and mean_reward > best_mean_reward:
                                self.model.save('models/break/train/best')
                                print('save best mean reward:', mean_reward)
                            best_mean_reward = max(best_mean_reward, mean_reward)
                        break
            except KeyboardInterrupt:
                cmd = input('\nsave/load/keep/render/exit ?\n')
                if cmd == 'save' or cmd == 's':
                    path = input('save path: ')
                    self.model.save(path)
                    pass
                if cmd == 'load' or cmd == 'l':
                    path = input('load path: ')
                    self.model.load(path)
                    pass
                elif cmd == 'keep' or cmd == 'k':
                    pass
                elif cmd == 'render' or cmd == 'r':
                    self.env.do_render = not self.env.do_render
                    pass
                elif cmd == 'exit' or cmd == 'e':
                    return
                
        self.model.save('models/break/finish/finish')


    def make_action(self, observation, test=True):
        action = self.model.choose_action(observation)
        return self.action_map[action]

