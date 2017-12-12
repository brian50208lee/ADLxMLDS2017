import time

import scipy
import numpy as np
import tensorflow as tf

from agent_dir.agent import Agent
from agent_dir.PG import PolicyGradient


def prepro(o, image_size=[80,80]):
    y = 0.2126 * o[:,:,0] + 0.7152 * o[:,:,1] + 0.0722 * o[:,:,2]
    resized = scipy.misc.imresize(y.astype(np.uint8), image_size)
    return np.expand_dims(resized.astype(np.float32), axis=2)

class Agent_PG(Agent):
    def __init__(self, env, args):
        super(Agent_PG,self).__init__(env)

        # enviroment infomation
        self.action_map = range(env.action_space.n)

        # model parameters
        self.n_actions = len(self.action_map)
        self.inputs_shape = [80, 80, 1]

        # learning parameters
        self.max_episode = 5000

        # model
        self.model = PolicyGradient(
                        inputs_shape=self.inputs_shape, 
                        n_actions=self.n_actions,
                        gamma=0.99,
                        optimizer=tf.train.AdamOptimizer,
                        learning_rate=0.001,
                        output_graph_path='models/pong/tb{}'.format(time.strftime("%y%m%d_%H%M%S", time.localtime()))
                     )

        # load
        if args.test_pg or args.load_best:
            print('loading trained model')
            self.model.load('models/pong/112/best')


    def init_game_setting(self):
        self.pre_observation = None

    def train(self):
        best_mean_reward = -21
        reward_hist = [-21]
        episode = 0
        while episode < self.max_episode:
            try:
                episode += 1
                episode_reward = 0.0
                pre_observation = None
                observation = self.env.reset()
                observation = prepro(observation)
                while True:
                    # feature
                    if pre_observation is None:
                        pre_observation = observation
                    feature_observation = observation - pre_observation
                    feature_observation = np.sign(feature_observation).astype(np.float32)

                    # do action
                    action = self.model.choose_action(feature_observation)
                    next_observation, reward, done, _ = self.env.step(self.action_map[action])
                    self.model.store_transition(feature_observation, action, reward)

                    # next step
                    pre_observation = observation
                    observation = prepro(next_observation)
                    episode_reward += reward

                    #print(feature_observation[:,-14:-7,0])
                    #input()

                    # done
                    if done:
                        # show info
                        info = 'episode: {}  reward: {}'.format(
                                episode, episode_reward)
                        print(info)
                        reward_hist.append(episode_reward)
                        # save best
                        if len(reward_hist) > 30:
                            mean_reward = np.array(reward_hist[max(len(reward_hist)-30,0):]).astype('float32').mean()
                            if best_mean_reward != -21 and mean_reward > best_mean_reward:
                                self.model.save('models/pong/train/best')
                                print('save best mean reward:', mean_reward)
                            best_mean_reward = max(best_mean_reward, mean_reward)
                        # learn
                        self.model.summary(step=episode, reward_hist=reward_hist)
                        if episode_reward > min(reward_hist[-5:]):
                            self.model.learn()
                        else:
                            self.model.clear_transition()
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
                
        self.model.save('models/pong/finish/finish')
                
    def make_action(self, observation, test=True):
        observation = prepro(observation)
        # feature
        if self.pre_observation is None:
                self.pre_observation = observation
        feature_observation = observation - self.pre_observation
        action = self.model.choose_best_action(feature_observation)
        return self.action_map[action]

