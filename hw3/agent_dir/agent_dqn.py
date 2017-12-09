import numpy as np

from agent_dir.agent import Agent
from agent_dir.DQN import DeepQNetwork

class Agent_DQN(Agent):
    def __init__(self, env, args):
        super(Agent_DQN,self).__init__(env)
        # enviroment infomation
        self.action_map = [1, 2, 3] # stop, right, left

        # model parameters
        self.n_actions = len(self.action_map)
        self.inputs_shape = [84, 84, 4]

        # learning parameters
        self.learn_freq = 4
        self.learn_start = 5000
        self.max_step = 10e6
        self.explore_rate = 1.0
        self.min_explore_rate = 0.1
        self.decrease_explore_rate = (self.explore_rate - self.min_explore_rate) / (self.max_step * 0.01)
        
        # model
        self.model = DeepQNetwork(
                        n_actions=self.n_actions, 
                        inputs_shape=self.inputs_shape,
                        learning_rate=0.005, 
                        discount=0.99,
                        memory_size=10000,
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
                            self.model.update_target()

                    # slow motion
                    #print('\n', feature_observation[:,list(range(0,21))])
                    #input()

                    if done:
                        # show info
                        print('episode:', episode, 
                              '  reward:', episode_reward, 
                              '  step:', step,
                              '  explore_rate:', self.explore_rate)
                        # history mean, save best
                        reward_hist.append(episode_reward)
                        if len(reward_hist) > 200:
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
        return self.action_map(action)

