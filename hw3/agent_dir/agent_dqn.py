import numpy as np
from agent_dir.agent import Agent
from agent_dir.DQN import DeepQNetwork

def prepro(I):
    """ prepro 84x84x4 uint8 frame into 3034 (41x74) 1D float vector """
    I = I[-46:-5,5:-5,:] # crop
    I[I != 0] = 1. # everything else set to 1
    return I.astype(np.float)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        super(Agent_DQN,self).__init__(env)

        # enviroment infomation
        print('Action Num:', env.action_space.n)
        print('Observe Shape:', env.observation_space.shape)
        self.action_map = [1, 2, 3] # stop, left, right

        # model parameters
        self.n_actions = len(self.action_map)
        self.inputs_shape = [41, 74, 4]

        # learning parameters
        self.learn_freq = 4
        self.learn_start = 5000
        self.replace_target_freq = 10000
        self.max_episode = 1000000
        
        # model
        self.model = DeepQNetwork(
                        n_actions=self.n_actions, 
                        inputs_shape=self.inputs_shape,
                        learning_rate=0.0001, 
                        discount=0.99,
                        epsilon_max=0.9,
                        epsilon_increment=0.00001,
                        memory_size=10000,
                     )

        # load
        if args.test_pg or args.load_best:
            self.model.load('models/break/train/best')


    def init_game_setting(self):
        self.pre_observation = None


    def train(self):
        best_mean = 0.
        reward_hist = []
        step = 0
        for episode in range(self.max_episode):
            try:
                observation = self.env.reset()
                observation = prepro(observation)
                episode_reward = 0.0
                while True:
                    # transition
                    action = self.model.choose_action(observation)
                    pre_observation = observation
                    observation, reward, done, _ = self.env.step(self.action_map[action])
                    observation = prepro(observation)
                    self.model.store_transition(pre_observation, action, reward, observation)
                    
                    # update step
                    episode_reward += reward
                    step += 1

                    # learn and replace
                    if step > self.learn_start:
                        if step % self.learn_freq == 0:
                            self.model.learn()
                        if step % self.replace_target_freq == 0:
                            self.model.replace_target()

                    # slow motion
                    #print('\n', feature_observation[:,list(range(0,21))])
                    #input()

                    if done:
                        # show info
                        print('episode:', episode, 
                              '  reward:', episode_reward, 
                              '  step:', step,
                              '  epsilon:', self.model.epsilon)
                        # history mean, save best
                        reward_hist.append(episode_reward)
                        if len(reward_hist) > 100:
                            mean = np.array(reward_hist[max(len(reward_hist)-30,0):]).astype('float32').mean()
                            if best_mean != 0. and mean > best_mean:
                                self.model.save('models/break/train/best')
                                print('save best mean reward:', mean)
                            best_mean = max(best_mean, mean)
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
                
        self.model.save('models/break_finish/finish')


    def make_action(self, observation, test=True):
        observation = prepro(observation)
        feature_observation = observation if self.pre_observation is None else observation - self.pre_observation
        self.pre_observation = observation
        action = self.model.choose_action(feature_observation)
        return self.action_map(action)

