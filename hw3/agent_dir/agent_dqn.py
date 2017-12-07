import numpy as np
from agent_dir.agent import Agent
from agent_dir.DQN import DeepQNetwork

def prepro(I):
    """ prepro 84x84x4 uint8 frame into 777 (21x37) 1D float vector """
    I = I[:,:,0] # chanel 0
    I = I[-46:-5,5:-5] # crop
    I = I[::2,::2] # downsample by 2
    I[I != 0] = 1. # everything else set to 1
    return I.astype(np.float)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        super(Agent_DQN,self).__init__(env)

        # enviroment infomation
        print('Action Num:', env.action_space.n)
        print('Observe Shape:', env.observation_space.shape)

        # hyperparameters
        self.n_actions = 2
        self.n_features = [21, 37]
        self.n_hidden = 200 # number of hidden layer neurons
        
        # model
        self.model = DeepQNetwork(
                        n_actions=self.n_actions, 
                        n_features=self.n_features,
                        learning_rate=0.0005, 
                        reward_decay=0.99,
                        memory_size=5000,
                        epsilon_increment=0.0001
                     )

        # load
        if args.test_pg or args.load_best:
            self.model.load('models/pong/112/best')


    def init_game_setting(self):
        pass


    def train(self):
        best_mean = 0.
        reward_hist = []
        for episode in range(1,1000000):
            try:
                pre_observation = None
                observation = self.env.reset()
                observation = prepro(observation)
                episode_reward = 0.0
                while True:
                    # diff_observation
                    feature_observation = observation if pre_observation is None else observation - pre_observation
                    pre_observation = observation

                    # transition
                    action = self.model.choose_action(feature_observation)
                    observation, reward, done, info = self.env.step(action+2)
                    observation = prepro(observation)
                    episode_reward += reward

                    # store
                    self.model.store_transition(feature_observation, action, reward, observation-pre_observation)
                    
                    # slow motion
                    #print('\n', feature_observation[:,list(range(0,21))])
                    #input()

                    if done:
                        # show info
                        print('episode:', episode, '  reward:', episode_reward)
                        
                        # history mean, save best
                        reward_hist.append(episode_reward)
                        if len(reward_hist) > 100:
                            mean = np.array(reward_hist[max(len(reward_hist)-30,0):]).astype('float32').mean()
                            if best_mean != 0. and mean > best_mean:
                                self.model.save('models/break_train/best')
                                print('save best mean reward:', mean)
                            best_mean = max(best_mean, mean)

                        # learn
                        #self.model.learn()
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
        return self.env.get_random_action()

