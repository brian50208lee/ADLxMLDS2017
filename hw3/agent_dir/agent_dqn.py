import numpy as np
from agent_dir.agent import Agent
from agent_dir.DQN import DeepQNetwork

def prepro(I):
    """ prepro 84x84x4 uint8 frame into 3034 (41x74) 1D float vector """
    I = I[:,:,0] # chanel 0
    I = I[-46:-5,5:-5] # crop
    I[I != 0] = 1. # everything else set to 1
    return I.astype(np.float)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        super(Agent_DQN,self).__init__(env)

        # enviroment infomation
        print('Action Num:', env.action_space.n)
        print('Observe Shape:', env.observation_space.shape)

        # hyperparameters
        self.n_actions = 3
        self.n_features = [41, 74]
        
        # model
        self.model = DeepQNetwork(
                        n_actions=self.n_actions, 
                        n_features=self.n_features,
                        learning_rate=0.0005, 
                        reward_decay=0.99,
                        memory_size=10000,
                        epsilon_max=0.9,
                        epsilon_increment=0.000001,
                        replace_target_iter=1000
                     )

        # load
        if args.test_pg or args.load_best:
            self.model.load('models/break/train/best')


    def init_game_setting(self):
        self.pre_observation = None


    def train(self):
        best_mean = 0.
        reward_hist = []
        for episode in range(1,1000000):
            try:
                observation = self.env.reset()
                observation = prepro(observation)
                episode_reward = 0.0
                step = 0
                while True:
                    # diff_observation
                    pre_observation = observation

                    # transition
                    action = self.model.choose_action(pre_observation)
                    observation, reward, done, info = self.env.step(action+1)
                    observation = prepro(observation)
                    episode_reward += reward

                    # store
                    self.model.store_transition(pre_observation, action, reward, observation)
                    
                    # learn
                    step += 1
                    if step % 4 == 0:
                        self.model.learn()

                    # slow motion
                    #print('\n', feature_observation[:,list(range(0,21))])
                    #input()

                    if done:
                        # show info
                        print('episode:', episode, '  reward:', episode_reward, '  memory_count:', self.model.memory_counter)
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
        action = self.model.choose_action(feature_observation) + 1
        return action

