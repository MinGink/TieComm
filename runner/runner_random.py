import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from torch.optim import Adam
from .runner import Runner

from modules.utils import merge_dict
import time

Transition = namedtuple('Transition', ('obs', 'actions', 'action_outs', 'rewards',
                                       'episode_masks', 'episode_agent_masks', 'values'))



class RunnerRandom(Runner):
    def __init__(self, args, env, agent):
        super(RunnerRandom, self).__init__(args, env, agent)

        self.args = args
        self.env = env
        self.agent = agent
        self.algo = args.algo
        self.n_agents =args.n_agents

        self.total_steps = 0

        self.gamma = self.args.gamma
        self.lamda = self.args.lamda


        self.params = [p for p in self.agent.parameters()]
        self.optimizer = Adam(params=self.agent.parameters(), lr=args.lr)
        self.no_group = list(range(self.n_agents))







    def run_an_episode(self):
        memory = []
        info = dict()
        log = dict()
        episode_return = 0

        self.reset()
        #state = self.env.get_state()
        obs = self.env.get_obs()

        #prev_hid = self.agent.init_hidden(batch_size=state.shape[0])

        for t in range(self.args.episode_length):

            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)

            if self.agent =="tiecomm_random":
                set = self.agent.random_set()
            else:
                set = [self.no_group]

            after_comm = self.agent.communicate(obs_tensor, set)
            action_outs, values = self.agent.agent(after_comm)

            actions = self.choose_action(action_outs)

            rewards, dones, env_info = self.env.step(actions)

            next_state = self.env.get_state()
            next_obs = self.env.get_obs()

            episode_mask = np.zeros(np.array(rewards).shape)
            episode_agent_mask = np.array(dones) + 0

            if all(dones) or t == self.args.episode_length - 1:
                episode_mask = np.ones(np.array(rewards).shape)

            trans = Transition(np.array(obs), actions, action_outs, np.array(rewards),
                                    episode_mask, episode_agent_mask, values)
            memory.append(trans)


            obs = next_obs


            episode_return += float(sum(rewards))
            self.total_steps += 1

            if all(dones) or t == self.args.episode_length - 1:
                log['episode_return'] = [episode_return]
                log['episode_steps'] = [t + 1]
                log['num_steps'] = t + 1
                break

        return memory ,log