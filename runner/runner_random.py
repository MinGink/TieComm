import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from torch.optim import Adam
from .runner import Runner
import argparse
from modules.utils import merge_dict
import time

Transition = namedtuple('Transition', ('obs', 'actions', 'action_outs', 'rewards',
                                       'episode_masks', 'episode_agent_masks', 'values'))



class RunnerRandom(Runner):
    def __init__(self, config, env, agent):
        super(RunnerRandom, self).__init__(config, env, agent)


        self.n_agents = self.args.n_agents
        self.no_group = list(range(self.n_agents))
        self.algo = self.args.agent



    def run_an_episode(self):

        memory = []
        log = dict()
        episode_return = 0

        self.reset()
        obs = self.env.get_obs()


        step = 1
        done = False
        while not done and step < self.args.episode_length:

            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)

            if self.agent =="tiecomm_random":
                set = self.agent.random_set()
            else:
                set = [self.no_group]

            after_comm = self.agent.communicate(obs_tensor, set)
            action_outs, values = self.agent.agent(after_comm)

            actions = self.choose_action(action_outs)

            rewards, done, env_info = self.env.step(actions)

            next_obs = self.env.get_obs()

            episode_mask = np.zeros(np.array(rewards).shape)
            # episode_agent_mask = np.array(dones) + 0

            if done or step == self.args.episode_length - 1:
                episode_mask = np.ones(np.array(rewards).shape)

            trans = Transition(np.array(obs), actions, action_outs, np.array(rewards),
                                    episode_mask, episode_mask, values)
            memory.append(trans)


            obs = next_obs
            episode_return += float(sum(rewards))
            step += 1
            self.total_steps += 1


        log['episode_return'] = [episode_return]
        log['episode_steps'] = [step]
        log['num_steps'] = step

        if self.args.env == 'tj':
            log['success_rate'] = self.env.get_success_rate()


        return memory ,log