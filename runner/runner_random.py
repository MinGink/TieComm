import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from .runner import Runner
from modules.utils import merge_dict, multinomials_log_density


Transition = namedtuple('Transition', ('obs', 'actions', 'action_outs', 'rewards',
                                       'episode_masks', 'episode_agent_masks', 'values'))



class RunnerRandom(Runner):
    def __init__(self, config, env, agent):
        super(RunnerRandom, self).__init__(config, env, agent)


        self.n_agents = self.args.n_agents
        self.no_group = list(range(self.n_agents))
        self.algo = self.args.agent
        self.random_prob = self.args.random_prob



    def run_an_episode(self):

        memory = []
        log = dict()
        episode_return = 0

        self.reset()
        obs = self.env.get_obs()


        step = 1
        done = False
        while not done and step <= self.args.episode_length:

            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)
            set = self.agent.random_set(self.random_prob)
            after_comm = self.agent.communicate(obs_tensor, set)
            action_outs, values = self.agent.agent(after_comm)
            actions = self.choose_action(action_outs)
            rewards, done, env_info = self.env.step(actions)
            next_obs = self.env.get_obs()

            episode_mask = np.ones(rewards.shape)
            episode_agent_mask = np.ones(rewards.shape)
            if done:
                episode_mask = np.zeros(rewards.shape)
            else:
                if 'is_completed' in env_info:
                    episode_agent_mask = 1 - env_info['is_completed'].reshape(-1)

            trans = Transition(np.array(obs), actions, action_outs, np.array(rewards),
                                    episode_mask, episode_agent_mask, values)
            memory.append(trans)


            obs = next_obs
            episode_return += rewards
            step += 1


        log['episode_return'] = [episode_return]
        log['episode_steps'] = [step-1]

        if self.args.env == 'tj':
            merge_dict(self.env.get_stat(),log)


        return memory ,log