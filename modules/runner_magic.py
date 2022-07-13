import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from torch.optim import Adam

from .utils import merge_dict
from .runner import Runner
import time

class magicRunner(Runner):
    def __init__(self, args, env, agent):
        super(magicRunner, self).__init__(args, env, agent)

        self.args = args

        self.env = env
        # self.agent = agent_REGISTRY[args.algo](args)
        self.agent = agent
        self.total_steps = 0

        self.gamma = self.args.gamma
        self.lamda = self.args.lamda


        self.transition = namedtuple('Transition', ('state', 'obs',
                                                    'actions','action_outs','rewards',
                                                    'episode_masks', 'episode_agent_masks','values'))



        self.params = list(self.agent.parameters())
        self.optimizer = Adam(params=self.params, lr=args.lr)



    def run_an_episode(self):
        memory = []
        info = dict()
        log = dict()
        episode_return = 0

        self.reset()
        state = self.env.get_state()
        obs = self.env.get_obs()

        prev_hid = torch.zeros(1, self.args.n_agents, self.args.hid_size)
        # prev_hid = self.agent.init_hidden(batch_size=state.shape[0])

        for t in range(self.args.episode_length):
            if t == 0:
                prev_hid = self.agent.init_hidden(batch_size=1)
            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)
            obs_tensor = obs_tensor.unsqueeze(0)
            obs_tensor = [obs_tensor, prev_hid]
            action_outs, values, prev_hid = self.agent(obs_tensor, info)

            if (t + 1) % self.args.detach_gap == 0:
                prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())

            actions = self.choose_action(action_outs)

            rewards, dones, env_info = self.env.step(actions)

            next_state = self.env.get_state()
            next_obs = self.env.get_obs()


            episode_mask = np.zeros(np.array(rewards).shape)
            episode_agent_mask = np.array(dones) + 0

            if all(dones) or t == self.args.episode_length - 1:
                episode_mask = np.ones(np.array(rewards).shape)

            trans = self.transition(state, np.array(obs), actions, action_outs, np.array(rewards),
                                    episode_mask, episode_agent_mask, values)
            memory.append(trans)

            state = next_state
            obs = next_obs

            episode_return += float(sum(rewards))
            self.total_steps += 1

            if all(dones) or t == self.args.episode_length - 1:
                log['episode_return'] = [episode_return]
                log['episode_steps'] = [t + 1]
                log['num_steps'] = t + 1
                break

        return memory, log


