import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from torch.optim import Adam
from modules.utils import merge_dict
import time
import argparse
from .runner import Runner


class RunnerBaseline(Runner):
    def __init__(self, config, env, agent):
        super(RunnerBaseline, self).__init__(config, env, agent)


    def run_an_episode(self):

        memory = []
        info = dict()
        log = dict()
        episode_return = 0

        self.reset()
        obs = self.env.get_obs()

        #prev_hid = self.agent.init_hidden(batch_size=state.shape[0])

        for t in range(self.args.episode_length):
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.n_agents, dtype=int)
            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)
            action_outs, values = self.agent(obs_tensor,info)

            actions = self.choose_action(action_outs)

            rewards, dones, env_info = self.env.step(actions)

            #next_state = self.env.get_state()
            next_obs = self.env.get_obs()
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = actions[-1] if not self.args.comm_action_one else np.ones(self.args.n_agents,
                                                                                                dtype=int)

            episode_mask = np.zeros(np.array(rewards).shape)
            episode_agent_mask = np.array(dones) + 0

            if all(dones) or t == self.args.episode_length - 1:
                episode_mask = np.ones(np.array(rewards).shape)


            trans = self.transition(np.array(obs), actions, action_outs, np.array(rewards),
                                    episode_mask, episode_agent_mask, values)
            memory.append(trans)


            #state = next_state
            obs = next_obs


            episode_return += float(sum(rewards))
            self.total_steps += 1

            if all(dones) or t == self.args.episode_length - 1:
                log['episode_return'] = [episode_return]
                log['episode_steps'] = [t + 1]
                log['num_steps'] = t + 1
                break

        return memory ,log