import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from torch.optim import Adam
from modules import Runner

from .utils import merge_dict
import time


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


        self.transition = namedtuple('Transition', ('state', 'obs',
                                                    'actions','action_outs','rewards',
                                                    'episode_masks', 'episode_agent_masks','values'))



        self.params = [p for p in self.agent.parameters()]
        self.optimizer = Adam(params=self.agent.parameters(), lr=args.lr)
        self.no_group = list(range(self.n_agents))







    def run_an_episode(self):
        memory = []
        info = dict()
        log = dict()
        episode_return = 0

        self.reset()
        state = self.env.get_state()
        obs = self.env.get_obs()

        #prev_hid = self.agent.init_hidden(batch_size=state.shape[0])

        for t in range(self.args.episode_length):

            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)
            if self.algo == 'tiecomm_random':
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

        return memory ,log



    def compute_grad(self, batch):

        log = dict()

        rewards = torch.Tensor(batch.rewards)
        actions = torch.stack(batch.actions,dim=0)
        episode_masks = torch.Tensor(batch.episode_masks)
        episode_agent_masks = torch.Tensor(batch.episode_agent_masks)
        values = torch.stack(batch.values, dim=0).squeeze(-1) # (batch, n,1
        action_outs = torch.stack(batch.action_outs, dim=0)

        batch_size = len(batch.actions)
        n = self.args.n_agents

        coop_returns = torch.Tensor(batch_size, n)
        ncoop_returns = torch.Tensor(batch_size, n)
        returns = torch.Tensor(batch_size, n)
        advantages = torch.Tensor(batch_size, n)

        prev_coop_return = 0
        prev_ncoop_return = 0

        for i in reversed(range(batch_size)):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * (1-episode_masks[i])
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * (1- episode_masks[i]) * (1-episode_agent_masks[i])

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i])


        for i in reversed(range(batch_size)):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()


        actions_taken = torch.gather(action_outs, dim=-1, index = actions.unsqueeze(-1)).squeeze(-1)
        log_actions_taken = torch.log(actions_taken + 1e-10)

        action_loss = (-advantages.detach().view(-1) * log_actions_taken.view(-1)).sum()
        value_loss = ((values - returns).pow(2).view(-1)).sum()


        total_loss = action_loss + self.args.value_coeff * value_loss


        #self.agent_optimiser.zero_grad()
        total_loss.backward()
        #nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        #self.agent_optimiser.step()


        log['action_loss'] = action_loss.item()
        log['value_loss'] = value_loss.item()
        log['total_loss'] = total_loss.item()

        return log



