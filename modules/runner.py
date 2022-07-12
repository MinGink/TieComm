import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from torch.optim import Adam

from .utils import merge_dict
import time


class Runner(object):
    def __init__(self, args, env, agent):

        self.args = args
        self.env = env
        self.agent = agent

        self.total_steps = 0

        self.gamma = self.args.gamma
        self.lamda = self.args.lamda


        self.transition = namedtuple('Transition', ('state', 'obs',
                                                    'actions','action_outs','rewards',
                                                    'episode_masks', 'episode_agent_masks','values'))



        self.params = [p for p in self.agent.parameters()]
        self.optimizer = Adam(params=self.agent.parameters(), lr=args.lr)




    def train_batch(self, epoch_size):

        batch_data, batch_log = self.collect_epoch_data(epoch_size)
        self.optimizer.zero_grad()
        train_log = self.compute_grad(batch_data)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= batch_log['num_steps']
        merge_dict(batch_log, train_log)
        self.optimizer.step()
        return train_log




    def collect_epoch_data(self, epoch_size=1):
        epoch_data = []
        epoch_log = dict()
        num_episodes = 0
        for i in range(epoch_size):
            episode_data, episode_log = self.run_an_episode()
            epoch_data += episode_data
            merge_dict(episode_log, epoch_log)
            num_episodes += 1

        epoch_data = self.transition(*zip(*epoch_data))
        epoch_log['num_episodes'] = num_episodes

        return epoch_data, epoch_log


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
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.n_agents, dtype=int)
            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)
            action_outs, values = self.agent(obs_tensor,info)

            actions = self.choose_action(action_outs)

            rewards, dones, env_info = self.env.step(actions)

            next_state = self.env.get_state()
            next_obs = self.env.get_obs()
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = actions[-1] if not self.args.comm_action_one else np.ones(self.args.n_agents,
                                                                                                dtype=int)

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



    def reset(self):
        self.env.reset()


    def get_env_info(self):
        return self.env.get_env_info()


    def choose_action(self, action_out):
        dist = torch.distributions.Categorical(action_out)
        action = dist.sample()
        return action


    def save_model(self):
        return self.agent.save_model()

class magicRunner(object):
    def __init__(self, args, env, agent):

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

    def train_batch(self, epoch_size):

        batch_data, batch_log = self.collect_epoch_data(epoch_size)
        self.optimizer.zero_grad()
        train_log = self.compute_grad(batch_data)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= batch_log['num_steps']
        merge_dict(batch_log, train_log)
        self.optimizer.step()
        return train_log

    def collect_epoch_data(self, epoch_size=1):
        epoch_data = []
        epoch_log = dict()
        num_episodes = 0
        for i in range(epoch_size):
            episode_data, episode_log = self.run_an_episode()
            epoch_data += episode_data
            merge_dict(episode_log, epoch_log)
            num_episodes += 1

        epoch_data = self.transition(*zip(*epoch_data))
        epoch_log['num_episodes'] = num_episodes

        return epoch_data, epoch_log

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

    def compute_grad(self, batch):

        log = dict()

        rewards = torch.Tensor(batch.rewards)
        actions = torch.stack(batch.actions, dim=0)
        episode_masks = torch.Tensor(batch.episode_masks)
        episode_agent_masks = torch.Tensor(batch.episode_agent_masks)
        values = torch.stack(batch.values, dim=0).squeeze(-1)  # (batch, n,1
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
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * (1 - episode_masks[i])
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * (1 - episode_masks[i]) * (
                        1 - episode_agent_masks[i])

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                         + ((1 - self.args.mean_ratio) * ncoop_returns[i])

        for i in reversed(range(batch_size)):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        actions_taken = torch.gather(action_outs, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        log_actions_taken = torch.log(actions_taken + 1e-10)

        action_loss = (-advantages.detach().view(-1) * log_actions_taken.view(-1)).sum()
        value_loss = ((values - returns).pow(2).view(-1)).sum()

        total_loss = action_loss + self.args.value_coeff * value_loss

        # self.agent_optimiser.zero_grad()
        total_loss.backward()
        # nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        # self.agent_optimiser.step()

        log['action_loss'] = action_loss.item()
        log['value_loss'] = value_loss.item()
        log['total_loss'] = total_loss.item()

        return log

    def reset(self):
        self.env.reset()

    def get_env_info(self):
        return self.env.get_env_info()

    def choose_action(self, action_out):
        dist = torch.distributions.Categorical(action_out)
        action = dist.sample()
        return action

    def save_model(self):
        return self.agent.save_model()