import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from torch.optim import Adam
from modules.utils import merge_dict, multinomials_log_density
import time

import argparse

Transition = namedtuple('Transition', ('obs', 'action_outs', 'actions', 'rewards',
                                        'episode_masks', 'episode_mini_masks', 'values'))
class Runner(object):
    def __init__(self, config, env, agent):

        self.args = argparse.Namespace(**config)
        self.env = env
        self.agent = agent
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents

        self.gamma = self.args.gamma
        # self.lamda = self.args.lamda


        self.params = [p for p in self.agent.parameters()]
        self.optimizer = Adam(params=self.agent.parameters(), lr=self.args.lr)



    def train_batch(self, batch_size):

        batch_data, batch_log = self.collect_batch_data(batch_size)
        self.optimizer.zero_grad()
        train_log = self.compute_grad(batch_data)
        merge_dict(batch_log, train_log)

        for p in self.params:
            if p._grad is not None:
                p._grad.data /= batch_log['num_steps']
        self.optimizer.step()
        return train_log




    def collect_batch_data(self, batch_size):

        batch_data = []
        batch_log = dict()
        num_episodes = 0

        while len(batch_data) < batch_size:
            episode_data, episode_log = self.run_an_episode()
            batch_data += episode_data
            merge_dict(episode_log, batch_log)
            num_episodes += 1

        batch_log['num_episodes'] = num_episodes
        batch_log['num_steps'] = len(batch_data)

        batch_data = Transition(*zip(*batch_data))

        return batch_data, batch_log


    def run_an_episode(self):

        memory = []
        info = dict()
        log = dict()
        episode_return = 0

        self.reset()
        obs = self.env.get_obs()


        step = 1
        done = False
        while not done and step <= self.args.episode_length:

            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)

            action_outs, values = self.agent(obs_tensor,info)
            actuals = self.choose_action(action_outs)
            rewards, done, env_info = self.env.step(actuals)

            next_obs = self.env.get_obs()

            episode_mask = np.ones(rewards.shape)
            episode_mini_mask = np.ones(rewards.shape)

            if done:
                episode_mask = np.zeros(rewards.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)



            episode_mask = np.zeros(np.array(rewards).shape)
            if done or step == self.args.episode_length-1:
                episode_mask = np.ones(np.array(rewards).shape)

            trans = Transition(np.array(obs), action_outs, actuals, np.array(rewards),
                                    episode_mask, episode_mini_mask, values)


            memory.append(trans)

            obs = next_obs
            episode_return += rewards
            step += 1

        log['episode_return'] = episode_return
        log['episode_steps'] = [step-1]

        if self.args.env == 'tj':
            merge_dict(self.env.get_stat(),log)

        return memory, log




    def _compute_returns(self, rewards, masks, next_value):
        returns = [torch.zeros_like(next_value)]
        for rew, done in zip(reversed(rewards), reversed(masks)):
            ret = returns[0] * self.gamma * (1 - done) + rew
            returns.insert(0, ret)
        return returns




    def compute_grad(self, batch):
        return self.compute_agent_grad(batch)




    def compute_agent_grad(self, batch):

        log = dict()

        n = self.n_agents
        batch_size = self.args.batch_size


        rewards = torch.Tensor(batch.rewards)
        actions = torch.Tensor(batch.actions)
        actions = actions.transpose(1, 2).view(-1, n, 1)

        episode_masks = torch.Tensor(batch.episode_masks)
        episode_mini_masks = torch.Tensor(batch.episode_mini_masks)


        values = torch.cat(batch.values, dim=0)  # (batch, n,1
        action_outs = torch.stack(batch.action_outs, dim=0)

        # returns = self._compute_returns(rewards, episode_masks, values[-1])
        # returns = torch.stack(returns)[:-1]

        ncoop_returns = torch.Tensor(batch_size, n)
        returns = torch.Tensor(batch_size, n)
        advantages = torch.Tensor(batch_size, n)
        values = values.view(batch_size, n)

        prev_ncoop_return = 0

        for i in reversed(range(rewards.size(0))):
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] =  ncoop_returns[i]


        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        # element of log_p_a: [(batch_size*n) * num_actions[i]]
        log_p_a = [action_outs.view(-1, self.n_actions)]
        # actions: [(batch_size*n) * dim_actions]
        actions = actions.contiguous().view(-1, 1)

        log_prob = multinomials_log_density(actions, log_p_a)
        action_loss = -advantages.view(-1) * log_prob.squeeze()

        actor_loss = action_loss.sum()



        # actions_taken = torch.gather(action_outs, dim=-1, index = actions.unsqueeze(-1)).squeeze(-1)
        # log_actions_taken = torch.log(actions_taken + 1e-10)
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        # action_loss = (-advantages.detach().view(-1) * log_actions_taken.view(-1)).sum()
        critic_loss = value_loss.sum()




        total_loss = actor_loss + self.args.value_coeff * critic_loss
        total_loss.backward()


        log['action_loss'] = actor_loss.item()
        log['value_loss'] = critic_loss.item()
        log['total_loss'] = total_loss.item()

        return log



    def reset(self):
        self.env.reset()


    def get_env_info(self):
        return self.env.get_env_info()


    def choose_action(self, log_p_a):
        log_p_a = [log_p_a.unsqueeze(0)]
        p_a = [[z.exp() for z in x] for x in log_p_a]
        ret = torch.stack([torch.stack([torch.multinomial(x, 1).detach() for x in p]) for p in p_a])
        action = [x.squeeze().data.numpy() for x in ret]
        return action


    def save_model(self):
        return self.agent.save_model()