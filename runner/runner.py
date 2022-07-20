import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from torch.optim import Adam
from modules.utils import merge_dict
import time
import argparse


class Runner(object):
    def __init__(self, config, env, agent):

        self.args = argparse.Namespace(**config)
        self.env = env
        self.agent = agent

        self.total_steps = 0

        self.gamma = self.args.gamma
        # self.lamda = self.args.lamda

        self.transition = namedtuple('Transition', ('obs', 'actions','action_outs','rewards',
                                                    'episode_masks', 'episode_agent_masks','values'))



        self.params = [p for p in self.agent.parameters()]
        self.optimizer = Adam(params=self.agent.parameters(), lr=self.args.lr)




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




    def collect_epoch_data(self, epoch_size):
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
        obs = self.env.get_obs()

        #prev_hid = self.agent.init_hidden(batch_size=state.shape[0])

        step = 1
        done = False
        while not done and step < self.args.episode_length:
            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)
            action_outs, values = self.agent(obs_tensor,info)

            actions = self.choose_action(action_outs)

            next_obs, rewards, done, env_info = self.env.step(actions)


            episode_mask = np.zeros(np.array(rewards).shape)

            if done or step == self.args.episode_length-1:
                episode_mask = np.ones(np.array(rewards).shape)

            trans = self.transition(np.array(obs), actions, action_outs, np.array(rewards),
                                    episode_mask, episode_mask, values)


            memory.append(trans)

            obs = next_obs
            episode_return += float(sum(rewards))
            step += 1
            self.total_steps += 1

        log['episode_return'] = [episode_return]
        log['episode_steps'] = [step]
        log['num_steps'] = step


        return memory ,log

    def _compute_returns(self, rewards, masks, next_value):
        returns = [next_value]
        for rew, done in zip(reversed(rewards), reversed(masks)):
            ret = returns[0] * self.gamma + rew * (1 - done.unsqueeze(1))
            returns.insert(0, ret)
        return returns




    def compute_grad(self, batch):

        log = dict()

        rewards = torch.Tensor(batch.rewards)
        actions = torch.stack(batch.actions,dim=0)
        episode_masks = torch.Tensor(batch.episode_masks)
        values = torch.stack(batch.values, dim=0).squeeze(-1) # (batch, n,1
        action_outs = torch.stack(batch.action_outs, dim=0)

        # episode_agent_masks = torch.Tensor(batch.episode_agent_masks)


        batch_size = len(batch.actions)
        # n = self.args.n_agents

        returns = self._compute_returns(rewards, episode_masks, values)

        advantages = returns - values


        # coop_returns = torch.Tensor(batch_size, n)
        # ncoop_returns = torch.Tensor(batch_size, n)
        # returns = torch.Tensor(batch_size, n)
        # advantages = torch.Tensor(batch_size, n)
        #
        # prev_coop_return = 0
        # prev_ncoop_return = 0
        #
        # for i in reversed(range(batch_size)):
        #     coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * (1-episode_masks[i])
        #     ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * (1- episode_masks[i]) * (1-episode_agent_masks[i])
        #
        #     prev_coop_return = coop_returns[i].clone()
        #     prev_ncoop_return = ncoop_returns[i].clone()
        #
        #     returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
        #                 + ((1 - self.args.mean_ratio) * ncoop_returns[i])
        # for i in reversed(range(batch_size)):
        #     advantages[i] = returns[i] - values.data[i]

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
