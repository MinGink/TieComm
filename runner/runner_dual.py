import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from torch.optim import Adam

from modules.utils import merge_dict
from .runner import Runner
import time


class RunnerDual(Runner):
    def __init__(self, args, env, agent):
        super(RunnerDual, self).__init__(args, env, agent)

        self.args = args
        self.env = env
        self.agent = agent
        self.algo = args.algo
        self.n_agents = args.n_agents

        self.n_nodes = int(self.n_agents * (self.n_agents - 1)/2)

        self.total_steps = 0

        self.gamma = self.args.gamma
        self.lamda = self.args.lamda


        self.transition = namedtuple('Transition', ('state', 'obs',
                                                    'actions','action_outs','rewards',
                                                    'episode_masks', 'episode_agent_masks','values',
                                                    'god_action_out', 'god_value', 'god_action','god_reward'
                                                    ))



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






    def run_an_episode(self):
        memory = []
        log = dict()
        episode_return = 0

        self.reset()
        state = self.env.get_state()
        obs = self.env.get_obs()

        #prev_hid = self.agent.init_hidden(batch_size=state.shape[0])

        for t in range(self.args.episode_length):

            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)
            set,god_action_out, god_value, god_action = self.agent.god(obs_tensor)
            after_comm = self.agent.communicate(obs_tensor, set)
            action_outs, values = self.agent.agent(after_comm)

            actions = self.choose_action(action_outs)

            rewards, dones, env_info = self.env.step(actions)

            god_reward = np.sum(rewards)

            next_state = self.env.get_state()
            next_obs = self.env.get_obs()

            episode_mask = np.zeros(np.array(rewards).shape)
            episode_agent_mask = np.array(dones) + 0

            if all(dones) or t == self.args.episode_length - 1:
                episode_mask = np.ones(np.array(rewards).shape)


            trans = self.transition(state, np.array(obs), actions, action_outs, np.array(rewards),
                                    episode_mask, episode_agent_mask, values, god_action_out, god_value, god_action, god_reward)
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
        log={}
        agent_log = self.compute_agent_grad(batch)
        god_log = self.compute_god_grad(batch)
        merge_dict(agent_log, log)
        merge_dict(god_log, log)

        return log




    def compute_god_grad(self, batch):

        log = dict()
        batch_size = len(batch.actions)
        n = self.n_nodes

        episode_masks = torch.Tensor(batch.episode_masks)[:,:1].repeat(1,n)
        #episode_agent_masks = torch.Tensor(batch.episode_agent_masks)

        god_rewards = torch.Tensor(batch.god_reward).unsqueeze(-1).repeat(1,n)
        god_actions = torch.stack(batch.god_action, dim=0)
        god_values = torch.stack(batch.god_value, dim=0).squeeze(-1)
        god_action_outs = torch.stack(batch.god_action_out, dim=0)



        returns = torch.Tensor(batch_size, n)
        advantages = torch.Tensor(batch_size, n)
        prev_return = 0

        for i in reversed(range(batch_size)):
            returns[i] = god_rewards[i] + self.args.gamma * prev_return * (1 - episode_masks[i])
            prev_return = returns[i].clone()

        for i in reversed(range(batch_size)):
            advantages[i] = returns[i] - god_values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()



        actions_taken = torch.gather(god_action_outs, dim=-1, index = god_actions.unsqueeze(-1)).squeeze(-1)
        log_actions_taken = torch.log(actions_taken + 1e-10)

        action_loss = (-advantages.detach().view(-1) * log_actions_taken.view(-1)).sum()


        value_loss = ((god_values - returns).pow(2).view(-1)).sum()

        total_loss = action_loss + self.args.value_coeff * value_loss

        total_loss.backward()

        log['god_action_loss'] = action_loss.item()
        log['god_value_loss'] = value_loss.item()
        log['god_total_loss'] = total_loss.item()


        return log






    def compute_agent_grad(self, batch):

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
