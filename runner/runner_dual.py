import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from torch.optim import Adam
from modules.utils import merge_dict, multinomials_log_densities
from .runner import Runner
import time

Transition = namedtuple('Transition', ('obs', 'action_outs', 'actions', 'rewards', 'values',
                                       'episode_masks', 'episode_agent_masks'))

God_Transition = namedtuple('God_Transition', ('god_action_out', 'god_value', 'god_action', 'god_reward','episode_masks',
                                       ))



class RunnerDual(Runner):
    def __init__(self,  config, env, agent):
        super(RunnerDual, self).__init__( config, env, agent)
        self.n_agents = self.args.n_agents
        self.n_nodes = int(self.n_agents * (self.n_agents - 1)/2)
        self.interval = self.args.interval




    def run_an_episode(self):

        memory = []
        god_memory = []
        log = dict()
        episode_return = np.zeros(self.n_agents)

        self.reset()
        obs = self.env.get_obs()

        step = 1
        done = False


        obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)
        set, god_action_out, god_value, god_action = self.agent.god(obs_tensor)
        god_reward = np.zeros(1)
        num_group = 0
        while not done and step <= self.args.episode_length:

            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)

            if step % self.interval == 0:
                set, god_action_out, god_value, god_action = self.agent.god(obs_tensor)

            after_comm = self.agent.communicate(obs_tensor, set)
            action_outs, values = self.agent.agent(after_comm)

            actions = self.choose_action(action_outs)
            rewards, dones, env_info = self.env.step(actions)
            god_reward += np.sum(rewards)

            next_obs = self.env.get_obs()

            episode_mask = np.ones(rewards.shape)

            episode_agent_mask = np.ones(rewards.shape)
            if done:
                episode_mask = np.zeros(rewards.shape)
            else:
                if 'is_completed' in env_info:
                    episode_agent_mask = 1 - env_info['is_completed'].reshape(-1)

            trans = Transition(np.array(obs),  action_outs, actions, np.array(rewards), values,
                                    episode_mask, episode_agent_mask)

            if step % self.interval == 0:
                god_trans = God_Transition(god_action_out, god_value, god_action, god_reward,np.ones(god_value.shape),)
                god_memory.append(god_trans)
                god_reward = np.zeros(1)


            memory.append(trans)


            obs = next_obs
            episode_return += rewards.astype(episode_return.dtype)
            step += 1
            num_group += len(set)


        god_trans = God_Transition(god_action_out, god_value, god_action, god_reward, np.zeros(god_value.shape))
        god_memory.append(god_trans)


        log['episode_return'] = episode_return
        log['episode_steps'] = [step-1]
        log['num_groups'] = num_group / (step-1)

        if self.args.env == 'tj':
            merge_dict(self.env.get_stat(), log)

        return (memory,god_memory), log



    def compute_grad(self, batch):
        log=dict()
        agent_log = self.compute_agent_grad(batch[0])
        god_log = self.compute_god_grad(batch[1])
        merge_dict(agent_log, log)
        merge_dict(god_log, log)
        return log


    def compute_god_grad(self, batch):

        log = dict()

        dim_actions = self.n_nodes
        batch_size = len(batch.god_value)


        episode_masks = torch.Tensor(np.array(batch.episode_masks)).squeeze(-1)
        values = torch.cat(batch.god_value, dim=0)
        rewards = torch.Tensor(np.array(batch.god_reward)).unsqueeze(-1)
        actions = torch.stack(batch.god_action, dim=0)
        action_outs = torch.stack(batch.god_action_out, dim=0)



        returns = torch.Tensor(batch_size, 1)
        advantages = torch.Tensor(batch_size, 1)
        values = values.view(batch_size, 1)
        prev_returns = 0

        for i in reversed(range(batch_size)):
            returns[i] = rewards[i] + self.args.gamma * prev_returns * episode_masks[i]
            prev_returns = returns[i].clone()

        for i in reversed(range(batch_size)):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()



        log_p_a = action_outs
        # actions: [(batch_size*n) * dim_actions]
        actions = actions.contiguous().view(-1, dim_actions)
        log_prob = multinomials_log_densities(actions, log_p_a)
        # the log prob of each action head is multiplied by the advantage
        action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
        actor_loss = action_loss.sum()


        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        critic_loss = value_loss.sum()

        total_loss = actor_loss + self.args.value_coeff * critic_loss
        total_loss.backward()

        log['god_action_loss'] = actor_loss.item()
        log['god_value_loss'] = critic_loss.item()
        log['god_total_loss'] = total_loss.item()

        return log


    def collect_batch_data(self, batch_size):
        batch_data = []
        god_batch_data = []
        batch_log = dict()
        num_episodes = 0

        while len(batch_data) < batch_size:
            episode_data,episode_log = self.run_an_episode()
            batch_data += episode_data[0]
            god_batch_data += episode_data[1]
            merge_dict(episode_log, batch_log)
            num_episodes += 1

        batch_log['num_episodes'] = num_episodes
        batch_log['num_steps'] = len(batch_data)
        batch_data = Transition(*zip(*batch_data))
        god_batch_data = God_Transition(*zip(*god_batch_data))

        return (batch_data, god_batch_data), batch_log