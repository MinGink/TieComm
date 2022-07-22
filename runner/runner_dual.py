import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from torch.optim import Adam

from modules.utils import merge_dict
from .runner import Runner
import time

Transition = namedtuple('Transition', ('obs', 'actions', 'action_outs', 'rewards',
                                            'episode_masks', 'episode_agent_masks', 'values',
                                            'god_action_out', 'god_value', 'god_action', 'god_reward'
                                            ))


class RunnerDual(Runner):
    def __init__(self,  config, env, agent):
        super(RunnerDual, self).__init__( config, env, agent)


        self.n_agents = self.args.n_agents
        self.n_nodes = int(self.n_agents * (self.n_agents - 1)/2)




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

            set,god_action_out, god_value, god_action = self.agent.god(obs_tensor)
            after_comm = self.agent.communicate(obs_tensor, set)
            action_outs, values = self.agent.agent(after_comm)

            actions = self.choose_action(action_outs)

            rewards, dones, env_info = self.env.step(actions)

            god_reward = np.sum(rewards)

            next_obs = self.env.get_obs()

            episode_mask = np.zeros(np.array(rewards).shape)
            #episode_agent_mask = np.array(dones) + 0

            if done or step == self.args.episode_length - 1:
                episode_mask = np.ones(np.array(rewards).shape)


            trans = Transition(np.array(obs), actions, action_outs, np.array(rewards),
                                    episode_mask, episode_mask, values, god_action_out, god_value, god_action, god_reward)
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



    def collect_epoch_data(self, epoch_size):
        epoch_data = []
        epoch_log = dict()
        num_episodes = 0
        for i in range(epoch_size):
            episode_data, episode_log = self.run_an_episode()
            epoch_data += episode_data
            merge_dict(episode_log, epoch_log)
            num_episodes += 1

        epoch_data = Transition(*zip(*epoch_data))
        epoch_log['num_episodes'] = num_episodes

        return epoch_data, epoch_log
