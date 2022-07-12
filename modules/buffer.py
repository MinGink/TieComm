import random
from collections import namedtuple,deque
import torch
import numpy as np


class Buffer:

    def __init__(self, args):

        self.gamma = args.gamma
        self.buffer_size = args.buffer_size

        self.memory = []
        self.transition = namedtuple('Transition', ('state', 'action','reward','terminated'))

    def add(self, state, action, reward, done):

        """Saves a transition."""
        transition = self.transition(state, action, reward, done)
        self.memory.append(transition)


    def compute_return(self):

        rewards = [t.reward for t in self.memory]
        masks = [t.done for t in self.memory]

        # rewards = np.array(rewards)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-10)
        # rewards.tolist()

        returns = []
        discounted_reward = 0
        for reward, mask in zip(reversed(rewards),reversed(masks)):
            if mask:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            #print(discounted_reward)
            returns.insert(0,discounted_reward)

        return rewards, returns, masks


    def get_all_list_data(self):

        states = [t.state for t in self.memory]
        actions = [t.action for t in self.memory]


        rewards, returns, old_masks = self.compute_return()

        masks = list()
        for mask in old_masks:
            if mask:
                masks.append(1)
            else:
                masks.append(0)

        return states, actions, rewards, returns, masks,


    def reset(self):
        self.memory=[]


    def __len__(self):
        return len(self.memory)