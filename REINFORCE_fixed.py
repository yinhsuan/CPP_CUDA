# Spring 2022, IOC 5259 Reinforcement Learning
# HW1-partII: REINFORCE and baseline

import gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler

import matplotlib.pyplot as plt
import time

##############################
random_seed = 20  
lr = 0.01
environment = 'LunarLander-v2'
# number of episode for 1 update
batch_size = 4
##############################

        
class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the shared layer(s), the action layer(s), and the value layer(s)
            2. Random weight initialization of each layer
    """
    def __init__(self):
        super(Policy, self).__init__()
        
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]

        self.c1 = nn.Linear(self.observation_dim, 16)
        self.c2 = nn.Linear(16, 16)

        self.a1 = nn.Linear(16, 16)
        self.a2 = nn.Linear(16, self.action_dim)

        self.v1 = nn.Linear(16, 16)
        self.v2 = nn.Linear(16, 1)
        

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding 
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """

        x = F.relu(self.c1(state))
        x = F.relu(self.c2(x))

        a = F.relu(self.a1(x))
        action_prob = F.softmax(self.a2(a), dim=1)

        v = F.relu(self.v1(x))
        state_value = F.relu(self.v2(v))

        # action_prob -> torch.Tensor
        # state_value -> torch.Tensor

        return action_prob, state_value


    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """

        probs, state_value = self.forward(state)
        m = Categorical(probs)
        action = m.sample(batch_size)        

        # action.item() -> int
        # m.log_prob(action) -> torch.Tensor
        # state_value -> torch.Tensor

        return action, m.log_prob(action), state_value

def calculate_return(rewards, gamma=0.99):
    # Tensor: rewards.shape = [N, batch_size]
    rewards = torch.stack(rewards, axis=0)
    # Tensor: reversed_rewards.shape = [N, batch_size]
    reversed_rewards = torch.flip(rewards, dims=[0])
    g_t = torch.zeros(batch_size)
    gamma_list = torch.full(batch_size, gamma)

    returns = []
    for r in reversed_rewards.size(dim=0): # [N]
        g_t = torch.add(r, torch.mul(gamma_list, g_t))
        returns.insert(0, g_t.float())
    
    return torch.squeeze(returns)


def calculate_loss(log_probs, values, returns):
    # log_probs = torch.cat(log_probs)
    log_probs = torch.stack(log_probs, axis=0)
    log_probs = torch.squeeze(log_probs, axis=0) # [N, batch_size]

    values = torch.stack(values, axis=0)
    values = torch.squeeze(values) # [N, batch_size]

    returns = torch.tensor(returns)
    returns = torch.stack(returns, axis=0) # [N, batch_size]

    values = torch.squeeze(values)
    returns =  torch.sub(returns, torch.mean(returns, 0)) / torch.std(returns, 0)

    advantage = returns  - values
    policy_lose = torch.sum(-log_probs * advantage, dim=0)
    value_loss = torch.sum((returns - values)**2, dim=0)
    
    return policy_lose + value_loss


def train(lr=0.01, batch_size=32):
    '''
        Train the model using SGD (via backpropagation)
        TODO: In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode
    '''    
    ER = []
    R = []

    state_list = torch.empty(batch_size)
    action_list = torch.empty(batch_size)
    reward_list = torch.empty(batch_size)
    done_list = torch.empty(batch_size)
    t_ep_list = torch.zeros(batch_size)
    cmp_list = torch.zeros(10, dtype=torch.bool)
    max_list = torch.full(batch_size, 9999)
    ep_reward_list = torch.zeros(batch_size)
    log_prob_list = torch.empty(batch_size)
    value_list = torch.empty(batch_size)
    returns_list = torch.zeros(batch_size)
    # ones_list = torch.one(batch_size)
    done_mask = torch.ones(batch_size)

    print("goal: ", env.spec.reward_threshold)
    
    # Instantiate the policy model and the optimizer
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    # EWMA reward for tracking the learning progress
    ewma_reward_list = torch.zeros(batch_size)

    time_start = time.time()
    
    # run inifinitely many episodes
    i_episode = 0

    while True:
        # reset environment and episode reward
        ep_reward_list = torch.zeros(batch_size)
        
        log_probs = []
        values = []
        rewards = []
        returns = []

        # for i in range(batch_size):
        tmp_list = []
        for index in batch_size:
            # reset environment and episode reward
            tmp = env_list[index].reset()
            # Tensor: tmp = [1, 8]
            tmp = torch.FloatTensor(tmp).unsqueeze(0)
            tmp_list.append(tmp)
        state_list = torch.stack(tmp_list)
        state_list = torch.squeeze(state_list)

        t_ep_list = torch.zeros(batch_size)
        t = 0

        while torch.count_nonzero(done_mask).item() != 0:
            torch.add(t_ep_list, 1)
            # take a step
            action_list, log_prob_list, value_list = model.select_action(state_list)

            target = {'s_list': [], 'r_list': [], 'd_list': []}
            for index in batch_size:
                state, reward, done, _ = env_list[index].step(action_list[index].item())
                target['s_list'].append(state)
                target['r_list'].append(reward)
                target['d_list'].append(done)
            state_list = torch.stack(target['s_list'])
            reward_list = torch.stack(target['r_list'])
            done_list = torch.stack(target['d_list'])

            # log data
            log_probs.append(torch.mul(log_prob_list, done_mask))
            values.append(torch.mul(value_list, done_mask))
            rewards.append(torch.mul(reward_list, done_mask))

            # update done_mask
            cmp_list = torch.lt(t_ep_list, max_list)
            done_mask = done_mask & (~done_list) & cmp_list

            # if done_list[index]:
            #     break
        
        t += torch.sum(t_ep_list)
        ep_reward_list = torch.sum(rewards, dim=0)
        returns += calculate_return(rewards)

        loss = calculate_loss(log_probs, values, returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        torch.div(ep_reward_list, batch_size)
        t /= batch_size
            
        # update EWMA reward and log the results
        ewma_reward_list = torch.add(torch.mul(0.05, ep_reward_list), torch.mul((1 - 0.05), ewma_reward_list))
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, torch.sum(ep_reward_list), torch.sum(ewma_reward_list)))

        ER.append(torch.sum(ewma_reward_list))
        R.append(torch.sum(ep_reward_list))

        # check if we have "solved" the cart pole problem
        if torch.sum(ewma_reward_list) > env.spec.reward_threshold or i_episode >= 2000:
            time_end = time.time()
            torch.save(model.state_dict(), './preTrained/LunarLander_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(torch.sum(ewma_reward_list), t))

            # plt.plot(range(1, i_episode+1), R, 'r:')
            # plt.plot(range(1, i_episode+1), ER, 'b')
            # plt.legend(['ewma reward', 'ep reward'])
            # plt.savefig('LunarLander.png')
            # plt.show()
            time_c= time_end - time_start
            print('time cost', time_c, 's')
            break


def test(name, n_episodes=10):
    '''
        Test the learned model (no change needed)
    '''      
    model = Policy()
    
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    render = True
    max_episode_len = 10000
    
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    

if __name__ == '__main__':
    lr = 0.01
    random_seed = 20 # Seed should be different among processes
    batch_size = 32
    
    env_list = []
    for i in range(batch_size):
        env = gym.make('LunarLander-v2')
        env.seed(random_seed) 
        env_list.append(env)
 
    torch.manual_seed(random_seed)  
    train(lr, batch_size)
    test('LunarLander_0.01.pth')