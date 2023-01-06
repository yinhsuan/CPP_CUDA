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

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

##############################
# random_seed = 4
# lr = 0.01
# environment = 'LunarLander-v2'
# number of episode for 1 update
# batch_size = 4
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

        return action_prob, state_value


    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        state = state.to(device)
        # Tensor: probs.shape = [32, 4]
        probs, state_value = self.forward(state) # Tensor: state_value.shape = [32, 1] -> [32]
#         print('state_value: ', state_value.shape)
        state_value = state_value.squeeze(1) # Tensor: state_value.shape = [32]

        m = Categorical(probs)
        action = m.sample() # Tensor: action.shape = [32]
        
        return action, m.log_prob(action), state_value


def calculate_return(rewards, t_ep_torch, gamma=0.99):
    reversed_rewards = torch.flip(rewards, dims=[0]).to(device) # Tensor: reversed_rewards.shape = [N, 32]
    g_t = torch.zeros(batch_size).to(device)
    gamma_torch = torch.empty(batch_size).fill_(gamma).to(device)

    returns = []
    for batch in reversed_rewards: # [N]
        g_t = torch.add(batch, torch.mul(gamma_torch, g_t))
        returns.insert(0, g_t.float())
    returns = torch.stack(returns, axis=0) # Tensor: returns.shape = [N, 32]
    
#     print('before returns: ', returns)
    
    # handle the direction(padding) of returns
#     print('returns: ', returns.shape)
#     tmp = []
#     returns = torch.split(returns, 1, dim=1)
# #     print('split returns: ', returns)
#     for i in range(0, len(returns)):
#         print('returns[i]: ', returns[i])
#         tmp.append(torch.roll(returns[i], -(len(returns[i]) - t_ep_torch[i].item()), 0)) # element move upward
#     returns = torch.cat(tuple(tmp), 1)
#     print('after returns: ', returns)
    
    return returns


def calculate_loss(log_probs, values, returns, t_ep_torch, batch_size):
    log_probs = torch.stack(log_probs, axis=0).to(device)
    values = torch.stack(values, axis=0).to(device)
    # print('before returns: ', len(returns), ', ', returns[0].shape)
    # returns = torch.stack(returns, axis=0).to(device)
    # print('after returns: ', returns.shape)

    returns = returns.to(device)
    t_ep_torch = t_ep_torch.to(device)
    
    # mean = torch.sum(returns, dim=0) / torch.count_nonzero(returns, dim=0) # Tensor: mean.shape = [32]
    # std = torch.sqrt(torch.sum((returns - mean)**2, dim=0) / torch.count_nonzero(returns, dim=0)) # Tensor: std.shape = [32]
    mean = torch.sum(returns, dim=0) / t_ep_torch # Tensor: mean.shape = [32]
    std = torch.sqrt(torch.sum((returns - mean)**2, dim=0) / t_ep_torch) # Tensor: std.shape = [32]

    returns =  (returns - mean) / std  # Tensor: returns.shape = [N, 32]

    advantage = returns - values # Tensor: advantage.shape = [N, 32]
    # policy_lose = torch.sum(torch.mul(-log_probs, advantage), dim=0) # Tensor: policy_lose.shape = [32]
    # value_loss = torch.sum(torch.sub(returns, values)**2, dim=0) # Tensor: value_loss.shape = [32]
    
    # loss = torch.sum(policy_lose + value_loss) / torch.sum(t_ep_torch)

    policy_lose = torch.sum(torch.mul(-log_probs, advantage)) # Tensor: policy_lose.shape = [32]
    value_loss = torch.sum(torch.sub(returns, values)**2) # Tensor: value_loss.shape = [32]
    
#     loss = (policy_lose + value_loss) / torch.sum(t_ep_torch)
#     loss = (policy_lose + value_loss) / batch_size
    loss = (policy_lose + value_loss)

    return loss


def train(env_list, number_of_episode_per_update, lr=0.01, batch_size=32):
    '''
        Train the model using SGD (via backpropagation)
        TODO: In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode
    '''    
    ER = []
    R = []

    # t_ep_torch = torch.zeros(batch_size, dtype=torch.int)

    print("goal: ", env.spec.reward_threshold)
    
    # Instantiate the policy model and the optimizer
    model = Policy()
    model.train()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    # EWMA reward for tracking the learning progress
    ewma_reward = 0
    ep_process = int(number_of_episode_per_update / batch_size)
    remain = number_of_episode_per_update % batch_size
    if remain > 0:
        ep_process += 1

    
    
    # run inifinitely many episodes
    i_episode = 0
    ep_reward = 0.0
    action_time = 0.0
    
    time_start_total = time.time()

    while True:
#         i_episode += (batch_size * ep_process)
        i_episode += number_of_episode_per_update
  
        t = 0.0
        loss = 0.0
        
        reward_list = []
        
        # ========================= START TIME !!!!
        time_start_action = time.time()
        
        
        # ======================= START: MAINTAIN the SAME CALCULATION
        for i in range(ep_process):
            
            
            mask = torch.ones(batch_size, dtype=torch.int) # 0: done/ 1: keep going
            mask = mask.to(device)

            t_ep_torch = torch.zeros(batch_size, dtype=torch.int)
            t_ep_torch = t_ep_torch.to(device)

            log_probs = []
            values = []
            rewards = []
            ep_reward = 0.0

            # reset environment
            state_init = []
            for index in range(batch_size):
                state_i = env_list[index].reset()
                state_i = torch.FloatTensor(state_i).unsqueeze(0) # Tensor: state_i.shape = [1, 8]
                state_init.append(state_i)
            state_torch = torch.stack(state_init, axis=0) # Tensor: state_torch.shape = [32, 1, 8]
            state_torch = torch.squeeze(state_torch) # Tensor: state_torch.shape = [32, 8]
            if state_torch.dim() == 1:
                state_torch = torch.unsqueeze(state_torch, 0)

            while torch.count_nonzero(mask).item() != 0:
                t_ep_torch = torch.add(t_ep_torch, mask)

                # Tensor: action_torch.shape = [32]
                # Tensor: log_prob_torch.shape = [32]
                # Tensor: value_torch.shape = [32]
                action_torch, log_prob_torch, value_torch = model.select_action(state_torch)
                action_torch = action_torch.to(device)
                log_prob_torch = log_prob_torch.to(device)
                value_torch = value_torch.to(device)
                
                state_list = []
                reward_list = []
                done_list = []
                for index in range(batch_size):
                    state, reward, done, _ = env_list[index].step(action_torch[index].item())

                    # if (not done) & (mask[index]):
                    if mask[index]:
                        s = torch.tensor(state)
                        r = torch.tensor(reward)
                        d = torch.tensor(done)
                    else:
                        s = torch.tensor(state).detach()
                        r = torch.tensor(reward).detach()
                        d = torch.tensor(done).detach()

                    state_list.append(s) # do not convert to tensor
                    reward_list.append(r)
                    done_list.append(d)
                    
                state_torch = torch.stack(state_list).to(device) # Tensor: state_torch.shape = [32, 8]
                reward_torch = torch.stack(reward_list).to(device) # Tensor: reward_torch.shape = [32]
                done_torch = torch.stack(done_list).to(device) # Tensor: done_torch.shape = [32]

                # log data
                log_probs.append(torch.mul(log_prob_torch, mask))
                values.append(torch.mul(value_torch, mask))
                rewards.append(torch.mul(reward_torch, mask))
                
                # update mask
                cmp_torch = torch.lt(t_ep_torch, 9999)
                mask = mask & (~done_torch) & cmp_torch

            rewards = torch.stack(rewards, axis=0) # Tensor: rewards.shape = [N, 32] 
            ep_reward += torch.sum(rewards)

            returns = calculate_return(rewards, t_ep_torch) # Tensor: returns.shape = [N, 32]
            loss += calculate_loss(log_probs, values, returns, t_ep_torch, batch_size) # Tensor: loss.shape = X

            # how many ACTIONs per episode
            t += torch.sum(t_ep_torch).detach()

        # ======================= END: MAINTAIN the SAME CALCULATION

        ep_reward /= (batch_size * ep_process)
        t /= (batch_size * ep_process)
#         loss /= ep_process => wrong
#         loss /= (batch_size * ep_process)

        optimizer.zero_grad()
        loss.backward()

        # ========================= END TIME !!!!
        time_end_action = time.time()
        time_c = time_end_action - time_start_action
#         print('time cost', time_c, 's')
        
        action_time += time_c
        
        optimizer.step()
        scheduler.step()

        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
#         print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        ER.append(ewma_reward.cpu().numpy())
        R.append(ep_reward.cpu().numpy())

        # check if we have "solved" the cart pole problem
#         if ewma_reward > env.spec.reward_threshold or i_episode >= 3000:
        if i_episode >= 3000:
            time_end_total = time.time()
            total_time= time_end_total - time_start_total
            
            f = open("total_time.txt", "a")
            f.write(f"{total_time}\n")
            f.close()
            
            f = open("action_time.txt", "a")
            f.write(f"{action_time}\n")
            f.close()
            
            
            
            
#             torch.save(model.state_dict(), './preTrained/LunarLander_{}.pth'.format(lr))
#             print("Solved! Running reward is now {} and "
#                   "the last episode runs to {} time steps!".format(torch.sum(ewma_reward), t))

#             plt.plot(range(1, len(R)+1), R, 'r:')
#             plt.plot(range(1, len(ER)+1), ER, 'b')
#             plt.legend(['ewma reward', 'ep reward'])
#             plt.savefig('LunarLander.png')
#             plt.show()
            
            
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
        state = torch.FloatTensor(state).unsqueeze(0)
        running_reward = 0
        for t in range(max_episode_len+1):
            action, _, _ = model.select_action(state)
            state, reward, done, _ = env.step(action.item())
            state = torch.tensor(state)
            state = torch.unsqueeze(state, 0)
            running_reward += reward
#             if render:
#                  env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()


if __name__ == '__main__':
    lr = 0.01
     # Seed should be different for each batch_size
#     random_seed = 7 # 1 -> 20/ 4 -> ?/ 8 -> ?/ 16 -> ?/ 32 -> ?
#     random_seed = 4
    random_seed = 20 # tried from 20 - 32 (when batch_size = 1)
    batch_size = 8
    
    number_of_episode_per_update = 8
    
#     if torch.cuda.is_available():
#         device = "cuda:0"
#     else:
#         device = "cpu"
    dev = 'cuda' if torch.cuda.is_available() else 'cpu' 
        
    device = torch.device(dev)
    
    env_list = []
    for i in range(batch_size):
        env = gym.make('LunarLander-v2')
        env.seed(random_seed) 
        env_list.append(env)
 
    torch.manual_seed(random_seed)  
    for i in range(5):
        train(env_list, number_of_episode_per_update, lr, batch_size)
        
    print("All Done!!!")
#     test('LunarLander_0.01.pth')