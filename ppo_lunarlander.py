import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
import numpy as np
from collections import deque
import random

def cuda_tensor(data):
    return torch.tensor(data, dtype=torch.float32, device='cuda')

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lin0 = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.lin0(x)
        action_probs = self.actor(x)
        value = self.critic(x)

        return action_probs, value

class Agent:
    def __init__(self, eps = 0.2, batch_size = 64, timesteps = 512, epoch = 24, gamma = 0.995, entropy_scalar = 0.002):
        self.network = Model().to('cuda')
        self.epsilon = eps
        self.optimizer = optim.Adam(self.network.parameters(), 3e-4)
        self.buffer = deque(maxlen=timesteps)
        self.env = gym.make('LunarLander-v2')
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.epoch = epoch
        self.gamma = gamma
        self.entropy_scalar = entropy_scalar
    
    def choose_action(self, state):
        with torch.no_grad():
            probs, _ = self.network.forward(cuda_tensor(state))
            action = int(Categorical(probs).sample())
            return action, float(probs[action])
    
    def push_memory(self, i_state, action, prob, reward, done, f_state):
        self.buffer.append((i_state, action, prob, reward, done, f_state))

    def learn(self):
        i_states, actions, action_probs, rewards, dones, f_states = zip(*self.buffer)
        i_states = torch.stack(tuple(map(cuda_tensor, i_states)))
        f_states = torch.stack(tuple(map(cuda_tensor, f_states)))
        action_probs = cuda_tensor(action_probs)
        actions = np.array(actions)
        rewards = cuda_tensor(rewards)
        dones = cuda_tensor(dones)

        rewards = (rewards - rewards.mean()) / rewards.std()

        with torch.no_grad():
            _, f_state_vals = self.network.forward(f_states)
            _, i_state_vals = self.network.forward(i_states)
            f_state_vals = torch.squeeze(f_state_vals)
            i_state_vals = torch.squeeze(i_state_vals)
            advantages = rewards + f_state_vals * (1-dones) - i_state_vals
        
        for i in range(self.epoch):
            sample_idx = random.sample(range(self.timesteps), self.batch_size)

            i_state_sample = i_states[sample_idx, :]
            f_state_sample = f_states[sample_idx, :]
            reward_sample = rewards[sample_idx]
            action_sample = actions[sample_idx]
            advantage_sample = advantages[sample_idx]
            old_probs = action_probs[sample_idx]
            dones_sample = dones[sample_idx]

            i_probs, i_vals = map(torch.squeeze, self.network.forward(i_state_sample))

            with torch.no_grad():
                _, f_vals = self.network.forward(f_state_sample)
                f_vals = torch.squeeze(f_vals)

            r = torch.exp(torch.log(i_probs[np.arange(self.batch_size), action_sample]) - torch.log(old_probs))
            clip_loss = -torch.min(r * advantage_sample, torch.clamp(r, 1 - self.epsilon, 1+self.epsilon) * advantage_sample).mean()
            value_loss = F.smooth_l1_loss(i_vals , reward_sample + (1-dones_sample) * self.gamma * f_vals)
            entropy = -Categorical(i_probs).entropy().mean() * self.entropy_scalar

            final_loss = clip_loss+value_loss+entropy
            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()

    def play(self):
        i_state = self.env.reset()
        for j in range(self.timesteps):
            action, prob = self.choose_action(i_state)
            f_state, reward, done, _ = self.env.step(action)

            self.env.render('rgb_array')
            self.push_memory(i_state, action, prob, reward, done, f_state)
            if done:
                i_state = self.env.reset()
            else:
                i_state = np.copy(f_state)

agent = Agent()
agent.network.load_state_dict(torch.load("C:/Users/love4/Downloads/lander_ppo (1).pth"))


for k in range(100):
    agent.play()
    agent.learn()
