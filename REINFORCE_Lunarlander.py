import torch
from torch import nn
import gym 
import numpy as np
from collections import deque

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net.forward(x)

def _to_cuda_tensor(x):
    return torch.tensor(x, dtype=torch.float32, device='cuda')

class Agent:
    def __init__(self): 
        self.gamma=.99
        self.policy=Policy().to('cuda')
        self.policy_optimizer=torch.optim.Adam(self.policy.parameters())
    
    def decide_action(self, state:np.ndarray): 
        probs=self.policy.forward(_to_cuda_tensor(state)).detach().cpu().numpy()
        action=np.random.choice((0, 1, 2, 3), p=probs)
        return action
    
    def learn(self, memory:deque):
        states, actions, rewards=zip(*memory)
        states=torch.stack(tuple(map(_to_cuda_tensor, states)))

        pred_probs=self.policy.forward(states)[np.arange(len(memory)), actions]
        log_probs=torch.log(pred_probs)
        rewards=_to_cuda_tensor(rewards)
        discounted_rewards=torch.zeros_like(rewards)
        
        for i in range(rewards.shape[0]):
            for pow, j in enumerate(rewards[i:]):
                discounted_rewards[i]+=j*(self.gamma**pow)
        
        policy_grad=(-log_probs*discounted_rewards).sum()

        self.policy_optimizer.zero_grad()
        policy_grad.backward()
        self.policy_optimizer.step()

env=gym.make('LunarLander-v2')
agent=Agent()

for episode in range(500):
    memory=deque()
    i_state=env.reset()
    while True:
        action=agent.decide_action(i_state)
        f_state, reward, done, _=env.step(action)
        memory.append((i_state, action, reward))
        env.render()
        if done:
            agent.learn(memory)
            break
        i_state=np.copy(f_state)