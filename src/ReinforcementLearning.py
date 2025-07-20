import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd


## PPO implementation
## the actor layer decides which action to take
## the critic layer output a value that evaluate the actor decision
## more complex implementation with images can be implemented with convolutions


class WalkLearner(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor=nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Tanh()
        )

        self.critic=nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
    
    def forward(self,x):
        return self.actor(x), self.critic(x)
    
    def act(self,x):
        action = self.actor(x)
        return action

    def evaluate(self,x):
        return self.critic(x)


