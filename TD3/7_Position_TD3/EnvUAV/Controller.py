import torch
import torch.nn as nn
import numpy as np


class Controller:
    def __init__(self, path, prefix, s_dim=3, a_dim=1, hidden=32):
        self.actor = Actor(s_dim, a_dim, hidden)
        self.actor.load_state_dict(torch.load(path + '/Net/' + prefix + 'Actor.pth'), strict=False)

    def get_action(self, s):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float)
            a = self.actor(s).numpy()
        return a.tolist()


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, hidden):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(nn.Linear(s_dim, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, hidden),
                                   nn.ReLU(),
                                   # nn.Linear(hidden, hidden),
                                   # nn.ReLU(),
                                   nn.Linear(hidden, a_dim),
                                   nn.Tanh())

    def forward(self, s):
        return self.actor(s)
