import copy
import torch
import torch.nn.functional as F
from Network import Actor, Critic
from ReplayBuffer import ReplayBuffer
import numpy as np
import json


class TD3:
    def __init__(self,
                 path,
                 s_dim=3,
                 a_dim=1,
                 hidden=32,
                 capacity=int(1e4),  # there must be an int() because we will do [0] * capacity
                 batch_size=256,
                 start_learn=512,
                 lr=3e-4,
                 var_init=1.,
                 var_decay=0.9999,
                 var_min=0.1,
                 gamma=0.99,
                 tau=0.1,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2):
        # Parameter Initialization
        self.path = path
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.hidden = hidden
        self.capacity = capacity
        self.batch_size = batch_size
        self.start_learn = start_learn
        self.lr = lr
        self.var = var_init
        self.var_decay = var_decay
        self.var_min = var_min
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.train_it = 0

        # Network
        self.actor = Actor(s_dim, a_dim, hidden)
        self.actor_target = copy.deepcopy(self.actor)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(s_dim, a_dim, hidden)
        self.critic_target = copy.deepcopy(self.critic)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # replay buffer, or memory
        self.memory = ReplayBuffer(s_dim, a_dim, capacity, batch_size)

    def get_action(self, s):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float)
            a = self.actor(s).numpy()
        a = np.clip(np.random.normal(a, self.var), -1., 1.)
        return a.tolist()

    def store_transition(self, s, a, s_, r):
        self.memory.store_transition(s, a, s_, r)
        if self.memory.counter >= self.start_learn:
            s, a, s_, r = self.memory.get_sample()
            self._learn(s, a, s_, r)

    def _learn(self, s, a, s_, r):
        self.train_it += 1

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = torch.randn_like(a) * self.policy_noise
            noise = torch.clip(noise, -self.noise_clip, self.noise_clip)

            a_ = self.actor_target(s_) + noise
            a_ = torch.clip(a_, -1., 1.)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(s_, a_)
            target_Q = torch.min(target_Q1, target_Q2)
            td_target = r + self.gamma * target_Q

        # update critic
        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        if self.train_it % self.policy_freq == 0:
            # update actor
            # 两种写法都是可行的，可以直接用一个，也可以取min
            # q1, q2 = self.critic(s, self.actor(s))
            # q = torch.min(q1, q2)
            q = self.critic.Q1(s, self.actor(s))
            actor_loss = -torch.mean(q)
            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.step()

            # update target network
            self._soft_update(self.critic_target, self.critic)
            self._soft_update(self.actor_target, self.actor)

            # update varaiance
            self.var = max(self.var * self.var_decay, self.var_min)

    def store_net(self, prefix1, prefix2):
        torch.save(self.actor.state_dict(), self.path + '/Net' + prefix1 + '/' + prefix2 + '_Actor.pth')

    def load_net(self, prefix1, prefix2):
        self.actor.load_state_dict(torch.load(self.path + '/Net' + prefix1 + '/' + prefix2 + '_Actor.pth'))

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )