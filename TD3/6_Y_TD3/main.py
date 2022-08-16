from EnvUAV.Env import YControlEnv
from TD3 import TD3
import matplotlib.pyplot as plt
import os
import json
import time

path = os.path.dirname(os.path.realpath(__file__))
env = YControlEnv()
for i in range(10):
    agent = TD3(path, s_dim=6)
    index1 = str(i)
    # Construct environment and agent
    if not os.path.exists(path+'/Net'+index1):
        os.makedirs(path+'/Net'+index1)
    # Train process
    for episode in range(200):
        print(index1, episode)
        s = env.reset()
        ep_r = 0.
        for step in range(512):
            a = agent.get_action(s)
            s_, r, done, _ = env.step(a[0])
            # agent.learn(s, a, s_, r)
            agent.store_transition(s, a, s_, r)
            s = s_
            ep_r += r
        agent.store_net(prefix1=index1, prefix2=str(episode))
