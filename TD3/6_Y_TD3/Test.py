import numpy as np
import matplotlib.pyplot as plt
from EnvUAV.Env import YControlEnv
from TD3 import TD3
import os
import time
import json

path = os.path.dirname(os.path.realpath(__file__))
env = YControlEnv()
agent = TD3(path, s_dim=6)
total_steps = 0
agent.var = 0.
agent.load_net('', '58')
# 251
height = []
action = []
target = []
targets = np.array([[0, 5, 0],
                   [0, 0, 0],
                   [0, 3, 0]])

s = env.reset()
for i in range(1):
    env.target = targets[i]
    for ep_step in range(500):
        a = agent.get_action(s)
        s_, r, done, info = env.step(a[0])
        s = s_

        height.append(env.current_pos[1])
        target.append(env.target[1])
        action.append(a[0])

    print(s[0])

index = np.array(range(len(target)))*0.01
plt.plot(index, target, label='target')
plt.plot(index, height, label='y')
plt.legend()
plt.show()

with open(path + '/TD3_Y.json', 'w') as f:
    json.dump(height, f)