import numpy as np
import matplotlib.pyplot as plt
from EnvUAV.Env import HeightControlEnv
from TD3 import TD3
import os
import time
import json

path = os.path.dirname(os.path.realpath(__file__))
env = HeightControlEnv()
agent = TD3(path)
total_steps = 0
agent.var = 0.
# selected agent
agent.load_net('', '480')
height = []
action = []
velocity = []
acc = []
target = []
targets = [5, 0, 1]

s = env.reset()
for i in range(1):
    env.target = targets[i]
    for ep_step in range(500):
        # if RENDER:
        #     env.render()
        a = agent.get_action(s)
        s_, r, done, info = env.step(a[0])
        s = s_

        target.append(env.target)
        height.append(env.current_pos[2])
        action.append(a[0])
        velocity.append(s[1])
        acc.append(s[2])

print(s[0])
index = np.array(range(len(target)))*0.01
plt.plot(index, target, label='target')
plt.plot(index, height, label='height')
# plt.plot(index, velocity, label='velocity')
# plt.plot(index, acc, label='acceleration')
plt.legend()
plt.show()

# with open(path + '/TD3_Z.json', 'w') as f:
#     json.dump(height, f)

