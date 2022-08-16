import numpy as np
import matplotlib.pyplot as plt
from EnvUAV.Env import PitchControlEnv
from TD3 import TD3
import os
import time
import json

path = os.path.dirname(os.path.realpath(__file__))
env = PitchControlEnv()
agent = TD3(path)
total_steps = 0
agent.var = 0.
# your selected index
agent.load_net('', '462')
pitch = []
action = []
target = []
targets = np.array([1, 0, 1/6])

s = env.reset()
for i in range(1):
    env.target = targets[i]
    for ep_step in range(100):
        a = agent.get_action(s)
        s_, r, done, info = env.step(a[0])
        s = s_

        pitch.append(env.current_ori[1])
        target.append(env.target)
        action.append(a[0])


index = np.array(range(len(target)))*0.01
plt.plot(index, target, label='target')
plt.plot(index, pitch, label='pitch')
plt.legend()
plt.show()

with open(path + '/TD3_Pitch.json', 'w') as f:
    json.dump(pitch, f)