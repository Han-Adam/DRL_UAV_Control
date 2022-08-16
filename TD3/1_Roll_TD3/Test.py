import numpy as np
import matplotlib.pyplot as plt
from EnvUAV.Env import RollControlEnv
from TD3 import TD3
import os
import time
import json

path = os.path.dirname(os.path.realpath(__file__))
env = RollControlEnv()
agent = TD3(path)
total_steps = 0
agent.var = 0.
# your selected index
agent.load_net('', '69')

roll = []
action = []
target = []
targets = np.array([1, -1/6, 0])

s = env.reset()
for i in range(1):
    env.target = targets[i]
    for ep_step in range(100):
        a = agent.get_action(s)
        s_, r, done, info = env.step(a[0])
        s = s_
        roll.append(env.current_ori[0])
        target.append(env.target)
        action.append(a[0])

index = np.array(range(len(target)))*0.01
plt.plot(index, target, label='target')
plt.plot(index, roll, label='roll')
plt.legend()
plt.show()

with open(path + '/TD3_Roll.json', 'w') as f:
    json.dump(roll, f)