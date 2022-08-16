import numpy as np
import matplotlib.pyplot as plt
from EnvUAV.Env import PositionControlEnv
import os
import argparse
import json

path = os.path.dirname(os.path.realpath(__file__))
env = PositionControlEnv()
x = []
x_target = []
roll = []

y = []
y_target = []
pitch = []

z = []
z_target = []
yaw = []

pos = []
ang = []
targets = np.array([[10, -10, 5, 0],
                    [0, 0, 0, 0]])
s = env.reset(base_ori=[-np.pi/6, -np.pi/6, np.pi/6])
for i in range(1):
    target = targets[i, :]
    print(target)
    for ep_step in range(1000):
        env.step(target)

        pos.append(env.current_pos.tolist())
        ang.append(env.current_ori.tolist())

        x.append(env.current_pos[0])
        x_target.append(target[0])
        roll.append(env.current_ori[0])

        y.append(env.current_pos[1])
        # vel.append(env.current_vel[1])
        y_target.append(target[1])
        pitch.append(env.current_ori[1])

        z.append(env.current_pos[2])
        z_target.append(target[2])
        yaw.append(env.current_ori[2])

index = np.array(range(len(x))) * 0.01
zeros = np.zeros_like(index)
roll = np.array(roll) / np.pi*180
pitch = np.array(pitch) / np.pi*180
yaw = np.array(yaw) / np.pi*180
plt.subplot(3, 2, 1)
plt.plot(index, x, label='x')
plt.plot(index, x_target, label='x_target')
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(index, pitch, label='pitch')
plt.plot(index, zeros)
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(index, y, label='y')
plt.plot(index, y_target, label='y_target')
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(index, roll, label='roll')
plt.plot(index, zeros)
plt.legend()

plt.subplot(3, 2, 5)
plt.plot(index, z, label='z')
plt.plot(index, z_target, label='z_target')
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(index, yaw, label='yaw')
plt.plot(index, zeros)
plt.legend()

plt.show()

with open(path + '/TD3_fix_pos.json', 'w') as f:
    json.dump(pos, f)
with open(path + '/TD3_fix_ang.json', 'w') as f:
    json.dump(ang, f)