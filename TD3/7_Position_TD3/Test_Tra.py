import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from EnvUAV.Env import PositionControlEnv
import os
import time
import json

path = os.path.dirname(os.path.realpath(__file__))
env = PositionControlEnv(render=False)

length = 5000
index = np.array(range(length))/length#*2

# tx = 2*np.sin(2*np.pi*index)*np.cos(np.pi*index)
# ty = 2*np.sin(2*np.pi*index)*np.sin(np.pi*index)
# tz = -np.sin(2*np.pi*index)*np.cos(np.pi*index)-np.sin(2*np.pi*index)*np.sin(np.pi*index)

tx = 2*np.cos(2*np.pi*index)
ty = 2*np.sin(2*np.pi*index)
tz = -np.cos(2*np.pi*index)-np.sin(2*np.pi*index)


targets = np.vstack([tx, ty, tz]).T
print(targets.shape)


position_record = []
target_record = []
error_record = []

env.reset()
for i in range(length):
    target = targets[i, :]
    env.step(target)
    pos = env.current_pos
    position_record.append(pos.tolist())
    error_record.append(np.linalg.norm(pos-target))
    target_record.append(target)

print(np.average(error_record))

fig = plt.figure()
ax = fig.gca(projection='3d')

position = np.array(position_record)
px = position[:, 0]
py = position[:, 1]
pz = position[:, 2]

ax.plot(px, py, pz, label='track')
ax.plot(tx, ty, tz, label='target')
ax.view_init(azim=30., elev=10)

plt.show()

with open(path + '/TD3_circle.json', 'w') as f:
    json.dump(position_record, f)
# with open(path + '/TD3_quad.json', 'w') as f:
#     json.dump(position_record, f)