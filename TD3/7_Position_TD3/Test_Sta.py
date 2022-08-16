import numpy as np
import matplotlib.pyplot as plt
from EnvUAV.Env import PositionControlEnv
import os
import json
import time


def calculate_peak(x, target):
    peak = max(x) if target >= 0 else min(x)
    return (peak-target)/target


def calculate_error(x, target):
    x = np.array(x)[-500:]
    diff = np.abs(x-target)
    return np.average(diff)


def calculate_rise(x, target):
    x = np.abs(x)
    target = np.abs(target)
    t1 = np.max(np.argwhere((x < target*0.1)))
    t2 = np.min(np.argwhere((x > target*0.9)))
    return (t2-t1)*0.01


path = os.path.dirname(os.path.realpath(__file__))
env = PositionControlEnv()


target_container = []
peak_container = []
error_container = []
rise_container = []
for i in range(500):
    target = np.random.rand(3)*4+1
    for j in range(3):
        if np.random.rand() <= 0.5:
            target[j] = -target[j]

    env.reset()
    print(i, target)
    trace = np.zeros(shape=[1000, 3])
    for ep_step in range(1000):
        trace[ep_step] = env.current_pos
        env.step(target)

    peak = [calculate_peak(trace[:, i], target[i]) for i in range(3)]
    error = [calculate_error(trace[:, i], target[i]) for i in range(3)]
    rise = [calculate_rise(trace[:, i], target[i]) for i in range(3)]

    target_container.append(target)
    peak_container.append(peak)
    error_container.append(error)
    rise_container.append(rise)

error_container = (np.array(error_container)*1000).tolist()
peak_container = (np.array(peak_container)*100).tolist()

print('error', np.mean(error_container, axis=0), np.std(error_container, axis=0))
print('rise', np.mean(rise_container, axis=0), np.std(rise_container, axis=0))
print('peak', np.mean(peak_container, axis=0), np.std(peak_container, axis=0))
# with open(path+'/FixPointRecord/peak.json', 'w') as f:
#     json.dump(peak_container, f)
# with open(path+'/FixPointRecord/peak.json', 'w') as f:
#     json.dump(peak_container, f)
# with open(path+'/FixPointRecord/error.json', 'w') as f:
#     json.dump(error_container, f)
# with open(path+'/FixPointRecord/rise.json', 'w') as f:
#     json.dump(rise_container, f)

index = np.array(range(1000))*0.01
plt.subplot(3, 1, 1)
plt.plot(index, trace[:,0], label='x')
plt.subplot(3, 1, 2)
plt.plot(index, trace[:,1], label='y')
plt.subplot(3, 1, 3)
plt.plot(index, trace[:,2], label='z')
plt.show()