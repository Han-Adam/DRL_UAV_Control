import os
from .UAV import UAV
from .Sence import Sence
import numpy as np
import pybullet as p


def _get_diff(ang, target):
    diff = ang - target
    if diff < -np.pi:
        diff += 2*np.pi
    elif diff > np.pi:
        diff -= 2*np.pi
    return diff


class YawControlEnv:
    def __init__(self,
                 model='cf2x',
                 render=False,
                 random=True,
                 time_step=0.01):
        '''
        :param model: The model/type of the uav.
        :param render: Whether to render the simulation process
        :param random: Whether to use random initialization setting
        :param time_step: time_steps
        '''
        self.render = render
        self.model = model
        self.random = random
        self.time_step = time_step
        self.path = os.path.dirname(os.path.realpath(__file__))

        self.client = None
        self.time = None
        self.sence = None
        self.current_pos = self.last_pos = None
        self.current_ori = self.last_ori = None
        self.current_vel = self.last_vel = None
        self.current_ang_vel = self.last_ang_vel = None
        self.target = None
        self.uav = None

    def close(self):
        p.disconnect(self.client)

    def reset(self, target=None):
        if p.isConnected():
            p.disconnect(self.client)
        self.client = p.connect(p.GUI if self.render else p.DIRECT)
        self.time = 0.
        self.sence = Sence(client=self.client,
                           time_step=self.time_step)
        base_pos = np.array([0., 0., 0.])
        base_ori = np.array([0., 0., 0.])
        self.current_pos = self.last_pos = np.array(base_pos)
        self.current_ori = self.last_ori = np.array(base_ori)
        self.current_vel = self.last_vel = np.array([0., 0., 0.])
        self.current_ang_vel = self.last_ang_vel = np.array([0., 0., 0.])
        self.target = (np.random.rand()-0.5)*2*np.pi if target is None else target
        self.uav = UAV(path=self.path,
                       client=self.client,
                       time_step=self.time_step,
                       base_pos=base_pos,
                       base_ori=p.getQuaternionFromEuler(base_ori))
        return self._get_s()

    def step(self, a):
        self.last_pos = self.current_pos
        self.last_ori = self.current_ori
        self.last_vel = self.current_vel
        self.last_ang_vel = self.current_ang_vel

        self.uav.apply_action(a, self.time)
        p.stepSimulation()
        self.time += self.time_step

        current_pos, current_ori = p.getBasePositionAndOrientation(self.uav.id)
        current_ori = p.getEulerFromQuaternion(current_ori)
        current_vel, current_ang_vel = p.getBaseVelocity(self.uav.id)
        self.current_pos = np.array(current_pos)
        self.current_ori = np.array(current_ori)
        self.current_vel = np.array(current_vel)
        self.current_ang_vel = np.array(current_ang_vel)

        s_ = self._get_s()
        r = self._get_r()
        done = False
        infor = None
        return s_, r, done, infor

    def _get_s(self):
        yaw = self.current_ori[2]
        y_v = self.current_ang_vel[2]
        y_acc = (self.current_ang_vel[2] - self.last_ang_vel[2]) / self.time_step
        target = self.target
        diff = _get_diff(yaw, target)
        s = [diff, y_v, y_acc]
        return s

    def _get_r(self):
        last_y = self.last_ori[2]
        current_y = self.current_ori[2]
        target = self.target
        last_diff = np.abs(_get_diff(last_y, target))
        current_diff = np.abs(_get_diff(current_y, target))
        r = last_diff - current_diff
        return r
