import os
from .UAV import UAV
from .Scene import Scene
from .Controller import Controller
import numpy as np
import pybullet as p


def _get_diff(ang, target):
    diff = ang - target
    if diff < -np.pi:
        diff += 2*np.pi
    elif diff > np.pi:
        diff -= 2*np.pi
    return diff


class XControlEnv:
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
        self.scene = None
        self.current_pos = self.last_pos = None
        self.current_ori = self.last_ori = None
        self.current_vel = self.last_vel = None
        self.current_ang_vel = self.last_ang_vel = None
        self.target = None
        self.uav = None

        self.height_controller = Controller(path=self.path, prefix='Height_')
        self.pitch_controller = Controller(path=self.path, prefix='Pitch_')

    def close(self):
        p.disconnect(self.client)

    def reset(self, target=None):
        if p.isConnected():
            p.disconnect(self.client)
        self.client = p.connect(p.GUI if self.render else p.DIRECT)
        self.time = 0.
        self.scene = Scene(client=self.client,
                           time_step=self.time_step)
        base_pos = np.array([0., 0., 0.])
        base_ori = np.array([0., 0., 0.])
        self.current_pos = self.last_pos = np.array(base_pos)
        self.current_ori = self.last_ori = np.array(base_ori)
        self.current_vel = self.last_vel = np.array([0., 0., 0.])
        self.current_ang_vel = self.last_ang_vel = np.array([0., 0., 0.])

        height_target = np.random.rand() * 10. - 5.
        x_target = np.random.rand() * 10. - 5.
        self.target = np.array([x_target, 0., height_target]) if target is None else target

        self.uav = UAV(path=self.path,
                       client=self.client,
                       time_step=self.time_step,
                       base_pos=base_pos,
                       base_ori=p.getQuaternionFromEuler(base_ori))
        return self._get_s()

    def step(self, a):
        h_s = self._get_height_s()
        pitch_target = a * np.pi / 6
        p_s = self._get_pitch_s(pitch_target)

        h_a = self.height_controller.get_action(h_s)
        r_a = [0] # self.roll_controller.get_action(r_s)
        p_a = self.pitch_controller.get_action(p_s)
        y_a = [0] # self.yaw_controller.get_action(y_s)

        self.last_pos = self.current_pos
        self.last_ori = self.current_ori
        self.last_vel = self.current_vel
        self.last_ang_vel = self.current_ang_vel

        self.uav.apply_action(h_a[0],
                              r_a[0],
                              p_a[0],
                              y_a[0],
                              self.current_ori,
                              self.current_ang_vel,
                              self.time)
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

    def _get_height_s(self):
        hight = self.current_pos[2]
        velocity = self.current_vel[2]
        acceleration = (self.current_vel[2] - self.last_vel[2])/self.time_step
        target = self.target[2]
        s = [hight-target, velocity, acceleration]
        return s

    def _get_roll_s(self):
        roll = self.current_ori[0]
        r_v = self.current_ang_vel[0]
        r_acc = (self.current_ang_vel[0] - self.last_ang_vel[0]) / self.time_step
        target = self.target[1]
        diff = _get_diff(roll, target)
        s = [diff, r_v, r_acc]
        return s

    def _get_pitch_s(self, pitch_target):
        pitch = self.current_ori[1]
        p_v = self.current_ang_vel[1]
        p_acc = (self.current_ang_vel[1] - self.last_ang_vel[1]) / self.time_step
        target = pitch_target
        diff = _get_diff(pitch, target)
        s = [diff, p_v, p_acc]
        return s

    def _get_yaw_s(self):
        yaw = self.current_ori[2]
        y_v = self.current_ang_vel[2]
        y_acc = (self.current_ang_vel[2] - self.last_ang_vel[2]) / self.time_step
        target = self.target[3]
        diff = _get_diff(yaw, target)
        s = [diff, y_v, y_acc]
        return s

    def _get_s(self):
        x = self.current_pos[0]
        velocity = self.current_vel[0]
        acceleration = (self.current_vel[0] - self.last_vel[0]) / self.time_step
        target = self.target[0]

        pitch = self.current_ori[1]
        pitch_v = self.current_ang_vel[1]
        pitch_acc = (self.current_ang_vel[1] - self.last_ang_vel[1]) / self.time_step

        s = [x - target, velocity, acceleration, pitch, pitch_v, pitch_acc]
        return s

    def _get_r(self):
        last_x = self.last_pos[0]
        current_x = self.current_pos[0]
        target = self.target[0]
        last_diff = np.abs(last_x - target)
        current_diff = np.abs(current_x - target)
        r = last_diff - current_diff
        return r
