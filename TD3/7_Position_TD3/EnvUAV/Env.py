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


class PositionControlEnv:
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
        self.ang_bound = np.pi/6

        self.x_controller = Controller(path=self.path, prefix='X_', s_dim=6)
        self.y_controller = Controller(path=self.path, prefix='Y_', s_dim=6)
        self.height_controller = Controller(path=self.path, prefix='Height_')
        self.roll_controller = Controller(path=self.path, prefix='Pitch_')
        self.pitch_controller = Controller(path=self.path, prefix='Pitch_')
        self.yaw_controller = Controller(path=self.path, prefix='Yaw_')

    def close(self):
        p.disconnect(self.client)

    def reset(self, target=None, base_ori=None):
        if p.isConnected():
            p.disconnect(self.client)
        self.client = p.connect(p.GUI if self.render else p.DIRECT)
        self.time = 0.
        self.scene = Scene(client=self.client,
                           time_step=self.time_step)
        base_pos = np.array([0., 0., 0.])
        base_ori = np.array([0., 0., 0.]) if base_ori is None else base_ori
        self.current_pos = self.last_pos = np.array(base_pos)
        self.current_ori = self.last_ori = np.array(base_ori)
        self.current_vel = self.last_vel = np.array([0., 0., 0.])
        self.current_ang_vel = self.last_ang_vel = np.array([0., 0., 0.])
        self.uav = UAV(path=self.path,
                       client=self.client,
                       time_step=self.time_step,
                       base_pos=base_pos,
                       base_ori=p.getQuaternionFromEuler(base_ori))

    def step(self, target):
        x_s = self._get_x_s(target[0])
        pitch_target = self.y_controller.get_action(x_s)
        pitch_target = -pitch_target[0] * self.ang_bound
        pitch_s = self._get_pitch_s(pitch_target)
        p_a = self.pitch_controller.get_action(pitch_s)

        y_s = self._get_y_s(target[1])
        roll_target = self.y_controller.get_action(y_s)
        roll_target = roll_target[0] * self.ang_bound
        roll_s = self._get_roll_s(roll_target)
        r_a = self.roll_controller.get_action(roll_s)

        h_s = self._get_height_s(target[2])
        h_a = self.height_controller.get_action(h_s)

        yaw_s = self._get_yaw_s(0.)
        y_a = self.yaw_controller.get_action(yaw_s)

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
        # 在环境当中，我们均以np.array的形式来存储。
        self.current_pos = np.array(current_pos)
        self.current_ori = np.array(current_ori)
        self.current_vel = np.array(current_vel)
        self.current_ang_vel = np.array(current_ang_vel)

    def _get_x_s(self, x_target):
        x = self.current_pos[0]
        velocity = self.current_vel[0]
        acceleration = (self.current_vel[0] - self.last_vel[0]) / self.time_step

        pitch = self.current_ori[1]
        pitch_v = self.current_ang_vel[1]
        pitch_acc = (self.current_ang_vel[1] - self.last_ang_vel[1]) / self.time_step

        s = [x - x_target, velocity, acceleration, -pitch, -pitch_v, -pitch_acc]
        return s

    def _get_y_s(self, y_target):
        y = self.current_pos[1]
        velocity = self.current_vel[1]
        acceleration = (self.current_vel[1] - self.last_vel[1]) / self.time_step

        roll = self.current_ori[0]
        roll_v = self.current_ang_vel[0]
        roll_acc = (self.current_ang_vel[0] - self.last_ang_vel[0]) / self.time_step

        s = [y - y_target, velocity, acceleration, roll, roll_v, roll_acc]
        return s

    def _get_height_s(self, height_target):
        height = self.current_pos[2]
        velocity = self.current_vel[2]
        acceleration = (self.current_vel[2] - self.last_vel[2])/self.time_step
        s = [height-height_target, velocity, acceleration]
        return s

    def _get_roll_s(self, roll_target):
        roll = self.current_ori[0]
        r_v = self.current_ang_vel[0]
        r_acc = (self.current_ang_vel[0] - self.last_ang_vel[0]) / self.time_step
        diff = _get_diff(roll, roll_target)
        s = [diff, r_v, r_acc]
        return s

    def _get_pitch_s(self, pitch_target):
        pitch = self.current_ori[1]
        p_v = self.current_ang_vel[1]
        p_acc = (self.current_ang_vel[1] - self.last_ang_vel[1]) / self.time_step
        diff = _get_diff(pitch, pitch_target)
        s = [diff, p_v, p_acc]
        return s

    def _get_yaw_s(self, yaw_target):
        yaw = self.current_ori[2]
        y_v = self.current_ang_vel[2]
        y_acc = (self.current_ang_vel[2] - self.last_ang_vel[2]) / self.time_step
        diff = _get_diff(yaw, yaw_target)
        s = [diff, y_v, y_acc]
        return s
