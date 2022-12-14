import pybullet as p
import yaml
import numpy as np
from scipy.integrate import ode


class UAV(object):
    def __init__(self,
                 path,
                 client,
                 time_step,
                 base_pos,
                 base_ori):
        self.path = path
        self.client = client
        with open(self.path+'/File/uva.yaml', 'r', encoding='utf-8') as F:
            param_dict = yaml.load(F, Loader=yaml.FullLoader)
        self.time_step = time_step
        # mass and length
        self.M = param_dict['M']
        self.L = param_dict['L']
        # thrust and torque coefficient,
        self.CT = param_dict['CT']
        self.CM = param_dict['CM']
        thrust = self.CT
        torque_xy = self.CT * self.L / np.sqrt(2)
        torque_z = self.CM
        row_weight = np.array([[thrust], [torque_xy], [torque_xy], [torque_z]])
        matrix = np.array([[1, 1, 1, 1],
                           [-1, -1, 1, 1],
                           [-1, 1, 1, -1],
                           [-1, 1, -1, 1]])
        self.MATRIX = matrix * row_weight
        self.MATRIX_INV = np.linalg.inv(self.MATRIX)
        # Moment of inertia
        self.J = param_dict['J']
        self.J_xx = self.J[0][0]
        self.J_yy = self.J[1][1]
        self.J_zz = self.J[2][2]
        self.J_R = param_dict['J_R']
        # Motor coefficient
        self.MOTOR_COEFFICIENT = param_dict['MOTOR_COEFFICIENT']
        self.MOTOR_BIAS = param_dict['MOTOR_BIAS']
        self.DRAG = param_dict['DRAG']
        # Load file
        path = self.path+'/File/cf2x.urdf'
        self.id = p.loadURDF(fileName=path,
                             basePosition=base_pos,
                             baseOrientation=base_ori,
                             physicsClientId=self.client,
                             flags=p.URDF_USE_INERTIA_FROM_FILE
                             )
        # Motor speed and its integrator
        self.motor_speed = np.array([0., 0., 0., 0.])

        self.integrator = ode(self.motor_dot).set_integrator('dopri5', first_step='0.00005', atol='10e-6', rtol='10e-6')
        self.integrator.set_initial_value(self.motor_speed)

    def motor_dot(self, t, speed, speed_d):
        speed_dot = (speed_d - speed) / self.DRAG
        return speed_dot

    def mixer(self, h_a, r_a, p_a, y_a, roll, pitch):
        h_a = (0.5 + 0.3 * h_a) / (np.cos(roll) * np.cos(pitch))
        h_a = h_a * np.array([1., 1., 1., 1.])
        r_a = 0.1 * r_a * np.array([-1., -1., 1., 1.])
        p_a = 0.1 * p_a * np.array([-1., 1., 1., -1.])
        y_a = 0.1 * y_a * np.array([-1., 1., -1., 1.])
        a = h_a + y_a + p_a + r_a
        # ?????????????????????
        min_a = np.min(a)
        if min_a < 0:
            a -= min_a
        max_a = np.max(a)
        if max_a > 1:
            a /= max_a
        u = a**0.5
        return u

    def apply_action(self, h_a, r_a, p_a, y_a, euler, ang_v, t):
        # ????????????????????????????????????0.02s
        [roll, pitch, yaw] = euler
        u = self.mixer(h_a, r_a, p_a, y_a, roll, pitch)
        motor_speed_d = self.MOTOR_COEFFICIENT * u + self.MOTOR_BIAS

        self.integrator.set_f_params(motor_speed_d)
        self.motor_speed = self.integrator.integrate(t)
        thrust_torque = np.dot(self.MATRIX, self.motor_speed ** 2)
        force = np.array([0., 0., thrust_torque[0]])

        torque1 = thrust_torque[1:]
        torque2 = np.array([ang_v[1] * ang_v[2] * (self.J_yy - self.J_zz),
                            ang_v[2] * ang_v[1] * (self.J_zz - self.J_xx),
                            ang_v[0] * ang_v[1] * (self.J_xx - self.J_yy)])
        motor_speed_sum = np.dot(self.motor_speed, [1, -1, 1, -1])
        torque3 = self.J_R * motor_speed_sum * np.array([-ang_v[1], ang_v[0], 0.])
        torque = torque1 + torque2 + torque3

        # apply force and torque
        p.applyExternalForce(objectUniqueId=self.id,
                             linkIndex=-1,
                             forceObj=force,
                             posObj=np.array([0., 0., 0.]),
                             flags=p.LINK_FRAME)
        p.applyExternalTorque(objectUniqueId=self.id,
                              linkIndex=-1,
                              torqueObj=torque,
                              flags=p.LINK_FRAME)
