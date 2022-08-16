import pybullet as p
import pybullet_data


class Sence(object):
    def __init__(self,
                 client=0,
                 time_step=0.01,
                 g=0.):
        self.client = client
        self.time_step = time_step
        self.G = g
        self.construct()

    def construct(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.setGravity(0., 0., self.G)
        p.setTimeStep(self.time_step)
