import os
import sys
import time
import math
from math import sin, cos
import cv2
import pygame
import scipy.spatial as spatial
import numpy as np
import torch
from torch.nn import MSELoss
from torch.autograd import Variable

from lcp_physics.physics.bodies import Circle, Rect, Hull
from lcp_physics.physics.constraints import TotalConstraint, FixedJoint, Joint
from lcp_physics.physics.forces import ExternalForce, Gravity, vert_impulse, hor_impulse
from lcp_physics.physics.utils import Defaults, plot, reset_screen, Recorder, left_orthogonal, right_orthogonal
from lcp_physics.physics.world import World, run_world
from lcp_physics.physics.block_object_util import BlockObject

class SuctionSimulator:
	def __init__(self, param_file):
		self.block_object = BlockObject(param_file)
		self.particle_pos0 = torch.from_numpy(self.block_object.particle_coord)

	def get_init_vel(self, center, vel_w):
        """
        Compute particle velocity from the rotation center and angular velocity
        Args
            center -- x and y position of the rotation center, 1D tensor
            vel_w -- angular velocity, 1D tensor
        Return
            particle_vel -- (vw, vx, vy) 2D tensor, (N, 3)
        """
        p = self.particle_pos0 - center
        particle_vel = []
        for i in range(p.shape[0]):
            v = torch.cat([vel_w, vel_w * right_orthogonal(p[i])])
            particle_vel.append(v)
        particle_vel = torch.stack(particle_vel)
        return particle_vel

		