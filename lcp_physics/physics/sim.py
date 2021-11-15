import os
import sys

import time
import math
from math import sin, cos
import cv2
import pygame
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from torch.autograd import Variable

from .bodies import Circle, Rect, Hull, Composite
from .constraints import TotalConstraint, FixedJoint
from .forces import ExternalForce, Gravity, vert_impulse, hor_impulse
from .utils import Defaults, plot,    Recorder
from .world import World, run_world
from .action import build_mesh, random_action


def image_to_pos(mass_img):
    img = cv2.imread(mass_img)
    mask = img[:,:,0] < 255
    x_cord, y_cord = np.where(mask)
    x_cord, y_cord = x_cord - x_cord.min(), y_cord - y_cord.min()
    particle_pos = 2 * particle_radius * np.stack([x_cord, y_cord]).T
    return particle_pos, mask

def image_to_mass(mass_img, mask):
    img = cv2.imread(mass_img)
    mass_profile = img[:,:,0][mask].astype('float') / 1e3
    return mass_profile

def image_to_bottom_fric(fric_img, mask):
    img = cv2.imread(os.path.join(ROOT, fric_img)).astype('float') / 255
    bottom_fric_profile = np.stack([img[:,:,2][mask]/100, img[:,:,1][mask]], axis=-1)
    return bottom_fric_profile


class SimSingle():
    def __init__(self, particle_pos0, particle_radius, hand_radius, mass_gt=None, 
    			bottom_fric_gt=None, mass_est=None, bottom_fric_est=None, mask=None,
    			DT = Defaults.DT, DEVICE = Defaults.DEVICE):
		
		self.particle_pos0 = particle_pos0
		self.particle_radius = particle_radius
		self.N = self.particle_pos0.shape[0]

		if not mass_gt:
			self.mass_gt = 0.01 * torch.ones(self.N).to(DEVICE)
		else:
			self.mass_gt = mass_gt

		if not bottom_fric_gt:
			self.bottom_fric_gt = torch.FloatTensor([0.001, 0.1]).repeat(self.N, 1).to(DEVICE)
		else:
			self.bottom_fric_gt = bottom_fric_gt

    	if not mass_est:
			self.mass_est = 0.01 * torch.ones(self.N).to(DEVICE)
			self.mass_est = Variable(self.mass_est, requires_grad=True)
		else:
			self.mass_est = mass_est

		self.hand_radius = hand_radius
		self.mask = mask

	@classmethod
	def from_img(cls, mass_img_path, bottom_fric_img_path, particle_radius=None, 
					hand_radius=None):
		particle_pos0, mask = image_to_pos(mass_img_path)
    
        mass_gt = image_to_mass(mass_img_path, mask)
        bottom_fric_gt = image_to_bottom_fric(bottom_fric_img_path, mask)
        cls(particle_pos0, particle_radius, hand_radius, mass_gt=mass_gt, 
        	bottom_fric_gt=bottom_fric_gt, mask=mask)    

    def random_rest_composite_pose(self):
        # randomly initialize rotation and offset
        rotation = np.random.uniform() * 2* math.pi
        offset = np.random.uniform(low=[450, 250], high=[550, 350], size=2)
        return rotation, offset

    def sample_action(self, composite_body):
        # get composite object particle pos
        p = composite_body.get_particle_pos()
        # init random action (start position and direction)
        action = random_action(p, self.particle_radius, self.hand_radius)
        return action

    def run_episode(self, action):
        rotation, offset = self.random_rest_composite_pose()
        # init composite object with offset and rotation
        composite_body_gt = init_composite_object(self.particle_pos0,
                                               self.particle_radius, 
                                               self.mass_gt,
                                               self.bottom_fric_gt,
                                               rotation=rotation,
                                               offset=offset)
        if action is None:
	        action = self.sample_action(composite_body_gt)
        world = make_world(self.particle_pos0, composite_body_gt, action)
        recorder = None
        # recorder = Recorder(DT, screen)
        run_world(world, run_time=TIME, screen=screen, recorder=recorder)

        X1 = composite_body_gt.get_particle_pos()
        
        composite_body = init_composite_object(self.particle_pos0,
                                               self.particle_radius, 
                                               self.mass,
                                               self.bottom_fric_gt,
                                               rotation=rotation,
                                               offset=offset)
        world = make_world(self.particle_pos0, composite_body, action)
        run_world(world, run_time=TIME, screen=screen, recorder=recorder)

        X2 = composite_body.get_particle_pos()

        return rotation, offset, action, X1, X2