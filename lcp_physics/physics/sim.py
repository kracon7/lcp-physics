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

from .bodies import Circle, Rect, Hull, Composite, CompositeSquare
from .constraints import TotalConstraint, FixedJoint
from .forces import ExternalForce, Gravity, vert_impulse, hor_impulse
from .utils import Defaults, plot, Recorder, rgb2mass, mass2rgb, get_tensor
from .world import World, run_world
from .action import build_mesh, random_action


def image_to_pos(mass_img, particle_radius):
    img = cv2.imread(mass_img)
    mask = img[:,:,0] < 255
    x_cord, y_cord = np.where(mask)
    x_cord, y_cord = x_cord - x_cord.min(), y_cord - y_cord.min()
    particle_pos = 2 * particle_radius * np.stack([x_cord, y_cord]).T
    return particle_pos, mask

def image_to_mass(mass_img, mask):
    img = cv2.imread(mass_img)
    mass_profile = rgb2mass(img)[mask]
    return mass_profile

def image_to_bottom_fric(fric_img, mask):
    img = cv2.imread(fric_img).astype('float') / 255
    bottom_fric_profile = np.stack([img[:,:,2][mask]/100, img[:,:,1][mask]], axis=-1)
    return bottom_fric_profile


class SimSingle():
    def __init__(self, particle_pos0, particle_radius, hand_radius, mass_gt=None, 
                bottom_fric_gt=None, mass_est=None, bottom_fric_est=None, mask=None,
                DT = Defaults.DT, DEVICE = Defaults.DEVICE):
        
        self.particle_pos0 = particle_pos0
        self.particle_radius = particle_radius
        self.N = self.particle_pos0.shape[0]
        self.DEVICE = DEVICE

        if mass_gt is None:
            self.mass_gt = 0.01 * torch.ones(self.N).to(DEVICE)
        else:
            self.mass_gt = get_tensor(mass_gt)

        if bottom_fric_gt is None:
            self.bottom_fric_gt = torch.FloatTensor([0.001, 0.1]).repeat(self.N, 1).to(DEVICE)
        else:
            self.bottom_fric_gt = bottom_fric_gt

        if mass_est is None:
            self.mass_est = 0.05 * torch.ones(self.N).to(DEVICE)
        else:
            self.mass_est = get_tensor(mass_est)
        self.mass_est = Variable(self.mass_est, requires_grad=True)

        if bottom_fric_est is None:
            self.bottom_fric_est = torch.FloatTensor([0.001, 0.1]).repeat(self.N, 1).to(DEVICE)
        else:
            self.bottom_fric_est = bottom_fric_est
        self.bottom_fric_est = Variable(self.bottom_fric_est, requires_grad=True)

        self.hand_radius = hand_radius
        self.mask = mask

    @classmethod
    def from_img(cls, mass_img_path, bottom_fric_img_path, particle_radius=None, 
                    hand_radius=None):
        particle_pos0, mask = image_to_pos(mass_img_path, particle_radius)
    
        mass_gt = image_to_mass(mass_img_path, mask)
        bottom_fric_gt = image_to_bottom_fric(bottom_fric_img_path, mask)
        return cls(particle_pos0, particle_radius, hand_radius, mass_gt=mass_gt, 
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

    def init_composite_object(self, particle_pos, particle_radius, mass_profile, 
                    bottom_fric_profile, rotation=0, offset=[0,0]):
        rotation_matrix = np.array([[cos(rotation), -sin(rotation)],
                                    [sin(rotation), cos(rotation)]])
        particle_pos = particle_pos @ rotation_matrix.T + np.array(offset)

        composite_body = CompositeSquare(particle_pos, particle_radius, mass=mass_profile, 
                                    fric_coeff_b=bottom_fric_profile)
        return composite_body

    def make_world(self, composite_body, action):
        bodies = []
        joints = []
        bodies += composite_body.bodies
        joints += composite_body.joints
        
        # init hand object
        c1 = Circle(action[0], self.hand_radius, fric_coeff_b=[0.005, 0.45])
        bodies.append(c1)

        # init force and apply force
        f = 2 * action[1]
        initial_force = torch.FloatTensor([0, f[0], f[1]]).to(self.DEVICE)
        push_force = lambda t: initial_force if t < 0.3 else ExternalForce.ZEROS
        c1.add_force(ExternalForce(push_force))
        
        # init world
        world = World(bodies, joints, dt=Defaults.DT, extend=1, solver_type=1)

        return world

    def run_episode_random(self, time=10, screen=None, recorder=None):
        rotation, offset = self.random_rest_composite_pose()
        # init composite object with offset and rotation
        composite_body_gt = self.init_composite_object(
                                    self.particle_pos0,
                                    self.particle_radius, 
                                    self.mass_gt,
                                    self.bottom_fric_gt,
                                    rotation=rotation,
                                    offset=offset)
        action = self.sample_action(composite_body_gt)
        world = self.make_world(composite_body_gt, action)
        recorder = None
        # recorder = Recorder(DT, screen)
        run_world(world, run_time=time, screen=screen, recorder=recorder)

        X1 = composite_body_gt.get_particle_pos()
        
        composite_body = self.init_composite_object(
                                    self.particle_pos0,
                                    self.particle_radius, 
                                    self.mass_est,
                                    self.bottom_fric_gt,
                                    rotation=rotation,
                                    offset=offset)
        world = self.make_world(composite_body, action)
        run_world(world, run_time=time, screen=screen, recorder=recorder)

        X2 = composite_body.get_particle_pos()

        return rotation, offset, action, X1, X2