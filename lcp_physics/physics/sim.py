import os
import sys

import time
import math
from math import sin, cos
import cv2
import pygame
import numpy as np
import scipy.spatial as spatial
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from torch.autograd import Variable

from .bodies import Circle, Composite
from .constraints import TotalConstraint, FixedJoint
from .forces import ExternalForce, Gravity, vert_impulse, hor_impulse
from .utils import Defaults, plot, Recorder, rgb2mass, mass2rgb, get_tensor, rel_pose
from .world import World, run_world
from .action import build_exterior_mesh, random_action



def image_to_pos(mass_img, particle_radius):
    img = cv2.imread(mass_img)
    mask = img[:,:,0] < 255
    x_cord, y_cord = np.where(mask)
    x_cord, y_cord = x_cord - x_cord[0], y_cord - y_cord[0]
    particle_pos = 2 * particle_radius * np.stack([x_cord, y_cord]).T
    particle_pos = torch.from_numpy(particle_pos).float()
    mask = torch.from_numpy(mask)
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
        
        self.particle_pos0 = get_tensor(particle_pos0)
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
            self.bottom_fric_gt = get_tensor(bottom_fric_gt)

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
        self.action_mag = 20
        self.force_time = 0.1

        # build cononical exterior mesh
        polygon, polygon_coord, normals = build_exterior_mesh(particle_pos0, particle_radius)
        self.polygon = polygon
        self.polygon_coord = polygon_coord
        self.normals = normals


    @classmethod
    def from_img(cls, mass_img_path, bottom_fric_img_path, particle_radius=None, 
                    hand_radius=None):
        particle_pos0, mask = image_to_pos(mass_img_path, particle_radius)
    
        mass_gt = image_to_mass(mass_img_path, mask)
        bottom_fric_gt = image_to_bottom_fric(bottom_fric_img_path, mask)
        return cls(particle_pos0, particle_radius, hand_radius, mass_gt=mass_gt, 
                    bottom_fric_gt=bottom_fric_gt, mask=mask)    

    def random_rest_composite_pose(self, batch_size=1):
        # randomly initialize rotation and offset
        rotation = torch.rand(batch_size) * 2* math.pi
        offset = torch.cat([torch.FloatTensor(batch_size, 1).uniform_(450, 550),
                            torch.FloatTensor(batch_size, 1).uniform_(250, 350)], dim=-1)
        # offset = np.random.uniform(low=[450, 250], high=[550, 350], size=(batch_size, 2))
        return rotation, offset

    def sample_action(self, composite_body):
        # get composite object particle pos
        p = composite_body.get_particle_pos()
        curr_pose = rel_pose(self.particle_pos0, p).detach().cpu().numpy()
        theta, offset = curr_pose[0], curr_pose[1:]
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                    [np.sin(theta),  np.cos(theta)]])

        # randomly select the vertex and normal
        N = self.polygon.shape[0]
        idx = np.random.choice(N)
        vtx, nml = self.polygon_coord[idx], self.normals[idx]
        # print(nml)

        # # add 15 degree randomness to the normal direction
        # theta = np.random.uniform(-1/12*np.pi, 1/12*np.pi)
        # nml = np.array([[np.cos(theta), -np.sin(theta)],
        #                 [np.sin(theta),  np.cos(theta)]]) @ nml
        # print(nml)
        
        start_pos = vtx + self.hand_radius * nml

        while self.overlap_check(self.particle_pos0, self.particle_radius, 
                                start_pos, self.hand_radius):
            start_pos += 0.5 * nml

        start_pos = rotation_matrix @ start_pos + offset
        nml = rotation_matrix @ nml
        action = [start_pos, -nml]

        return action


    def overlap_check(self, pts1, r1, pts2, r2):
        '''
        check if circles with center pts1 and radius r1 has overelap with circles 
        with center pts2 and radius r2
        '''
        point_tree = spatial.cKDTree(pts1)
        neighbors_list = point_tree.query_ball_point(pts2, r1 + r2)
        if len(neighbors_list) > 0:
            return True
        else:
            return False

    def init_composite_object(self, particle_radius, mass_profile, bottom_fric_profile, 
                            rotation=torch.tensor(0.), offset=torch.tensor([0.,0.])):
        particle_pos = self.transform_particles(rotation, offset)

        composite_body = Composite(particle_pos, particle_radius, mass=mass_profile, 
                                    fric_coeff_b=bottom_fric_profile)
        return composite_body

    def transform_particles(self, rotation=0, offset=[0,0]):
        '''
        Apply rigid body transformation to the composite object and return particle pos
        '''
        rotation_matrix = torch.stack([torch.stack([torch.cos(rotation), -torch.sin(rotation)]), 
                                       torch.stack([torch.sin(rotation),  torch.cos(rotation)])])
        particle_pos = self.particle_pos0 @ rotation_matrix.t() + offset
        return particle_pos

    def make_world(self, composite_body, action):
        bodies = []
        joints = []
        bodies += composite_body.bodies
        joints += composite_body.joints

        composite_body.body_id = np.arange(len(composite_body.bodies))
        
        # init hand object
        c1 = Circle(action[0], self.hand_radius, mass=10, fric_coeff_b=[0.002, 0.05])
        bodies.append(c1)

        # init force and apply force
        f = self.action_mag * action[1]
        initial_force = torch.FloatTensor([0, f[0], f[1]]).to(self.DEVICE)
        c1.add_force(ExternalForce(initial_force, self.force_time))
        
        # init world
        world = World(bodies, joints, dt=Defaults.DT, extend=1, solver_type=1)

        return world

    def run_episode_random(self, time=10, screen=None, recorder=None):
        rotation, offset = self.random_rest_composite_pose()
        # init composite object with offset and rotation
        composite_body_gt = self.init_composite_object(self.particle_radius, 
                                                       self.mass_gt,
                                                       self.bottom_fric_gt,
                                                       rotation=rotation[0],
                                                       offset=offset[0])
        action = self.sample_action(composite_body_gt)
        world = self.make_world(composite_body_gt, action)
        recorder = None
        # recorder = Recorder(DT, screen)
        run_world(world, run_time=time, screen=screen, recorder=recorder)

        X1 = composite_body_gt.get_particle_pos()
        
        composite_body = self.init_composite_object(self.particle_radius, 
                                                    self.mass_est,
                                                    self.bottom_fric_gt,
                                                    rotation=rotation[0],
                                                    offset=offset[0])
        world = self.make_world(composite_body, action)
        run_world(world, run_time=time, screen=screen, recorder=recorder)

        X2 = composite_body.get_particle_pos()

        return rotation, offset, action, X1, X2