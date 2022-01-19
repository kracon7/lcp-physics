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

from .bodies import Circle, Rect, Hull, Composite, CompositeSquare
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
    particle_pos = torch.from_numpy(particle_pos)
    mask = torch.from_numpy(mask)
    return particle_pos.type(Defaults.DTYPE), mask

def image_to_mass(mass_img, mask):
    img = cv2.imread(mass_img)
    mass_profile = rgb2mass(img)[mask]
    return mass_profile

def image_to_bottom_fric(fric_img, mask):
    img = cv2.imread(fric_img).astype('float') / 255
    bottom_fric_profile = np.stack([img[:,:,2][mask]/2, img[:,:,1][mask]], axis=-1)
    return bottom_fric_profile


class SimSingle():
    def __init__(self, particle_pos0, particle_radius, hand_radius, mass_gt=None, 
                bottom_fric_gt=None, mass_est=None, bottom_fric_est=None, obj_mask=None,
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
            self.bottom_fric_est = get_tensor(bottom_fric_est)
        self.bottom_fric_est = Variable(self.bottom_fric_est, requires_grad=True)

        self.hand_radius = hand_radius
        self.obj_mask = obj_mask
        self.action_mag = 2
        self.force_time = 0.3

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
                    bottom_fric_gt=bottom_fric_gt, obj_mask=mask)    

    def random_rest_composite_pose(self, batch_size=1):
        # randomly initialize rotation and offset
        rotation = torch.rand(batch_size) * 2* math.pi
        offset = torch.cat([torch.FloatTensor(batch_size, 1).uniform_(450, 550),
                            torch.FloatTensor(batch_size, 1).uniform_(250, 350)], dim=-1)
        # offset = np.random.uniform(low=[450, 250], high=[550, 350], size=(batch_size, 2))
        return rotation.type(Defaults.DTYPE), offset.type(Defaults.DTYPE)

    def sample_action(self, composite_body, batch_size=1):
        # get composite object particle pos
        p = composite_body.get_particle_pos()
        if p.shape[0] == 1:
            curr_pose = composite_body.bodies[0].p.detach().cpu().numpy()
        else:
            curr_pose = rel_pose(self.particle_pos0, p).detach().cpu().numpy()
        theta, offset = curr_pose[0], curr_pose[1:]
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                    [np.sin(theta),  np.cos(theta)]])

        # randomly select the vertex and normal
        N = self.polygon.shape[0]
        idx = np.random.choice(N, size=batch_size, replace=False)
        action = []
        for i in idx:
            vtx, nml = self.polygon_coord[i], self.normals[i]
            # print(nml)

            # # add 15 degree randomness to the normal direction
            # theta = np.random.uniform(-1/12*np.pi, 1/12*np.pi)
            # nml = np.array([[np.cos(theta), -np.sin(theta)],
            #                 [np.sin(theta),  np.cos(theta)]]) @ nml
            # print(nml)
        
            start_pos = vtx + self.hand_radius * nml

            while self.overlap_check(self.polygon_coord, 0.1*self.particle_radius, 
                                     start_pos, self.hand_radius):
                start_pos += 0.5 * nml

            start_pos = rotation_matrix @ start_pos + offset
            nml = rotation_matrix @ nml
            action.append([start_pos, -nml])

        # making it compatible with earlier testing scripts
        if batch_size == 1:
            action = action[0]

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

    def init_composite_object(self, particle_pos, particle_radius, mass_profile, 
                bottom_fric_profile, rotation=torch.tensor([0.]).double(), offset=torch.tensor([0.,0.]).double()):
        particle_pos = self.transform_particles(rotation, offset)

        composite_body = CompositeSquare(particle_pos, particle_radius, mass=mass_profile, 
                                    fric_coeff_b=bottom_fric_profile)
        return composite_body

    def transform_particles(self, rotation=torch.tensor([0.]).double(), 
                                offset=torch.tensor([0.,0.]).double()):
        '''
        Apply rigid body transformation to the composite object and return particle pose
        including rotation
        Output
            pose -- tensor (N, 3)
        '''
        rotation_matrix = torch.stack([torch.cat([torch.cos(rotation), -torch.sin(rotation)], 0), 
                                       torch.cat([torch.sin(rotation),  torch.cos(rotation)], 0)])
        particle_pos = self.particle_pos0 @ rotation_matrix.t() + offset

        # add rotation
        particle_pose = torch.cat([rotation.repeat(self.N, 1), particle_pos], dim=1)

        return particle_pose

    def make_world(self, composite_body, action, extend=1, solver_type=1, verbose=0, strict_no_pen=True):
        bodies = []
        joints = []
        bodies += composite_body.bodies
        joints += composite_body.joints
        
        # init hand object
        c1 = Circle(action[0], self.hand_radius, mass=10, fric_coeff_b=[0.005, 0.45])
        bodies.append(c1)

        # init force and apply force
        f = self.action_mag * action[1]
        initial_force = torch.FloatTensor([0, f[0], f[1]]).to(self.DEVICE)
        push_force = lambda t: initial_force if t < self.force_time else ExternalForce.ZEROS
        c1.add_force(ExternalForce(push_force))
        
        # init world
        world = World(bodies, joints, dt=Defaults.DT, verbose=verbose, extend=extend, 
                solver_type=solver_type, strict_no_penetration=strict_no_pen)

        return world

    def run_episode_random(self, t=10, verbose=-1, screen=None, recorder=None):
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
        world = self.make_world(composite_body_gt, action, verbose)
        recorder = None
        # recorder = Recorder(DT, screen)
        run_world(world, run_time=t, screen=screen, recorder=recorder)

        X1 = composite_body_gt.get_particle_pos()
        
        composite_body = self.init_composite_object(
                                    self.particle_pos0,
                                    self.particle_radius, 
                                    self.mass_est,
                                    self.bottom_fric_gt,
                                    rotation=rotation,
                                    offset=offset)
        world = self.make_world(composite_body, action, verbose)
        run_world(world, run_time=t, screen=screen, recorder=recorder)

        X2 = composite_body.get_particle_pos()

        return rotation, offset, action, X1, X2

    def positions_run_world(self, world, dt=Defaults.DT, run_time=10,
                        screen=None, recorder=None):
        '''
        run world while recording particle positions at every step
        '''
        positions = [torch.cat([b.p for b in world.bodies])]

        if screen is not None:
            background = pygame.Surface(screen.get_size())
            background = background.convert()
            background.fill((255, 255, 255))

            animation_dt = dt
            elapsed_time = 0.
            prev_frame_time = -animation_dt
            start_time = time.time()

        while world.t < run_time:
            world.step()
            positions.append(torch.cat([b.p for b in world.bodies]))

            if screen is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return

                if elapsed_time - prev_frame_time >= animation_dt or recorder:
                    prev_frame_time = elapsed_time

                    screen.blit(background, (0, 0))
                    update_list = []
                    for body in world.bodies:
                        update_list += body.draw(screen)
                    for joint in world.joints:
                        update_list += joint[0].draw(screen)

                    if not recorder:
                        # Don't refresh screen if recording
                        pygame.display.update(update_list)
                    else:
                        recorder.record(world.t)

                elapsed_time = time.time() - start_time
    
        return positions

