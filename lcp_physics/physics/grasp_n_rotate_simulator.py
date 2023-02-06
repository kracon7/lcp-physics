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
from collections import defaultdict

from lcp_physics.physics.bodies import Circle, Rect, Hull
from lcp_physics.physics.constraints import TotalConstraint, FixedJoint, Joint
from lcp_physics.physics.forces import ExternalForce, Gravity, vert_impulse, hor_impulse
from lcp_physics.physics.utils import Defaults, plot, reset_screen, Recorder, left_orthogonal, right_orthogonal
from lcp_physics.physics.world import World, run_world
from lcp_physics.physics.block_object_util import BlockObject

class GraspNRotateSimulator:
    def __init__(self, param_file):
        self.block_object = BlockObject(param_file)
        self.particle_pos0 = torch.from_numpy(self.block_object.particle_coord).double()
        self.num_particle = self.block_object.num_particle
        self.particle_radius = self.block_object.voxel_size / 2

    def initialize(self, mass, mass_mapping, friction, friction_mapping, u_i, forces, init_vel):
        particle_pose = self.particle_pos0.clone()

        bodies = []
        joints = []
        N = particle_pose.shape[0]
        side = 2 * self.particle_radius

        for i in range(N):
            c = Rect(particle_pose[i], [side, side], mass=mass[mass_mapping[i]])
            c.v = init_vel[i]
            if i == u_i:
                c.forces.append(forces)
            bodies.append(c)

        for i in range(N-1):
            joints += [FixedJoint(bodies[i], bodies[-1])]

        # add contact exclusion
        no_contact = self.find_neighbors(particle_pose[:,:2], self.particle_radius)
        for i in range(N):
            neighbors = no_contact[i]
            for j in neighbors:
                bodies[i].add_no_contact(bodies[j])

        self.bodies = bodies
        self.joints = joints


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


    def find_neighbors(self, particle_pos, radius):
        '''
        find neighbors of particles for contact exlusion
        '''
        point_tree = spatial.cKDTree(particle_pos)
        neighbors_list = point_tree.query_ball_point(particle_pos, 4*radius)

        no_contact = defaultdict(list)
        for i in range(particle_pos.shape[0]):
            neighbors = neighbors_list[i]
            neighbors.remove(i)
            no_contact[i] = neighbors

        return no_contact

    def make_world(self, extend=1, solver_type=1, verbose=0, strict_no_pen=True):
        
        # init world
        world = World(self.bodies, self.joints, dt=Defaults.DT, verbose=verbose, extend=extend, 
                solver_type=solver_type, strict_no_penetration=strict_no_pen)

        return world


    def positions_run_world(self, world, dt=Defaults.DT, run_time=10,
                        screen=None, recorder=None):
        '''
        run world while recording particle positions at every step
        '''
        positions = {}
        t = round(world.t, 4)
        positions[t] = torch.stack([b.p for b in world.bodies])

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
            t = round(world.t, 4)
            positions[t] = torch.stack([b.p for b in world.bodies])

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