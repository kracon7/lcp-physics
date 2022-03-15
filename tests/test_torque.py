'''
Test fixed point rotation with an initial velocity and applied torque
'''

import os
import sys
from collections import defaultdict
import time
import math
from math import sin, cos
import cv2
import pygame
import scipy.spatial as spatial
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from torch.nn import MSELoss
from torch.autograd import Variable

from lcp_physics.physics.bodies import Circle, Rect, Hull
from lcp_physics.physics.constraints import TotalConstraint, FixedJoint, Joint
from lcp_physics.physics.forces import ExternalForce, Gravity, vert_impulse, hor_impulse
from lcp_physics.physics.utils import Defaults, plot, reset_screen, Recorder, left_orthogonal, right_orthogonal
from lcp_physics.physics.world import World, run_world
from lcp_physics.physics.action import build_mesh, random_action
from lcp_physics.physics.sim import SimSingle


TIME = 10
STOP_DIFF = 1e-3
DT = Defaults.DT
DEVICE = Defaults.DEVICE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

np.random.seed(10)
torch.random.manual_seed(0)

class CompositeSquare():
    def __init__(self, mass_img, particle_radius=10):
        self.mass_img = mass_img
        self.particle_radius = particle_radius

        self.particle_pos0, self.mask = self.image_to_pos(mass_img, particle_radius)
        self.num_particle = self.particle_pos0.shape[0]

    def initialize(self, rotation, translation, mass, mass_mapping, init_vel):
        '''
        Input:
            rotation -- theta angle, 1D tensor
            translation -- x and y translation, 1D tensor
            mass -- float or ndarray (N,), mass of each particle
            fric_coeff_s -- side friction coefficient
            fric_coeff_b -- bottom friction coefficient
        '''
        particle_pose = self.transform(rotation, translation)

        bodies = []
        joints = []
        N = particle_pose.shape[0]
        side = 2 * self.particle_radius

        for i in range(N):
            c = Rect(particle_pose[i], [side, side], mass=mass[mass_mapping[i]])
            c.v = init_vel[i]
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

    def get_particle_pos(self):
        pos = []
        for b in self.bodies:
            pos.append(b.pos)
        pos = torch.stack(pos)
        return pos

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

    def make_world(self, extend=0, solver_type=1, verbose=0, strict_no_pen=True):
        
        # init world
        world = World(self.bodies, self.joints, dt=Defaults.DT, verbose=verbose, extend=extend, 
                solver_type=solver_type, strict_no_penetration=strict_no_pen)

        return world

    def image_to_pos(self, mass_img, particle_radius):
        img = cv2.imread(mass_img)
        mask = img[:,:,0] < 255
        x_cord, y_cord = np.where(mask)
        x_cord, y_cord = x_cord - x_cord[0], y_cord - y_cord[0]
        particle_pos = 2 * particle_radius * np.stack([x_cord, y_cord]).T
        particle_pos = torch.from_numpy(particle_pos)
        mask = torch.from_numpy(mask)
        return particle_pos.type(Defaults.DTYPE), mask

    def transform(self, rotation=torch.tensor([0.]).double(), 
                        translation=torch.tensor([0.,0.]).double()):
        '''
        Apply rigid body transformation to the composite object and return particle pose
        including rotation
        Output
            pose -- tensor (N, 3)
        '''
        rotation_matrix = torch.stack([torch.cat([torch.cos(rotation), -torch.sin(rotation)], 0), 
                                       torch.cat([torch.sin(rotation),  torch.cos(rotation)], 0)])
        particle_pos = self.particle_pos0 @ rotation_matrix.t() + translation

        # add rotation
        particle_pose = torch.cat([rotation.repeat(self.num_particle, 1), particle_pos], dim=1)

        return particle_pose

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

    def apply_torque(self, body_idx, magnitude):
        """
        Apply torque to a particle.
        Args
            body_idx -- body index to apply torque
            magnitude -- magnitude of the torque
        """
        self.joints.append(Joint(self.bodies[body_idx], None, self.bodies[body_idx].pos.clone()))

        initial_force = torch.tensor([magnitude, 0, 0]).double().to(DEVICE)
        push_force = lambda t: initial_force if t < TIME else ExternalForce.ZEROS
        self.bodies[body_idx].add_force(ExternalForce(push_force))


def torque_run_world(world, dt=Defaults.DT, run_time=10, screen=None, recorder=None):
    '''
    run world while recording particle positions at every step
    '''
    constraints = []

    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

        animation_dt = dt
        elapsed_time = 0.
        prev_frame_time = -animation_dt
        start_time = time.time()

    while world.t < run_time:
        world.step()
        constraints.append(world.constraints[-2:])

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

    return constraints


def sim_demo(screen):

    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

    obj_name = 'L'
    mass_img_path = os.path.join(ROOT, 'fig/%s_mass.png'%obj_name)

    composite = CompositeSquare(mass_img_path, particle_radius=10)
    N = composite.num_particle
    rotation, translation = torch.tensor([0]).type(Defaults.DTYPE), torch.tensor([[500, 300]]).type(Defaults.DTYPE)
    
    mass = torch.tensor([0.2]).double()
    mass_mapping = [0 for _ in range(N)]
    
    # initial velocity of composite body
    init_vel = composite.get_init_vel(torch.tensor([0,0]).double(), torch.tensor([0]).double())

    composite.initialize(rotation, translation, mass, mass_mapping, init_vel)
    composite.apply_torque(0, 5)
    world = composite.make_world()
    recorder = None
    recorder = Recorder(DT, screen)
    constraints = torque_run_world(world, run_time=TIME, screen=screen, recorder=recorder)

    constraints = torch.stack(constraints).detach().numpy()
    timesteps = constraints.shape[0]

    fig, ax = plt.subplots(1,1)
    for i in range(timesteps):
        fx, = ax.plot(constraints[:i, 0], label='fx')
        fy, = ax.plot(constraints[:i, 1], label='fy')
        ax.set_xlim(0, timesteps)
        ax.set_ylim(constraints.min()-0.2, constraints.max()+0.2)
        ax.set_xlabel('time')
        ax.set_ylabel('force')
        ax.legend(handles=[fx, fy])
        fig.savefig(os.path.join(ROOT, 'tmp/temp%07d.png'%i), bbox_inches='tight')
        plt.cla()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '-nd':
        # Run without displaying
        screen = None
    else:
        pygame.init()
        width, height = 1000, 600
        screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF)
        screen.set_alpha(None)
        pygame.display.set_caption('2D Engine')
        reset_screen(screen)

    sim_demo(screen)