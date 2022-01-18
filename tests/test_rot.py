'''
Rod with 3 unknown mass. Define rotation center at make-world()
Rod composite is loaded from fig/rot_center_rod.png, it's a 12 x 1 grid
Regress mass with lcp-physics
'''

import os
import sys

import time
import cv2
import pygame
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from torch.nn import MSELoss
from torch.autograd import Variable

from lcp_physics.physics.bodies import Circle, Rect, Hull, Composite
from lcp_physics.physics.constraints import TotalConstraint, FixedJoint, Joint
from lcp_physics.physics.forces import ExternalForce, Gravity, vert_impulse, hor_impulse
from lcp_physics.physics.utils import Defaults, plot, reset_screen, Recorder
from lcp_physics.physics.world import World, run_world
from lcp_physics.physics.action import build_mesh, random_action
from lcp_physics.physics.sim import SimSingle


TIME = 5
STOP_DIFF = 1e-3
DT = Defaults.DT
DEVICE = Defaults.DEVICE
DTYPE = Defaults.DTYPE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM_P = 1
P_RADIUS = 10

np.random.seed(10)
torch.random.manual_seed(0)


def make_world(mass, rot_center):
    particle_pos = torch.stack([torch.zeros(NUM_P), 
                                torch.linspace(0, 2*P_RADIUS*(NUM_P-1), NUM_P)]
                              ).t().type(DTYPE) + torch.tensor([500, 300]).type(DTYPE)
    
    bodies = []
    joints = []
    N = particle_pos.shape[0]

    center = Circle(rot_center, 1, mass=0.1)
    bodies.append(center)
    joints.append(Joint(center, None, rot_center))

    for i in range(N):
        c = Rect(particle_pos[i], [2*P_RADIUS, 2*P_RADIUS], mass=0.1, fric_coeff_s=0, 
                    fric_coeff_b=[0., 0.0])
        bodies.append(c)
        joints += [FixedJoint(c, center)]

    for i in range(len(bodies)):
        for j in range(len(bodies)):
            bodies[i].add_no_contact(bodies[j])

    # init force and apply force
    initial_force = torch.FloatTensor([50, 0, 0]).type(DTYPE).to(DEVICE)
    push_force = lambda t: initial_force if t < 0.5 else ExternalForce.ZEROS
    center.add_force(ExternalForce(push_force))

    world = World(bodies, joints, extend=1, solver_type=1, strict_no_penetration=False)
    return world


def positions_run_world(world, dt=Defaults.DT, run_time=10,
                        screen=None, recorder=None):
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

    return positions


def main(screen):

    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

    mass_gt = torch.tensor([0.5, 1, 2]).type(DTYPE)
    rot_center = torch.tensor([0, -30]).type(DTYPE) + torch.tensor([500, 300]).type(DTYPE)
    
    world = make_world(mass_gt, rot_center)
    recorder = None
    recorder = Recorder(DT, screen)
    ground_truth_pos = positions_run_world(world, run_time=TIME, screen=screen, recorder=recorder)
    ground_truth_pos = torch.stack(ground_truth_pos)

    a=1


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

    main(screen)