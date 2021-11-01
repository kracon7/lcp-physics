import os
import sys

import math
import cv2
import pygame
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from lcp_physics.physics.bodies import Circle, Rect, Hull, Composite
from lcp_physics.physics.constraints import TotalConstraint, FixedJoint
from lcp_physics.physics.forces import ExternalForce, Gravity, vert_impulse, hor_impulse
from lcp_physics.physics.utils import Defaults, Recorder
from lcp_physics.physics.world import World, run_world


TIME = 10
DT = Defaults.DT
DEVICE = Defaults.DEVICE

def make_world(particle_pos, particle_radius, hand):
    '''
    build world based on particle positions
    '''
    bodies = []
    joints = []
    fric_coeff_s = 0.15

    composite_body = Composite(particle_pos, particle_radius)
    bodies += composite_body.bodies
    joints += composite_body.joints

    c = Circle(hand, 60)
    bodies.append(c)

    inclination = math.pi * 17/ 32
    r = Rect([inclination, 900, 450], [400, 10])
    bodies.append(r)
    joints.append(TotalConstraint(r))

    initial_force = torch.FloatTensor([0, 0.5, 0]).to(DEVICE)
    initial_force = Variable(initial_force, requires_grad=True)

    # Initial demo
    learned_force = lambda t: initial_force if t < 2 else ExternalForce.ZEROS
    c.add_force(ExternalForce(learned_force))

    world = World(bodies, joints, dt=DT, solver_type=2)
    return world
    

def fixed_joint_demo(screen, particle_pos, particle_radius):
    # particle_pos = np.array([[500, 300],
    #                          [500, 340],
    #                          [500, 380],
    #                          [540, 300],
    #                          [580, 300],
    #                          [500, 420]])
    world = make_world(particle_pos, particle_radius, [300, 350])
    recorder = None
    # recorder = Recorder(DT, screen)
    run_world(world, run_time=TIME, screen=screen, recorder=recorder)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '-nd':
        # Run without displaying
        screen = None
    else:
        pygame.init()
        width, height = 1600, 800
        screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF)
        screen.set_alpha(None)
        pygame.display.set_caption('2D Engine')

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img = cv2.imread(os.path.join(ROOT, 'fig/T-Shape.png'))
    img = cv2.resize(img, (40, 30))
    x_cord, y_cord = np.where(img[:,:,0]<1)
    x_cord, y_cord = x_cord - x_cord.min(), y_cord - y_cord.min()

    radius = 10

    particle_pos = 2 * radius * np.stack([x_cord, y_cord]).T + np.array([400, 300])

    fixed_joint_demo(screen, particle_pos, radius)
