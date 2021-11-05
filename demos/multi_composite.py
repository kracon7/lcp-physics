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


TIME = 25
DT = Defaults.DT
DEVICE = Defaults.DEVICE

def fixed_joint_demo(screen, particle_radius):
    bodies = []
    joints = []
    fric_coeff_s = 0.15

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img = cv2.imread(os.path.join(ROOT, 'fig/T-Shape.png'))
    img = cv2.resize(img, (20, 15))
    x_cord, y_cord = np.where(img[:,:,0]<1)
    x_cord, y_cord = x_cord - x_cord.min(), y_cord - y_cord.min()
    particle_pos = 2 * radius * np.stack([x_cord, y_cord]).T + np.array([300, 300])
    composite_body = Composite(particle_pos, particle_radius)
    bodies += composite_body.bodies
    joints += composite_body.joints

    img = cv2.imread(os.path.join(ROOT, 'fig/L-Shape.png'))
    img = cv2.resize(img, (10, 10))
    x_cord, y_cord = np.where(img[:,:,0]>=1)
    x_cord, y_cord = x_cord - x_cord.min(), y_cord - y_cord.min()
    particle_pos = 2 * radius * np.stack([x_cord, y_cord]).T + np.array([550, 400])
    composite_body = Composite(particle_pos, particle_radius)
    bodies += composite_body.bodies
    joints += composite_body.joints


    c = Circle([200, 400], 60)
    bodies.append(c)
    initial_force = torch.FloatTensor([0, 0.03, 0]).to(DEVICE)
    initial_force = Variable(initial_force, requires_grad=True)
    learned_force = lambda t: initial_force if t < 1 else ExternalForce.ZEROS
    c.add_force(ExternalForce(learned_force))

    world = World(bodies, joints, dt=DT, solver_type=3)

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

    radius = 10

    fixed_joint_demo(screen, radius)
