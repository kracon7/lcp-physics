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

def make_world(radius):
    '''
    build world based on particle positions
    '''
    bodies = []
    joints = []
    fric_coeff = 0.1

    pos = [500, 300]
    c1 = Circle(pos, radius, fric_coeff=fric_coeff)
    bodies.append(c1)

    c2 = Circle([pos[0]+2*radius, pos[1]+5], radius, fric_coeff=fric_coeff)
    bodies.append(c2)

    initial_force = torch.FloatTensor([0, 0.5, 0]).to(DEVICE)
    initial_force[2] = 0
    initial_force = Variable(initial_force, requires_grad=True)

    # Initial demo
    learned_force = lambda t: initial_force if t < 2 else ExternalForce.ZEROS
    c1.add_force(ExternalForce(learned_force))

    world = World(bodies, joints, dt=DT)
    return world
    

def fixed_joint_demo(screen):
    radius = 30
    world = make_world(radius)
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

    fixed_joint_demo(screen)
