import math
import sys

import pygame
import numpy as np
import torch
from torch.autograd import Variable

from lcp_physics.physics.bodies import Circle, Rect, Hull, Composite
from lcp_physics.physics.constraints import TotalConstraint, FixedJoint
from lcp_physics.physics.forces import ExternalForce, Gravity, vert_impulse, hor_impulse
from lcp_physics.physics.utils import Defaults, Recorder
from lcp_physics.physics.world import World, run_world


TIME = 10
DT = Defaults.DT

def make_world(particle_pos, hand):
    '''
    build world based on particle positions
    '''
    bodies = []
    joints = []
    fric_coeff = 0.15

    composite_body = Composite(particle_pos)
    bodies += composite_body.bodies
    joints += composite_body.joints

    c = Circle(hand, 60)
    bodies.append(c)

    initial_force = torch.DoubleTensor([0, 3, 0])
    initial_force[2] = 0
    initial_force = Variable(initial_force, requires_grad=True)

    # Initial demo
    learned_force = lambda t: initial_force if t < 0.2 else ExternalForce.ZEROS
    c.add_force(ExternalForce(learned_force))

    world = World(bodies, joints, dt=DT)
    return world
    

def fixed_joint_demo(screen):
    particle_pos = np.array([[500, 300],
                             [500, 340],
                             [500, 380],
                             [540, 300],
                             [580, 300],
                             [500, 420]])
    world = make_world(particle_pos, [200, 300])
    recorder = None
    # recorder = Recorder(DT, screen)
    run_world(world, run_time=TIME, screen=screen, recorder=recorder)


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

    fixed_joint_demo(screen)
