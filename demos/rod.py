import os
import sys

import time
import math
import cv2
import pygame
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from lcp_physics.physics.bodies import Circle, Composite
from lcp_physics.physics.constraints import TotalConstraint, FixedJoint
from lcp_physics.physics.forces import ExternalForce, Gravity, vert_impulse, hor_impulse
from lcp_physics.physics.utils import Defaults, plot, reset_screen, Recorder
from lcp_physics.physics.world import World, run_world


TIME = 5
DT = Defaults.DT
DEVICE = Defaults.DEVICE

def make_world_2(rod_mass, rod_fric, action):
    bodies = []
    joints = []
    
    p = np.array([[400, 300],
                  [440, 300],
                  [480, 300],
                  [520, 300],
                  [560, 300],
                  [600, 300]])
    rod = Composite(p, 20, mass=rod_mass, fric_coeff_b=rod_fric)
    bodies += rod.bodies
    joints += rod.joints

    # init hand object
    c1 = Circle(action[0], 40, fric_coeff_b=[0.005, 0.45])
    bodies.append(c1)

    # init force and apply force
    f = 5 * action[1]
    initial_force = torch.FloatTensor([0, f[0], f[1]]).to(DEVICE)
    c1.add_force(ExternalForce(initial_force, 0.5))
    
    # init world
    world = World(bodies, joints, dt=Defaults.DT, extend=1, solver_type=1)

    return world
    

def rod_demo_1(screen, recorder):
    '''uniform mass, uniform friction, push up
    '''
    
    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))
    
    rod_mass = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    rod_fric = [0.0001, 0.3]
    action = [[500, 360], [0, -1]]
    world = make_world_2(rod_mass, rod_fric, action)
    run_world(world, run_time=TIME, screen=screen, recorder=recorder)

def rod_demo_2(screen, recorder):
    '''dumbbell mass, uniform friction, push up
    '''
    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))
    
    rod_mass = np.array([0.28, 0.01, 0.01, 0.01, 0.01, 0.28])
    rod_fric = [0.0001, 0.3]
    action = [[500, 360], [0, -1]]
    world = make_world_2(rod_mass, rod_fric, action)
    run_world(world, run_time=TIME, screen=screen, recorder=recorder)

def rod_demo_3(screen, recorder):
    '''uniform mass, non-uniform friction, push up
    '''
    
    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))
    
    rod_mass = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    rod_fric = np.array([[0.0001, 0.3],
                         [0.0001, 0],
                         [0.0001, 0],
                         [0.0001, 0.3],
                         [0.0001, 0.3],
                         [0.0001, 0.3]])
    action = [[500, 360], [0, -1]]
    world = make_world_2(rod_mass, rod_fric, action)
    run_world(world, run_time=TIME, screen=screen, recorder=recorder)

def rod_demo_4(screen, recorder):
    '''uniform mass, uniform friction, push slightly towards right
    '''
    
    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))
    
    rod_mass = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    rod_fric = np.array([[0.0001, 0.3],
                         [0.0001, 0],
                         [0.0001, 0],
                         [0.0001, 0.3],
                         [0.0001, 0.3],
                         [0.0001, 0.3]])
    action = [[500, 360], [0.31, -0.95]]
    world = make_world_2(rod_mass, rod_fric, action)
    run_world(world, run_time=TIME, screen=screen, recorder=recorder)

def rod_demo_5(screen, recorder):
    '''non-uniform mass, uniform friction, push slightly towards right
    '''
    
    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))
    
    rod_mass = np.array([0.28, 0.01, 0.01, 0.01, 0.01, 0.28])
    rod_fric = np.array([[0.0001, 0.3],
                         [0.0001, 0],
                         [0.0001, 0],
                         [0.0001, 0.3],
                         [0.0001, 0.3],
                         [0.0001, 0.3]])
    action = [[500, 360], [0.31, -0.95]]
    world = make_world_2(rod_mass, rod_fric, action)
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
        reset_screen(screen)

    recorder = None
    # recorder = Recorder(DT, screen)

    rod_demo_1(screen, recorder)
    # reset_screen(screen)
    # rod_demo_2(screen, recorder)
    # reset_screen(screen)
    # rod_demo_3(screen, recorder)
    # reset_screen(screen)
    # rod_demo_4(screen, recorder)
    # reset_screen(screen)
    # rod_demo_5(screen, recorder)
