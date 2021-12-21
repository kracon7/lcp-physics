'''
Pushing single circle with uncertain mass
Initialize k hypothesis and maintain k sequence of estimations 
Keep track of their mean and variance
'''

import os
import sys

import time
import math
from math import sin, cos
import cv2
import pygame
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from torch.autograd import Variable

from lcp_physics.physics.constraints import TotalConstraint, FixedJoint
from lcp_physics.physics.forces import ExternalForce, Gravity, vert_impulse, hor_impulse
from lcp_physics.physics.utils import Defaults, plot, reset_screen, Recorder
from lcp_physics.physics.world import World, run_world
from lcp_physics.physics.action import build_mesh, random_action
from lcp_physics.physics.sim import SimSingle


TIME = 2
DT = Defaults.DT
DEVICE = Defaults.DEVICE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

np.random.seed(1)
torch.random.manual_seed(0)

def sys_id_demo(screen):

    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

    mass_img_path = os.path.join(ROOT, 'fig/circle_mass.png')
    bottom_fric_img_path = os.path.join(ROOT, 'fig/circle_fric.png')

    sim = SimSingle.from_img(mass_img_path, bottom_fric_img_path, particle_radius=10, 
                hand_radius=20)
    sim.mass_est = 0.25 * torch.rand(1) + 0.03
    sim.mass_est.requires_grad = True
    sim.bottom_fric_est = sim.bottom_fric_gt

    rotation, offset = torch.tensor([2.5241], dtype=torch.float64), \
                       torch.tensor([[452.2326, 266.8859]], dtype=torch.float64)
    composite_body = sim.init_composite_object(
                                sim.particle_pos0,
                                sim.particle_radius, 
                                sim.mass_gt,
                                sim.bottom_fric_gt,
                                rotation=rotation,
                                offset=offset)
    action = [np.array([426.59176035, 291.22821701]), np.array([ 0.81531208, -0.57902177])]
    world = sim.make_world(composite_body, action, 1)
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
        reset_screen(screen)

    sys_id_demo(screen)