'''
System identification with post-stabilization enabled
Loss: MSELoss constructed over ALL points in a SINGLE trajectory
Optimizer: RMSProp from PyTorch
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
from torch.nn import MSELoss
from torch.autograd import Variable

from lcp_physics.physics.bodies import Circle, Rect, Hull, Composite
from lcp_physics.physics.constraints import TotalConstraint, FixedJoint
from lcp_physics.physics.forces import ExternalForce, Gravity, vert_impulse, hor_impulse
from lcp_physics.physics.utils import Defaults, plot, reset_screen, Recorder
from lcp_physics.physics.world import World, run_world
from lcp_physics.physics.action import build_mesh, random_action
from lcp_physics.physics.sim import SimSingle


TIME = 4
STOP_DIFF = 1e-3
DT = Defaults.DT
DEVICE = Defaults.DEVICE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

np.random.seed(10)
torch.random.manual_seed(0)

def sim_demo(screen):

    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

    obj_name = 'hammer'
    mass_img_path = os.path.join(ROOT, 'fig/%s_mass.png'%obj_name)
    bottom_fric_img_path = os.path.join(ROOT, 'fig/%s_fric.png'%obj_name)
    default_actions = {'rod1':  {'action_mag': 15, 'force_time': 0.4},
                       'drill': {'action_mag': 20, 'force_time': 0.4},
                       'hammer': {'action_mag': 15, 'force_time': 0.4}}

    sim = SimSingle.from_img(mass_img_path, bottom_fric_img_path, particle_radius=10, 
                    hand_radius=10)
    sim.action_mag = default_actions[obj_name]['action_mag']
    sim.force_time = default_actions[obj_name]['force_time']
    
    # run ground truth to get ground truth trajectory
    rotation, offset = torch.tensor([0]).type(Defaults.DTYPE), torch.tensor([[500, 300]]).type(Defaults.DTYPE)
    
    mass = torch.tensor([0.1]).double()
    mass_mapping = [0 for _ in range(sim.N)]
    composite_body = sim.init_composite_object(mass, mass_mapping,
                                rotation=rotation, offset=offset)
    action = sim.sample_action(composite_body)
    world = sim.make_world(composite_body, action, verbose=-1)
    recorder = None
    # recorder = Recorder(DT, screen)
    estimated_pos = sim.positions_run_world(world, run_time=TIME, screen=screen, recorder=recorder)
    

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