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

from lcp_physics.physics.bodies import Circle, Rect, Hull, Composite
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


if __name__ == '__main__':


    pygame.init()
    width, height = 1000, 600
    screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF)
    screen.set_alpha(None)
    pygame.display.set_caption('2D Engine')
    reset_screen(screen)
    save_path = os.path.join(ROOT, 'tmp/test_sim')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    # select object name
    object_names = ['hammer', 'drill', 'rod1', 'rod2', 'circle']
    obj_name = object_names[4]
    mass_img_path = os.path.join(ROOT, 'fig/%s_mass.png'%obj_name)
    bottom_fric_img_path = os.path.join(ROOT, 'fig/%s_fric.png'%obj_name)

    # initializa sim 
    sim = SimSingle.from_img(mass_img_path, bottom_fric_img_path, particle_radius=10, 
                    hand_radius=20)

    for i in range(10):
        reset_screen(screen)
        rotation, offset = sim.random_rest_composite_pose()
        composite_body = sim.init_composite_object(
                                        sim.particle_pos0,
                                        sim.particle_radius, 
                                        sim.mass_gt,
                                        sim.bottom_fric_gt,
                                        rotation=rotation,
                                        offset=offset)
        action = sim.sample_action(composite_body)

        update_list = []
        for body in composite_body.bodies:
            update_list += body.draw(screen, pixels_per_meter=1)

        c1 = Circle(action[0], sim.hand_radius, fric_coeff_b=[0.005, 0.45])
        update_list += c1.draw(screen, pixels_per_meter=1)
        pygame.display.update(update_list)
        
        pygame.image.save(screen, os.path.join(save_path, '%07d.jpg'%(i)))