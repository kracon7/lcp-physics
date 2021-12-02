import os
import sys

import time
import math
from math import sin, cos
import cv2
import random
import pygame
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from torch.autograd import Variable

from lcp_physics.physics.bodies import Circle, Composite
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


def main(screen):

    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

	# ================    SYSTEM IDENTIFICATION   ===================

	# select object name
	object_names = ['hammer', 'driller', 'rod1', 'rod2']
	obj_name = object_names[1]
    mass_img_path = os.path.join(ROOT, 'fig/%s_mass.png'%obj_name)
    bottom_fric_img_path = os.path.join(ROOT, 'fig/%s_fric.png'%obj_name)

    # initializa sim 
    sim = SimSingle.from_img(mass_img_path, bottom_fric_img_path, particle_radius=10, 
                    hand_radius=20)
    
    learning_rate = 1e-4
    max_iter = 50

    dist_hist = []
    mass_err_hist = []

    for i in range(max_iter):
    	# run ground truth and prediction        
        rotation, offset, action, X1, X2 = sim.run_episode_random(time=TIME, screen=screen)

        plot_mass_error(sim.mask, sim.mass_gt, 
                        sim.mass_est.detach().numpy(), 'tmp/mass_err_%03d.png'%i)
        
        dist = torch.sum(torch.norm(X1 - X2, dim=1))
        dist.backward()
        grad = sim.mass_est.grad.data
        print(grad)
        grad.clamp_(1/learning_rate * -2e-3, 1/learning_rate * 2e-3)

        sim.mass_est = torch.clamp(sim.mass_est.data - learning_rate * grad, min=1e-5)
        sim.mass_est.requires_grad=True
        learning_rate *= 0.99
        print(i, '/', max_iter, dist.data.item())
        
        print('=======\n\n')
        last_dist = dist
        dist_hist.append(dist / sim.N)

        reset_screen(screen)

    plot(dist_hist)

	# ================         PATH PLANNING      ===================

	# generate target pose

	# compute path towards the target pose

	# random sampling actions and do forward simulation

	# select the best action

	# action execution 



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