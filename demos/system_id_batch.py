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

from lcp_physics.physics.bodies import Circle, Composite
from lcp_physics.physics.constraints import TotalConstraint, FixedJoint
from lcp_physics.physics.forces import ExternalForce, Gravity, vert_impulse, hor_impulse
from lcp_physics.physics.utils import Defaults, plot, reset_screen, Recorder
from lcp_physics.physics.world import World, run_world, run_world_batch, run_world_single
from lcp_physics.physics.action import build_mesh, random_action
from lcp_physics.physics.sim import SimSingle


TIME = 4
DT = Defaults.DT
DEVICE = Defaults.DEVICE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
hand_radius = 30
particle_radius = 10

np.random.seed(1)

def plot_mass_error(mask, m1, m2, save_path=None):
    err = np.zeros_like(mask).astype('float')
    err[mask] = m1 - m2

    ax = plt.subplot()
    im = ax.imshow(err, vmin=-0.1, vmax=0.1, cmap='plasma')

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    if save_path:
        plt.savefig(save_path)

    plt.clf()
    ax.cla()

def sys_id_demo(screen):
    mass_img_path = os.path.join(ROOT, 'fig/hammer_mass.png')
    bottom_fric_img_path = os.path.join(ROOT, 'fig/hammer_fric.png')

    sim = SimSingle.from_img(mass_img_path, bottom_fric_img_path, particle_radius=10, 
                    hand_radius=30)
    learning_rate = 1e-4
    max_iter = 100
    dist_hist = []
    mass_err_hist = []
    batch_size, TIME = 2, 2

    for k in range(max_iter):
        plot_mass_error(sim.mask, sim.mass_gt, 
                        sim.mass_est.detach().numpy(), 'tmp/mass_err_%03d.png'%k)

        rotation, offset = sim.random_rest_composite_pose(batch_size=batch_size)
        C, A, W = [], [], []
        for i in range(batch_size):
            composite_body_gt = sim.init_composite_object(sim.particle_radius, sim.mass_gt, 
                                sim.bottom_fric_gt, rotation=rotation[i], offset=offset[i])
            action = sim.sample_action(composite_body_gt)
            world_gt = sim.make_world(composite_body_gt, action)
            C.append(composite_body_gt)
            A.append(action)
            W.append(world_gt)    
        
        run_world_batch(W, run_time=TIME)

        X0, X1 = [], []
        for i in range(batch_size):
            X0.append(C[i].get_particle_pos())

            composite_body_est = sim.init_composite_object(sim.particle_radius, sim.mass_est, 
                                sim.bottom_fric_gt, rotation=rotation[i], offset=offset[i])
            world_est = sim.make_world(composite_body_est, A[i])
            run_world(world_est, run_time=TIME, print_time=True, screen=screen)
            X1.append(composite_body_est.get_particle_pos())
            reset_screen(screen)

        # compute loss
        dist = torch.sum(torch.norm(torch.stack(X0) - torch.stack(X1), dim=-1))
        dist.backward()
        grad = sim.mass_est.grad.data
        print(grad)
        grad.clamp_(1/learning_rate * -2e-3, 1/learning_rate * 2e-3)
        sim.mass_est = torch.clamp(sim.mass_est.data - learning_rate * grad, min=1e-5)
        sim.mass_est.requires_grad=True
        learning_rate *= 0.99
        print(k, '/', max_iter, dist.data.item())
        
        print('=======\n\n')
        # if abs((last_dist - dist).data.item()) < 1e-5:
        #     break
        last_dist = dist
        dist_hist.append(dist / sim.N)

    plot(dist_hist)

    


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