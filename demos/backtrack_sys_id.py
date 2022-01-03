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

np.random.seed(10)
torch.random.manual_seed(0)

def plot_mass_error(mask, m1, m2, save_path=None):
    err = np.zeros_like(mask).astype('float')
    err[mask] = m1 - m2

    ax = plt.subplot()
    im = ax.imshow(err, vmin=-0.15, vmax=0.15, cmap='plasma')

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

    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

    obj_name = 'drill'
    mass_img_path = os.path.join(ROOT, 'fig/%s_mass.png'%obj_name)
    bottom_fric_img_path = os.path.join(ROOT, 'fig/%s_fric.png'%obj_name)

    sim = SimSingle.from_img(mass_img_path, bottom_fric_img_path, particle_radius=10, 
                    hand_radius=10)
    sim.bottom_fric_est = sim.bottom_fric_gt
    sim.action_mag = 20
    sim.force_time = 0.3
    # sim.mass_est = 0.09 * torch.ones(sim.N).to(DEVICE)
    # sim.mass_est = Variable(sim.mass_est, requires_grad=True)
    gt_mean = sim.mass_gt.mean()
    sim.mass_est = 0.04 * torch.rand(sim.N) - 0.02 + gt_mean
    sim.mass_est.requires_grad = True
    
    max_iter = 20

    dist_hist, new_dist_hist = [], []
    mass_err_hist = []
    last_dist = 1e10
    for i in range(max_iter):
        plot_mass_error(sim.mask, sim.mass_gt, 
                        sim.mass_est.detach().numpy(), 'tmp/mass_err_%03d.png'%i)

        #################  Compute the gradient direction  ###################
        rotation, offset = torch.tensor([0]).type(Defaults.DTYPE), torch.tensor([[500, 300]]).type(Defaults.DTYPE)
        composite_body_gt = sim.init_composite_object(
                                    sim.particle_pos0,
                                    sim.particle_radius, 
                                    sim.mass_gt,
                                    sim.bottom_fric_gt,
                                    rotation=rotation,
                                    offset=offset)
        action = sim.sample_action(composite_body_gt)
        world = sim.make_world(composite_body_gt, action, verbose=-1)
        recorder = None
        # recorder = Recorder(DT, screen)
        run_world(world, run_time=TIME, screen=screen, recorder=recorder)
        X1 = composite_body_gt.get_particle_pos()
        
        composite_body = sim.init_composite_object(
                                    sim.particle_pos0,
                                    sim.particle_radius, 
                                    sim.mass_est,
                                    sim.bottom_fric_gt,
                                    rotation=rotation,
                                    offset=offset)
        world = sim.make_world(composite_body, action, verbose=-1)
        run_world(world, run_time=TIME, screen=screen, recorder=recorder)
        X2 = composite_body.get_particle_pos()
        
        dist = torch.sum(torch.norm(X1 - X2, dim=1))
        dist.backward()
        grad = torch.nan_to_num(sim.mass_est.grad.data)
        print(grad)

        #################  Backtracking line search  ###################
        cm = - 0.5 * torch.norm(grad)**2
        alpha, rho = 0.01 / torch.abs(grad).max(), 0.6
        count = 1

        while True:
            mass_est = torch.clamp(sim.mass_est.data - alpha * grad, min=1e-5)
            composite_body = sim.init_composite_object(
                                        sim.particle_pos0,
                                        sim.particle_radius, 
                                        mass_est,
                                        sim.bottom_fric_gt,
                                        rotation=rotation,
                                        offset=offset)
            world = sim.make_world(composite_body, action, verbose=-1)
            run_world(world, run_time=TIME, screen=screen, recorder=recorder)
            X2 = composite_body.get_particle_pos()
            new_dist = torch.sum(torch.norm(X1 - X2, dim=1))

            if new_dist - dist > alpha * cm:
                alpha *= rho
            else:
                break

            if count >= 3:
                break

            count += 1

        #################  Update mass_est  ###################
        learning_rate = alpha
        sim.mass_est = torch.clamp(sim.mass_est.data - learning_rate * grad, min=1e-5)
        sim.mass_est.requires_grad=True

        print(i, '/', max_iter, dist.data.item() / sim.N)
        
        print('=======\n\n')
        dist_hist.append(dist / sim.N)
        new_dist_hist.append(new_dist / sim.N)

        reset_screen(screen)

    fig, ax = plt.subplots(1, 1)
    ax.plot(dist_hist, color='r')
    ax.plot(new_dist_hist, color='b')
    plt.show()

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