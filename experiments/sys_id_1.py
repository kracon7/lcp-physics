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


TIME = 2
STOP_DIFF = 1e-3
DT = Defaults.DT
DEVICE = Defaults.DEVICE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

np.random.seed(10)
torch.random.manual_seed(0)

def plot_mass_error(mask, m1, m2, save_path=None):
    err = np.zeros_like(mask).astype('float')
    err[mask] = m1 - m2

    ax = plt.subplot()
    im = ax.imshow(err, vmin=-0.2, vmax=0.2, cmap='plasma')

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

    obj_name = 'rod1'
    mass_img_path = os.path.join(ROOT, 'fig/%s_mass.png'%obj_name)
    bottom_fric_img_path = os.path.join(ROOT, 'fig/%s_fric.png'%obj_name)
    default_actions = {'rod1':  {'action_mag': 15, 'force_time': 0.4},
                       'drill': {'action_mag': 20, 'force_time': 0.4},
                       'hammer': {'action_mag': 15, 'force_time': 0.4}}

    sim = SimSingle.from_img(mass_img_path, bottom_fric_img_path, particle_radius=10, 
                    hand_radius=5)
    sim.bottom_fric_est = sim.bottom_fric_gt
    sim.action_mag = default_actions[obj_name]['action_mag']
    sim.force_time = default_actions[obj_name]['force_time']
    sim.mass_est = torch.rand_like(sim.mass_gt, requires_grad=True)
    
    learning_rate = 1e-2
    max_iter = 20
    loss_hist = []
    last_loss = 1e10

    # setup optimizer
    optim = torch.optim.RMSprop([sim.mass_est], lr=learning_rate)

    # run ground truth to get ground truth trajectory
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
    ground_truth_pos = sim.positions_run_world(world, run_time=TIME, screen=screen, recorder=recorder)
    ground_truth_pos = [p.data for p in ground_truth_pos]
    ground_truth_pos = torch.cat(ground_truth_pos)

    mass_err_hist = []
    for i in range(max_iter):

        plot_mass_error(sim.obj_mask, sim.mass_gt, 
                        sim.mass_est.detach().numpy(), 'tmp/mass_err_%03d.png'%i)

        # run estimation
        composite_body_est = sim.init_composite_object(
                                    sim.particle_pos0,
                                    sim.particle_radius, 
                                    sim.mass_est,
                                    sim.bottom_fric_gt,
                                    rotation=rotation,
                                    offset=offset)
        world = sim.make_world(composite_body_est, action, verbose=-1)
        recorder = None
        # recorder = Recorder(DT, screen)
        estimated_pos = sim.positions_run_world(world, run_time=TIME, screen=screen, recorder=recorder)
        estimated_pos = torch.cat(estimated_pos)
        estimated_pos = estimated_pos[:len(ground_truth_pos)]
        clipped_ground_truth_pos = ground_truth_pos[:len(estimated_pos)]

        optim.zero_grad()
        loss = MSELoss()(estimated_pos, clipped_ground_truth_pos)
        loss.backward()

        optim.step()

        print('Iteration: {} / {}'.format(i+1, max_iter))
        print('Loss:', loss.item())
        print('Gradient:', sim.mass_est.grad)
        print('-----')
        if abs((last_loss - loss).item()) < STOP_DIFF:
            print('Loss changed by less than {} between iterations, stopping training.'
                  .format(STOP_DIFF))
            break
        last_loss = loss
        loss_hist.append(loss.item())

        with torch.no_grad():
            sim.mass_est.clamp_(min=1e-5)

        reset_screen(screen)

    plot(loss_hist)


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