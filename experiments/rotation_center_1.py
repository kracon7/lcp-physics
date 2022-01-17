'''
Rod with 3 unknown mass. Define rotation center at make-world()
Rod composite is loaded from fig/rot_center_rod.png, it's a 12 x 1 grid
Regress mass with lcp-physics
'''

import os
import sys

import time
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


def main(screen):

    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

    obj_name = 'rot_center_rod'
    mass_img_path = os.path.join(ROOT, 'fig/%s_mass.png'%obj_name)
    bottom_fric_img_path = os.path.join(ROOT, 'fig/%s_fric.png'%obj_name)

    sim = SimSingle.from_img(mass_img_path, bottom_fric_img_path, particle_radius=10, 
                    hand_radius=5)
    sim.bottom_fric_est = sim.bottom_fric_gt
    sim.action_mag = 15
    sim.force_time = 0.4

    # ground truth mass. shape (3, )
    mass_gt = torch.tensor([0.5, 1, 2]).type(Defaults.DTYPE)
    # tile it and pass it to sim
    sim.mass_gt = mass_gt.repeat_interleave(4)
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

    main(screen)