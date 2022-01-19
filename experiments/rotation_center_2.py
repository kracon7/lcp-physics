'''
Rod with 3 unknown mass. Define rotation center at make-world()
Rod composite is loaded from fig/rot_center_rod.png, it's a 12 x 1 grid
Regress mass with lcp-physics
'''

import os
import sys
import pickle
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
from lcp_physics.physics.constraints import TotalConstraint, FixedJoint, Joint
from lcp_physics.physics.forces import ExternalForce, Gravity, vert_impulse, hor_impulse
from lcp_physics.physics.utils import Defaults, plot, reset_screen, Recorder
from lcp_physics.physics.world import World, run_world
from lcp_physics.physics.action import build_mesh, random_action
from lcp_physics.physics.sim import SimSingle


TIME = 2
STOP_DIFF = 1e-3
DT = Defaults.DT
DEVICE = Defaults.DEVICE
DTYPE = Defaults.DTYPE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM_P = 12
P_RADIUS = 10

np.random.seed(10)
torch.random.manual_seed(10)
# torch.autograd.set_detect_anomaly(True)


def plot_mass_error(m1, m2, save_path=None):
    err = (m1 - m2).reshape(1,-1)

    ax = plt.subplot()
    im = ax.imshow(err, vmin=-0.08, vmax=0.08, cmap='plasma')
    # im = ax.imshow(err, cmap='plasma')

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    if save_path:
        plt.savefig(save_path)

    plt.clf()
    ax.cla()

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
    mass_gt = torch.tensor([0.3, 0.2, 0.4]).type(DTYPE)
    mass_est = torch.rand_like(mass_gt, requires_grad=True)
    print(mass_est)
    
    learning_rate = 0.05
    max_iter = 40
    loss_hist = []
    last_loss = 1e10

    # setup optimizer
    optim = torch.optim.Adam([mass_est], lr=learning_rate)

    batch_size, batch_gt_pos = 4, []

    # sample actions
    rotation, offset = torch.tensor([0]).type(Defaults.DTYPE), torch.tensor([[500, 300]]).type(Defaults.DTYPE)
    composite_body_gt = sim.init_composite_object(
                                    sim.particle_pos0,
                                    sim.particle_radius, 
                                    mass_gt.repeat_interleave(4),
                                    sim.bottom_fric_gt,
                                    rotation=rotation,
                                    offset=offset)
    batch_actions = sim.sample_action(composite_body_gt, batch_size)

    for b in range(batch_size):
        # run ground truth to get ground truth trajectory
        composite_body_gt = sim.init_composite_object(
                                    sim.particle_pos0,
                                    sim.particle_radius, 
                                    mass_gt.repeat_interleave(4),
                                    sim.bottom_fric_gt,
                                    rotation=rotation,
                                    offset=offset)
        action = batch_actions[b]
        world = sim.make_world(composite_body_gt, action, extend=0, verbose=-1)
        recorder = None
        # recorder = Recorder(DT, screen)
        ground_truth_pos = sim.positions_run_world(world, run_time=TIME, screen=screen, recorder=recorder)
        ground_truth_pos = [p.data for p in ground_truth_pos]
        ground_truth_pos = torch.cat(ground_truth_pos)

        batch_gt_pos.append(ground_truth_pos)

    mass_est_hist = []
    mass_est_hist.append(mass_est.clone().detach().numpy())
    for i in range(max_iter):

        plot_mass_error(mass_gt, mass_est.detach().numpy(), 'tmp/mass_err_%03d.png'%i)

        loss = 0
        optim.zero_grad()
        for b in range(batch_size):
            # run estimation
            composite_body_est = sim.init_composite_object(
                                        sim.particle_pos0,
                                        sim.particle_radius, 
                                        mass_est.repeat_interleave(4),
                                        sim.bottom_fric_gt,
                                        rotation=rotation,
                                        offset=offset)
            action, ground_truth_pos = batch_actions[b], batch_gt_pos[b]
            world = sim.make_world(composite_body_est, action, extend=0, verbose=-1)
            recorder = None
            # recorder = Recorder(DT, screen)
            estimated_pos = sim.positions_run_world(world, run_time=TIME, screen=screen, recorder=recorder)
            estimated_pos = torch.cat(estimated_pos)
            estimated_pos = estimated_pos[:len(ground_truth_pos)]
            clipped_ground_truth_pos = ground_truth_pos[:len(estimated_pos)]

            loss += MSELoss()(estimated_pos, clipped_ground_truth_pos)

        loss.backward()
        if torch.isnan(mass_est.grad).any():
            mass_est.grad.data = torch.zeros_like(mass_est)
        optim.step()

        print('Iteration: {} / {}'.format(i+1, max_iter))
        print('Loss: ', loss.item())
        print('Gradient: ', mass_est.grad)
        print('Mass: ', mass_est.data)
        print('-----')
        # if abs((last_loss - loss).item()) < STOP_DIFF:
        #     print('Loss changed by less than {} between iterations, stopping training.'
        #           .format(STOP_DIFF))
        #     break
        last_loss = loss
        loss_hist.append(loss.item())

        reset_screen(screen)

        mass_est_hist.append(mass_est.clone().detach().numpy())

    plot(loss_hist)

    with open('mass_est_hist.pkl', 'wb') as f:
	    pickle.dump(mass_est_hist, f)

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