'''
Pushing a rod with uncertain mass
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
torch.random.manual_seed(0)

def plot_mass_stat(mask, gt, mean, var, save_path=None):
    I0, I1, I2 = np.zeros_like(mask).astype('float'), \
                 np.zeros_like(mask).astype('float'), \
                 np.zeros_like(mask).astype('float')
    I0[mask], I1[mask], I2[mask] = gt.numpy(), \
                                   mean.detach().cpu().numpy(), \
                                   var.detach().cpu().numpy()

    fig, ax = plt.subplots(1,3)
    im0 = ax[0].imshow(I0, vmin=0, vmax=0.3, cmap='plasma')
    im1 = ax[1].imshow(I1, vmin=0, vmax=0.3, cmap='plasma')
    im2 = ax[2].imshow(I2, vmin=0, vmax=0.3, cmap='plasma')

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im2, cax=cax)
    if save_path:
        plt.savefig(save_path)

    plt.close()

def get_stat(data):
    '''
    compute the mean and variace of different sequences of estimations
    Input:
        data -- list of estimated mass
    Output:
        mean -- torch tensor
        var -- torch tensor
    '''
    data = torch.stack(data)
    var, mean = torch.var_mean(data, 0)
    return torch.sqrt(var), mean


def sys_id_demo(screen):

    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

    mass_img_path = os.path.join(ROOT, 'fig/rod2_mass.png')
    bottom_fric_img_path = os.path.join(ROOT, 'fig/rod2_fric.png')

    num_guess = 1
    sim_list = []
    for _ in range(num_guess):
        sim = SimSingle.from_img(mass_img_path, bottom_fric_img_path, particle_radius=10, 
                    hand_radius=20)
        sim.mass_est = 0.1 * torch.ones(16)
        sim.mass_est.requires_grad = True
        sim.bottom_fric_est = sim.bottom_fric_gt
        sim.action_mag = 10
        sim.force_time = 0.3
        sim_list.append(sim)
    
    update_step_size = 1e-2
    max_iter = 100

    mass_est_hist, dist_hist = [], []
    last_dist = 1e10
    for i in range(max_iter):

        temp = []
        for sim in sim_list:
            temp.append(sim.mass_est)

            rotation, offset, action, X1, X2 = sim.run_episode_random(time=TIME, screen=screen)
            
            dist = torch.sum(torch.norm(X1 - X2, dim=1))
            dist.backward()
            grad = torch.nan_to_num(sim.mass_est.grad.data)
            magnitude = torch.max(torch.abs(grad)).item()
            learning_rate = update_step_size / magnitude

            print(grad, '\n', learning_rate*grad)

            sim.mass_est = torch.clamp(sim.mass_est.data - learning_rate * grad, min=1e-3)
            sim.mass_est.requires_grad=True
            # print('\n bottom friction coefficient: ', mu.detach().cpu().numpy().tolist())
            update_step_size *= 0.99
            print(i, '/', max_iter, dist.data.item())
            
            print('=======\n\n')

            reset_screen(screen)

            dist_hist.append(dist.item())

        # compute the mean and variance and plot
        var, mean = get_stat(temp)

        plot_mass_stat(sim.mask, sim.mass_gt, mean, var, save_path=os.path.join(ROOT, 
                                                            'tmp/mass_stat_%d.jpg'%i))

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