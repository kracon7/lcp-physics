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
    im2 = ax[2].imshow(I2, vmin=0, vmax=0.1, cmap='plasma')

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

    obj_name = 'hammer'
    mass_img_path = os.path.join(ROOT, 'fig/%s_mass.png'%obj_name)
    bottom_fric_img_path = os.path.join(ROOT, 'fig/%s_fric.png'%obj_name)
    default_actions = {'rod1':  {'action_mag': 15, 'force_time': 0.2},
                       'drill': {'action_mag': 20, 'force_time': 0.3},
                       'hammer': {'action_mag': 20, 'force_time': 0.2}}

    num_guess = 4
    sim_list = []
    for _ in range(num_guess):
        sim = SimSingle.from_img(mass_img_path, bottom_fric_img_path, particle_radius=10, 
                    hand_radius=5)
        sim.action_mag = default_actions[obj_name]['action_mag']
        sim.force_time = default_actions[obj_name]['force_time']
        gt_mean = sim.mass_gt.mean()
        sim.mass_est = 0.06 * torch.rand(sim.N) - 0.03 + gt_mean
        sim.mass_est.requires_grad = True
        sim.bottom_fric_est = sim.bottom_fric_gt
        sim_list.append(sim)
    
    max_iter = 30

    mass_est_hist = []
    dist_hist = [[] for _ in range(num_guess)]
    for i in range(max_iter):

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

        temp = []
        for k, sim in enumerate(sim_list):
            temp.append(sim.mass_est)

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
            alpha, rho = torch.clamp(0.01 / torch.abs(grad).max(), min=1e-5, max=1e-3), 0.6
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

            print(i, '/', max_iter, '%8d'%k, '/', num_guess, '   dist: ', dist.data.item()/sim.N)            
            print('=======\n\n')

            reset_screen(screen)

            dist_hist[k].append(dist.item())

        # compute the mean and variance and plot
        var, mean = get_stat(temp)

        plot_mass_stat(sim.mask, sim.mass_gt, mean, var, save_path=os.path.join(ROOT, 
                                                            'tmp/mass_stat_%d.jpg'%i))
    
    fig, ax = plt.subplots(1,1)
    ax.plot(np.array(dist_hist).T)
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