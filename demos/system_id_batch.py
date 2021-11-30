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
from lcp_physics.physics.utils import Defaults, plot,  Recorder
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

class DataGen():
    def __init__(self):
        self.particle_pos0, self.mask = image_to_pos(mass_img_path)
    
        self.mass_profile_gt = image_to_mass(mass_img_path, self.mask)
        self.bottom_fric_profile_gt = image_to_bottom_fric(bottom_fric_img_path, self.mask)
    
        self.N = self.particle_pos0.shape[0]
    
        self.mass_profile = 0.02 * torch.ones(self.N).to(DEVICE)
        self.mass_profile = Variable(self.mass_profile, requires_grad=True)
        self.bottom_fric_profile = torch.FloatTensor([0.001, 0.1]).repeat(self.N, 1).to(DEVICE)
    
        self.particle_radius = 10
        self.hand_radius = 30

    def run_episode(self):
        rotation, offset = self.reset_composite_pose()
        # init composite object with offset and rotation
        composite_body_gt = init_composite_object(self.particle_pos0,
                                               self.particle_radius, 
                                               self.mass_profile_gt,
                                               self.bottom_fric_profile_gt,
                                               rotation=rotation,
                                               offset=offset)
        action = self.sample_action(composite_body_gt)
        world = make_world(self.particle_pos0, composite_body_gt, action)
        recorder = None
        # recorder = Recorder(DT, screen)
        run_world(world, run_time=TIME, screen=screen, recorder=recorder)

        X1 = composite_body_gt.get_particle_pos()
        
        composite_body = init_composite_object(self.particle_pos0,
                                               self.particle_radius, 
                                               self.mass_profile,
                                               self.bottom_fric_profile_gt,
                                               rotation=rotation,
                                               offset=offset)
        world = make_world(self.particle_pos0, composite_body, action)
        run_world(world, run_time=TIME, screen=screen, recorder=recorder)

        X2 = composite_body.get_particle_pos()

        return rotation, offset, action, X1, X2

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

def sys_id_demo():
    mass_img_path = os.path.join(ROOT, 'fig/hammer_mass.png')
    bottom_fric_img_path = os.path.join(ROOT, 'fig/hammer_fric.png')

    sim = SimSingle.from_img(mass_img_path, bottom_fric_img_path, particle_radius=10, 
                    hand_radius=30)
    learning_rate = 1e-4
    max_iter = 10
    dist_hist = []
    mass_err_hist = []
    batch_size, TIME = 5, 5

    for k in range(max_iter):
        plot_mass_error(sim.mask, sim.mass_gt, 
                        sim.mass_est.detach().numpy(), 'tmp/mass_err_%03d.png'%k)

        rotation, offset = sim.random_rest_composite_pose(batch_size=batch_size)
        C, A, W = [], [], []
        for i in range(batch_size):
            composite_body_gt = sim.init_composite_object(sim.particle_pos0, sim.particle_radius, 
                        sim.mass_gt, sim.bottom_fric_gt, rotation=rotation[i], offset=offset[i])
            action = sim.sample_action(composite_body_gt)
            world_gt = sim.make_world(composite_body_gt, action)
            C.append(composite_body_gt)
            A.append(action)
            W.append(world_gt)    
        
        run_world_batch(W, run_time=TIME)

        X0, X1 = [], []
        for i in range(batch_size):
            X0.append(C[i].get_particle_pos())

            composite_body_est = sim.init_composite_object(sim.particle_pos0, sim.particle_radius, 
                        sim.mass_est, sim.bottom_fric_gt, rotation=rotation[i], offset=offset[i])
            world_est = sim.make_world(composite_body_est, A[i])
            run_world_single(world_est, run_time=TIME)
            X1.append(composite_body_est.get_particle_pos())

        # compute loss
        dist = torch.sum(torch.norm(torch.stack(X0) - torch.stack(X1), dim=-1))
        dist.backward()
        grad = sim.mass_est.grad.data
        print(grad)
        grad.clamp_(1/learning_rate * -2e-3, 1/learning_rate * 2e-3)

        sim.mass_est = Variable(sim.mass_est.data - learning_rate * grad, 
                                        requires_grad=True)
        # print('\n bottom friction coefficient: ', mu.detach().cpu().numpy().tolist())
        learning_rate *= 0.99
        print(k, '/', max_iter, dist.data.item())
        
        print('=======\n\n')
        # if abs((last_dist - dist).data.item()) < 1e-5:
        #     break
        last_dist = dist
        dist_hist.append(dist / sim.N)

    plot(dist_hist)

    


if __name__ == '__main__':

    sys_id_demo()