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
from lcp_physics.physics.utils import Defaults, plot,    Recorder
from lcp_physics.physics.world import World, run_world, run_world_batch
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

    rotation, offset = sim.random_rest_composite_pose()
        # init composite object with offset and rotation
    composite_body = sim.init_composite_object(
                                    sim.particle_pos0,
                                    sim.particle_radius, 
                                    sim.mass_gt,
                                    sim.bottom_fric_gt,
                                    rotation=rotation,
                                    offset=offset)
    action = sim.sample_action(composite_body)
    world = sim.make_world(composite_body, action)

    world_after = run_world_batch([world], run_time=5)

    
    # learning_rate = 0.001
    # max_iter = 100

    # dist_hist = []
    # mass_err_hist = []
    # last_dist = 1e10
    # for i in range(max_iter):
        
    #     rotation, offset, action, X1, X2 = data_gen.run_episode()

    #     plot_mass_error(data_gen.mask, data_gen.mass_profile_gt, 
    #                     data_gen.mass_profile.detach().numpy(), 'tmp/mass_err_%03d.png'%i)
        
    #     dist = torch.sum(torch.norm(X1 - X2, dim=1))
    #     dist.backward()
    #     grad = data_gen.mass_profile.grad.data
    #     print(grad)
    #     grad.clamp_(1/learning_rate * -2e-3, 1/learning_rate * 2e-3)

    #     data_gen.mass_profile = Variable(data_gen.mass_profile.data - learning_rate * grad, 
    #                                     requires_grad=True)
    #     # print('\n bottom friction coefficient: ', mu.detach().cpu().numpy().tolist())
    #     learning_rate *= 0.9
    #     print(i, '/', max_iter, dist.data.item())
        
    #     print('=======\n\n')
    #     # if abs((last_dist - dist).data.item()) < 1e-5:
    #     #     break
    #     last_dist = dist
    #     dist_hist.append(dist / data_gen.N)

    # plot(dist_hist)



if __name__ == '__main__':

    sys_id_demo()