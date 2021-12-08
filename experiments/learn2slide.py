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
from lcp_physics.physics.utils import Defaults, plot, reset_screen, Recorder, rel_pose
from lcp_physics.physics.world import World, run_world, run_world_batch
from lcp_physics.physics.action import random_action
from lcp_physics.physics.sim import SimSingle


TIME = 2
DT = Defaults.DT
DEVICE = Defaults.DEVICE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

np.random.seed(1)


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


def main(screen):

    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

    # ================    SYSTEM IDENTIFICATION   ===================

    # select object name
    object_names = ['hammer', 'driller', 'rod1', 'rod2']
    obj_name = object_names[0]
    mass_img_path = os.path.join(ROOT, 'fig/%s_mass.png'%obj_name)
    bottom_fric_img_path = os.path.join(ROOT, 'fig/%s_fric.png'%obj_name)

    # initializa sim 
    sim = SimSingle.from_img(mass_img_path, bottom_fric_img_path, particle_radius=10, 
                    hand_radius=20)
    
    learning_rate = 1e-4
    max_iter = 40

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

    # plot(dist_hist)

    # ================         PATH PLANNING      ===================

    # disable gradient
    sim.mass_est.requires_grad = False
    sim.bottom_fric_est.requires_grad = False

    # generate target pose
    target_list = torch.tensor([[1.9, 300, 100],
                                [0.3, 700, 300],
                                [-2.9, 650, 400],
                                [-1.1, 550, 150],
                                [2.5, 200, 400]]).float()
    idx = torch.randint(target_list.shape[0], (1,))

    step_size = torch.tensor([0.1, 20, 20]).float()
    epsilon = 5
    batch_size = 10

    # init the composite at center, extract particle positions
    target_pose = target_list[idx].reshape(-1)
    target_particle_pos = sim.transform_particles(target_pose[0], target_pose[1:])
    start_pose = torch.tensor([0, 500, 300]).float()
    start_particle_pos = sim.transform_particles(start_pose[0], start_pose[1:])
    curr_particle_pos = start_particle_pos

    dist = torch.mean(torch.norm(target_particle_pos - 
                                 start_particle_pos, dim=1)).item()

    # render the start and target image
    composite_body_gt = sim.init_composite_object(sim.particle_radius, sim.mass_gt, sim.bottom_fric_gt)
    composite_body_gt.draw(target_particle_pos, save_path=os.path.join(ROOT, 
                                                            'tmp/target.jpg'))
    composite_body_gt.draw(curr_particle_pos, save_path=os.path.join(ROOT, 
                                                            'tmp/start.jpg'))

    step = 0
    # iterate while distance is larger than epsilon
    while dist > epsilon and step < 50:
        curr_pose = rel_pose(sim.particle_pos0, curr_particle_pos)

        # compute the pose of the next node
        next_node = curr_pose + \
                    torch.clamp(target_pose - curr_pose, min=-step_size, max=step_size)

        # random sampling actions and do forward simulation
        curr_particle_pos = sim.transform_particles(curr_pose[0], curr_pose[1:])
        next_node_particle_pos = sim.transform_particles(next_node[0], next_node[1:])
        C, A, W = [], [], []
        for i in range(batch_size):
            composite_body_est = sim.init_composite_object(sim.particle_radius, sim.mass_est, 
                        sim.bottom_fric_gt, rotation=curr_pose[0], offset=curr_pose[1:])
            action = sim.sample_action(composite_body_est)
            world_est = sim.make_world(composite_body_est, action)
            C.append(composite_body_est)
            A.append(action)
            W.append(world_est)

        W = run_world_batch(W, run_time=TIME)

        # select the best action
        body_id = C[0].body_id
        curr_best = {'error': torch.inf, 'action': None}
        for i in range(batch_size):
            pos = W[i].get_p()[body_id, 1:]
            # C[i].draw(pos, save_path=os.path.join(ROOT, 'tmp/composite_%d.jpg'%i))
            diff = torch.mean(torch.norm(pos - next_node_particle_pos, dim=-1)).item()
            if diff < curr_best['error']:
                curr_best['error'] = diff
                curr_best['action'] = A[i]

        print('Next node: {}. Error: {}'.format(next_node.tolist(), curr_best['error']))

        # action execution 
        composite_body = sim.init_composite_object(sim.particle_radius, sim.mass_est, 
                        sim.bottom_fric_gt, rotation=curr_pose[0], offset=curr_pose[1:])
        world = sim.make_world(composite_body, curr_best['action'])
        run_world(world, run_time=TIME, screen=screen, show_mass=True)
        curr_particle_pos = composite_body.get_particle_pos()

        composite_body.draw(curr_particle_pos, save_path=os.path.join(ROOT, 
                                                            'tmp/rrt_step%d.jpg'%step))

        step += 1
        reset_screen(screen)

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