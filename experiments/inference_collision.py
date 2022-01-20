'''
Gradient direction test using a single square and doing mass estimation
Force and torque is applied directly to the object
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

from lcp_physics.physics.bodies import Circle, Rect, Hull
from lcp_physics.physics.constraints import TotalConstraint, FixedJoint, Joint
from lcp_physics.physics.forces import ExternalForce, Gravity, vert_impulse, hor_impulse
from lcp_physics.physics.utils import Defaults, plot, reset_screen, Recorder
from lcp_physics.physics.world import World, run_world


TIME = 2
STOP_DIFF = 1e-4
DT = Defaults.DT
DEVICE = Defaults.DEVICE
DTYPE = Defaults.DTYPE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

np.random.seed(10)
torch.random.manual_seed(10)

def make_world(mass, force):
    bodies = []
    joints = []

    # c1 = Rect([500, 300], [20, 20], mass=mass)
    c1 = Circle([500, 300], 20, mass=mass)
    bodies.append(c1)

    ob = Rect([550, 300], [10, 200], mass=1)
    bodies.append(ob)
    joints.append(TotalConstraint(ob))

    push_force = lambda t: force if t < 0.5 else ExternalForce.ZEROS
    c1.add_force(ExternalForce(push_force))

    world = World(bodies, joints)
    return world


def positions_run_world(world, dt=Defaults.DT, run_time=10,
                        screen=None, recorder=None):
    positions = [torch.cat([b.p for b in world.bodies])]

    if screen is not None:
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

        animation_dt = dt
        elapsed_time = 0.
        prev_frame_time = -animation_dt
        start_time = time.time()

    while world.t < run_time:
        world.step()
        positions.append(torch.cat([b.p for b in world.bodies]))

        if screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            if elapsed_time - prev_frame_time >= animation_dt or recorder:
                prev_frame_time = elapsed_time

                screen.blit(background, (0, 0))
                update_list = []
                for body in world.bodies:
                    update_list += body.draw(screen)
                for joint in world.joints:
                    update_list += joint[0].draw(screen)

                if not recorder:
                    # Don't refresh screen if recording
                    pygame.display.update(update_list)
                else:
                    recorder.record(world.t)

            elapsed_time = time.time() - start_time
            # print('\r ', '{} / {}  {} '.format(int(world.t), int(elapsed_time),
            #                                   1 / animation_dt), end='')
    return positions

def main(screen):

    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

    mass_gt = torch.tensor([0.8]).type(DTYPE)
    mass_est = torch.rand_like(mass_gt, requires_grad=True)

    ang = torch.rand(1) - 0.5
    initial_force = torch.cat([0.1*torch.rand(1), torch.cos(ang), torch.sin(ang)]).type(DTYPE)
    
    print(mass_est)

    learning_rate = 0.05
    max_iter = 100
    loss_hist = []
    last_loss = 1e10

    # setup optimizer
    optim = torch.optim.Adam([mass_est], lr=learning_rate)

    world = make_world(mass_gt, initial_force)
    recorder = None
    # recorder = Recorder(DT, screen)
    # p = run_world(world, run_time=TIME, screen=screen, recorder=recorder)
    ground_truth_pos = positions_run_world(world, run_time=TIME, screen=screen, recorder=recorder)
    ground_truth_pos = [p.data for p in ground_truth_pos]
    ground_truth_pos = torch.cat(ground_truth_pos)

    mass_est_hist = []
    mass_est_hist.append(mass_est.clone().detach().numpy())
    for i in range(max_iter):

        optim.zero_grad()
        world = make_world(mass_est, initial_force)
        recorder = None
        # recorder = Recorder(DT, screen)
        estimated_pos = positions_run_world(world, run_time=TIME, screen=screen, recorder=recorder)
        estimated_pos = torch.cat(estimated_pos)
        estimated_pos = estimated_pos[:len(ground_truth_pos)]
        clipped_ground_truth_pos = ground_truth_pos[:len(estimated_pos)]

        loss = MSELoss()(estimated_pos, clipped_ground_truth_pos)
        loss.backward()
        optim.step()

        print('Iteration: {} / {}'.format(i+1, max_iter))
        print('Loss: ', loss.item())
        print('Gradient: ', mass_est.grad)
        print('Mass: ', mass_est.data)
        print('-----')
        if abs((last_loss - loss).item()) < STOP_DIFF:
            print('Loss changed by less than {} between iterations, stopping training.'
                  .format(STOP_DIFF))
            break
        last_loss = loss
        loss_hist.append(loss.item())

        reset_screen(screen)

        mass_est_hist.append(mass_est.clone().detach().numpy())

    plot(loss_hist)

    fig, ax = plt.subplots(1,1)
    mass_est_hist = np.stack(mass_est_hist)
    ax.plot(mass_est_hist, color=np.array([207., 40., 30.])/255)
    ax.plot(mass_gt*np.ones(mass_est_hist.shape[0]), color=np.array([255., 143., 133.])/255, linestyle='dashed')
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

    main(screen)