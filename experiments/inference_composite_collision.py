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
from lcp_physics.physics.sim import SimSingle


TIME = 10
STOP_DIFF = 1e-4
DT = Defaults.DT
DEVICE = Defaults.DEVICE
DTYPE = Defaults.DTYPE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

np.random.seed(10)
torch.random.manual_seed(10)


def make_world(composite_body):
    bodies = []
    joints = []
    bodies += composite_body.bodies
    joints += composite_body.joints

    # add force to composite bodies
    initial_force = torch.FloatTensor([0, 0, -5]).to(Defaults.DEVICE)
    push_force = lambda t: initial_force if t < 4 else ExternalForce.ZEROS
    bodies[-1].add_force(ExternalForce(push_force))
    
    # add obstacle
    ob = Circle([670, 310], 20, mass=10)
    bodies.append(ob)
    joints.append(TotalConstraint(ob))

    # init world
    world = World(bodies, joints, dt=Defaults.DT)

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
            
    return positions

def main(screen):

    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

    obj_name = 'hammer'
    mass_img_path = os.path.join(ROOT, 'fig/%s_mass.png'%obj_name)
    bottom_fric_img_path = os.path.join(ROOT, 'fig/%s_fric.png'%obj_name)
    sim = SimSingle.from_img(mass_img_path, bottom_fric_img_path, particle_radius=10, 
                    hand_radius=10)
    sim.bottom_fric_est = sim.bottom_fric_gt

    mass_gt = torch.tensor([0.8]).type(DTYPE)
    mass_est = torch.rand_like(mass_gt, requires_grad=True)

    print(mass_est)

    learning_rate = 0.02
    max_iter = 100
    loss_hist = []
    last_loss = 1e10

    # setup optimizer
    optim = torch.optim.Adam([mass_est], lr=learning_rate)

    # run ground truth to get ground truth trajectory
    rotation, offset = torch.tensor([0]).type(Defaults.DTYPE), torch.tensor([[500, 300]]).type(Defaults.DTYPE)
    composite_body_gt = sim.init_composite_object(
                                sim.particle_pos0,
                                sim.particle_radius, 
                                mass_gt,
                                sim.bottom_fric_gt,
                                rotation=rotation,
                                offset=offset)
    world = make_world(composite_body_gt)
    recorder = None
    # recorder = Recorder(DT, screen)
    ground_truth_pos = positions_run_world(world, run_time=TIME, screen=screen, recorder=recorder)
    ground_truth_pos = [p.data for p in ground_truth_pos]
    ground_truth_pos = torch.cat(ground_truth_pos)

    mass_est_hist = []
    mass_est_hist.append(mass_est.clone().detach().numpy())
    for i in range(max_iter):

        optim.zero_grad()
        composite_body_est = sim.init_composite_object(
                                sim.particle_pos0,
                                sim.particle_radius, 
                                mass_est,
                                sim.bottom_fric_gt,
                                rotation=rotation,
                                offset=offset)
        world = make_world(composite_body_est)
        recorder = None
        # recorder = Recorder(DT, screen)
        estimated_pos = positions_run_world(world, run_time=TIME, screen=screen, recorder=recorder)
        estimated_pos = torch.cat(estimated_pos)
        estimated_pos = estimated_pos[:len(ground_truth_pos)]
        clipped_ground_truth_pos = ground_truth_pos[:len(estimated_pos)]

        loss = MSELoss()(estimated_pos, clipped_ground_truth_pos)
        loss.backward()
        if mass_est.grad.data.isnan().any():
            continue
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