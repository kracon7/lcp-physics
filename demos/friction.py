import os
import sys

import time
import math
import cv2
import pygame
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from lcp_physics.physics.bodies import Circle, Rect, Hull, Composite
from lcp_physics.physics.constraints import TotalConstraint, FixedJoint
from lcp_physics.physics.forces import ExternalForce, Gravity, vert_impulse, hor_impulse
from lcp_physics.physics.utils import Defaults, Recorder
from lcp_physics.physics.world import World, run_world


TIME = 5
DT = Defaults.DT
DEVICE = Defaults.DEVICE

def make_world(radius):
    '''
    build world based on particle positions
    '''
    bodies = []
    joints = []
    mu_s = 0.1

    pos = [500, 300]
    c1 = Circle(pos, radius, fric_coeff_s=mu_s)
    bodies.append(c1)

    c2 = Circle([pos[0] + 2*radius + 7, pos[1]], radius, fric_coeff_s=mu_s)
    bodies.append(c2)

    # c3 = Circle([pos[0]-2*radius, pos[1]-100], radius, fric_coeff_s=mu_s)
    # bodies.append(c3)

    initial_force = torch.FloatTensor([0, 0.02, 0]).to(DEVICE)
    initial_force = Variable(initial_force, requires_grad=True)

    # Initial demo
    learned_force = lambda t: initial_force if t < 1 else ExternalForce.ZEROS
    c1.add_force(ExternalForce(learned_force))

    world = World(bodies, joints, dt=DT, extend=1, solver_type=3)#, post_stab=True, strict_no_penetration=False)
    return world
    

def fixed_joint_demo(screen):
    radius = 30
    world = make_world(radius)
    recorder = None
    recorder = Recorder(DT, screen)
    
    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))
    
    animation_dt = float(world.dt)
    elapsed_time = 0.
    prev_frame_time = -animation_dt
    start_time = time.time()

    f_log, v_log = [], []

    # world.set_v(torch.tensor([0, 0.1, 0, 0, 0, 0]).type_as(world.Jc()))

    while world.t < TIME:
        # record the force and velocities
        f = world.apply_forces(world.t).detach().numpy()
        v = world.get_v().detach().numpy()

        world.step()

        f_log.append(f)
        v_log.append(v)

        if screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            if elapsed_time - prev_frame_time >= animation_dt or recorder:
                prev_frame_time = elapsed_time

                screen.blit(background, (0, 0))
                update_list = []
                for body in world.bodies:
                    update_list += body.draw(screen, pixels_per_meter=1)
                for joint in world.joints:
                    update_list += joint[0].draw(screen, pixels_per_meter=1)

                if not recorder:
                    # Don't refresh screen if recording
                    pygame.display.update(update_list)
                    # pygame.display.flip()  # XXX
                else:
                    recorder.record(world.t)

            elapsed_time = time.time() - start_time
            if not recorder:
                # Adjust frame rate dynamically to keep real time
                wait_time = world.t - elapsed_time
                if wait_time >= 0 and not recorder:
                    wait_time += animation_dt  # XXX
                    time.sleep(max(wait_time - animation_dt, 0))
                #     animation_dt -= 0.005 * wait_time
                # elif wait_time < 0:
                #     animation_dt += 0.005 * -wait_time
                # elapsed_time = time.time() - start_time

        elapsed_time = time.time() - start_time
        print('\r ', '{} / {}  {} '.format(int(world.t), int(elapsed_time),
                                               1 / animation_dt), end='')

    fig, ax = plt.subplots(2,1)
    f_log = np.stack(f_log)
    v_log = np.stack(v_log)

    ax[0].plot(f_log[:, 1])
    ax[1].plot(v_log[:, 1])
    ax[0].set_ylabel('fx')
    ax[1].set_ylabel('vx')
    # ax[1].set_ylim(-2, 30)
    ax[1].set_xlabel('timestep')

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

    fixed_joint_demo(screen)