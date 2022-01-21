'''
Bottom friction behavior test using a rod consists of three squares
Force is applied directly to the object
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


TIME = 10
STOP_DIFF = 1e-4
DT = Defaults.DT
DEVICE = Defaults.DEVICE
DTYPE = Defaults.DTYPE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

np.set_printoptions(suppress=True, precision=5)
torch.random.manual_seed(10)

def make_world(mass, force):
    bodies = []
    joints = []

    c1 = Rect([500, 300], [20, 20], mass=mass, fric_coeff_b=[0, 0.5])
    c2 = Rect([500, 340], [20, 20], mass=mass, fric_coeff_b=[0, 0.5])
    c3 = Rect([500, 260], [20, 20], mass=mass, fric_coeff_b=[0, 0.5])
    bodies += [c1, c2, c3]
    joints += [FixedJoint(c1, c2), FixedJoint(c1, c3)]

    c1.add_no_contact(c2)
    c2.add_no_contact(c3)

    # ob = Rect([550, 300], [10, 200], mass=1)
    # bodies.append(ob)
    # joints.append(TotalConstraint(ob))

    push_force = lambda t: force if t < 1 else ExternalForce.ZEROS
    c1.add_force(ExternalForce(push_force))

    world = World(bodies, joints, extend=1)
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

    mass_gt = torch.tensor([0.5]).type(DTYPE)
    initial_force = torch.tensor([0., 1., 0]).type(DTYPE)
    world = make_world(mass_gt, initial_force)
    recorder = None
    # recorder = Recorder(DT, screen)
    ground_truth_pos = positions_run_world(world, run_time=TIME, screen=screen, recorder=recorder)
    ground_truth_pos = [p.data for p in ground_truth_pos]
    Xx = torch.stack(ground_truth_pos)[:, 1].numpy()

    Vx = (Xx[:-1] - Xx[1:]) / Defaults.DT
    Ax = (Vx[:-1] - Vx[1:]) / Defaults.DT

    fig, ax = plt.subplots(1,1)
    ax.plot(Ax, color=np.array([207., 40., 30.])/255)
    # ax.plot(mass_gt*np.ones(mass_est_hist.shape[0]), color=np.array([255., 143., 133.])/255, linestyle='dashed')
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