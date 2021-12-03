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
from lcp_physics.physics.utils import Defaults, plot, reset_screen, Recorder, cross_2d
from lcp_physics.physics.world import World, run_world
from lcp_physics.physics.action import build_mesh, random_action
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


def sys_id_demo(screen):

    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

    recorder = Recorder(DT, screen)

    mass_img_path = os.path.join(ROOT, 'fig/hammer_mass.png')
    bottom_fric_img_path = os.path.join(ROOT, 'fig/hammer_fric.png')

    sim = SimSingle.from_img(mass_img_path, bottom_fric_img_path, particle_radius=10, 
                    hand_radius=20)
    
    composite_body = sim.init_composite_object(
                                    sim.particle_pos0,
                                    sim.particle_radius, 
                                    sim.mass_gt,
                                    sim.bottom_fric_gt,
                                    rotation=0,
                                    offset=[500, 300])
    p1 = composite_body.get_particle_pos()[:10]
    p1 = p1[1:] - p1[0]
    composite_body = sim.init_composite_object(
                                    sim.particle_pos0,
                                    sim.particle_radius, 
                                    sim.mass_gt,
                                    sim.bottom_fric_gt,
                                    rotation=-2.5,
                                    offset=[500, 300])
    p2 = composite_body.get_particle_pos()[:10]
    p2 = p2[1:] - p2[0]
    s = (p1[:, 0] * p2[:, 1] - p1[:, 1] * p2[:, 0]) / \
        (torch.norm(p1, dim=-1) * torch.norm(p2, dim=-1))
    c = torch.bmm(p1.unsqueeze(1), p2.unsqueeze(-1)).reshape(-1) / \
        (torch.norm(p1, dim=-1) * torch.norm(p2, dim=-1))
    theta = torch.atan2(s, c)

    bodies = []
    joints = []
    bodies += composite_body.bodies
    joints += composite_body.joints
    world = World(bodies, joints, dt=Defaults.DT, extend=1, solver_type=1)

    run_world(world, run_time=0.5, screen=screen, recorder=recorder)



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