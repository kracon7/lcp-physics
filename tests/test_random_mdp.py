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
from matplotlib import path
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

def main(screen):
    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

    obj_name = 'hammer'
    mass_img_path = os.path.join(ROOT, 'fig/%s_mass.png'%obj_name)
    bottom_fric_img_path = os.path.join(ROOT, 'fig/%s_fric.png'%obj_name)
    default_actions = {'rod1':  {'action_mag': 15, 'force_time': 0.2},
                       'drill': {'action_mag': 20, 'force_time': 0.4},
                       'hammer': {'action_mag': 20, 'force_time': 0.2}}

    sim = SimSingle.from_img(mass_img_path, bottom_fric_img_path, particle_radius=10, 
                hand_radius=5)
    sim.action_mag = default_actions[obj_name]['action_mag']
    sim.force_time = default_actions[obj_name]['force_time']
    gt_mean = sim.mass_gt.mean()
    sim.mass_est = 0.06 * torch.rand(sim.N) - 0.03 + gt_mean
    sim.mass_est.requires_grad = True
    sim.bottom_fric_est = sim.bottom_fric_gt
    rotation, offset = torch.tensor([0]).type(Defaults.DTYPE), torch.tensor([[500, 300]]).type(Defaults.DTYPE)

    # compute the action start point coordinates
    s = 20 # action map size
    obj_mask = sim.mask.numpy()
    x_cord, y_cord = np.where(obj_mask)
    obj_origin = np.stack([x_cord[0], y_cord[0]])
    xx, yy = np.arange(s) - obj_origin[0], np.arange(s) - obj_origin[1]
    xx = np.interp(np.linspace(xx[0], xx[-1], 4*s), xx, xx)
    yy = np.interp(np.linspace(yy[0], yy[-1], 4*s), yy, yy)

    xx, yy = np.tile(xx, (xx.shape[0], 1)).T, np.tile(yy, (yy.shape[0], 1))

    action_start_pts = 2 * sim.particle_radius * np.stack([xx, yy], axis=-1).reshape(-1,2)

    # check whether start points are inside the polygon
    polygon = path.Path(sim.polygon_coord[sim.polygon[:,0]])
    inside = polygon.contains_points(action_start_pts).reshape(xx.shape[0], yy.shape[0])

    action_mask = np.zeros_like(inside).astype('bool')
    for i in range(1, inside.shape[0]-1):
        for j in range(1, inside.shape[1]-1):
            if not inside[i,j]:
                if inside[i+1,j] or inside[i-1,j] or inside[i,j+1] or inside[i,j-1] or \
                        inside[i+1,j+1] or inside[i-1,j+1] or inside[i+1,j-1] or inside[i-1,j-1]:
                    action_mask[i, j] = True

    sim.action_mask = torch.from_numpy(action_mask).reshape(-1)

    for i in range(5):
        # randomly generating Q matrix
        Q = torch.rand(sim.action_mask.shape[0])

        # mask Q matrix and select the best action
        valid_Q = Q[sim.action_mask]
        pt = action_start_pts[sim.action_mask][torch.argmax(valid_Q)]
        idx = np.argmin(np.linalg.norm(sim.polygon_coord - pt, axis=1))
        vtx, nml = sim.polygon_coord[idx], sim.normals[idx]
        start_pos = vtx + sim.hand_radius * nml

        while sim.overlap_check(sim.particle_pos0, 1.5*sim.particle_radius, 
                                start_pos, sim.hand_radius):
            start_pos += 0.5 * nml

        start_pos = start_pos + offset.numpy()
        action = [start_pos, -nml]

        # execute action
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