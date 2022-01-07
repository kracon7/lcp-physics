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


def main():

    obj_name = 'hammer'
    mass_img_path = os.path.join(ROOT, 'fig/%s_mass.png'%obj_name)
    bottom_fric_img_path = os.path.join(ROOT, 'fig/%s_fric.png'%obj_name)
    default_actions = {'rod1':  {'action_mag': 15, 'force_time': 0.2},
                       'drill': {'action_mag': 20, 'force_time': 0.3},
                       'hammer': {'action_mag': 20, 'force_time': 0.2}}

    sim = SimSingle.from_img(mass_img_path, bottom_fric_img_path, particle_radius=10, 
                hand_radius=5)
    sim.action_mag = default_actions[obj_name]['action_mag']
    sim.force_time = default_actions[obj_name]['force_time']
    gt_mean = sim.mass_gt.mean()
    sim.mass_est = 0.06 * torch.rand(sim.N) - 0.03 + gt_mean
    sim.mass_est.requires_grad = True
    sim.bottom_fric_est = sim.bottom_fric_gt

    # compute the action start point coordinates
    s = 20 # action map size
    mask = sim.mask.numpy()
    x_cord, y_cord = np.where(mask)
    mask_origin = np.stack([x_cord[0], y_cord[0]])
    xx, yy = np.arange(s) - mask_origin[0], np.arange(s) - mask_origin[1]
    xx = np.interp(np.linspace(xx[0], xx[-1], 10*s), xx, xx)
    yy = np.interp(np.linspace(yy[0], yy[-1], 10*s), yy, yy)

    xx, yy = np.tile(xx, (xx.shape[0], 1)).T, np.tile(yy, (yy.shape[0], 1))

    action_start_pts = 2 * sim.particle_radius * np.stack([xx, yy], axis=-1)

    # check whether start points are inside the polygon
    polygon = path.Path(sim.polygon_coord[sim.polygon[:,0]])
    inside = polygon.contains_points(action_start_pts.reshape(-1,2)).reshape(xx.shape[0], yy.shape[0])

    boundary = np.zeros_like(inside)
    for i in range(1, inside.shape[0]-1):
        for j in range(1, inside.shape[1]-1):
            if not inside[i,j]:
                if inside[i+1,j] or inside[i-1,j] or inside[i,j+1] or inside[i,j-1] or \
                        inside[i+1,j+1] or inside[i-1,j+1] or inside[i+1,j-1] or inside[i-1,j-1]:
                    boundary[i, j] = 1

    fig, ax = plt.subplots(1,4)
    ax[0].imshow(sim.mask)
    ax[1].imshow(inside)
    ax[2].imshow(boundary)

    # ax[3].axis('square')
    polygon, polygon_coord, normals = sim.polygon, sim.polygon_coord, sim.normals
    for i in range(polygon.shape[0]):
        pt_coord = polygon_coord[polygon[i, 0]]
        ax[3].plot([pt_coord[0], pt_coord[0] + 3*normals[i,0]], 
                   [pt_coord[1], pt_coord[1] + 3*normals[i,1]], color='r')
    ax[3].plot([polygon_coord[0,0], polygon_coord[-1,0]], 
               [polygon_coord[0,1], polygon_coord[-1,1]], color='deepskyblue')
    ax[3].plot(polygon_coord[:, 0], polygon_coord[:,1], color='deepskyblue')

    valid_start_pts = action_start_pts[boundary>0]
    for pt in valid_start_pts:
        ax[3].plot(pt[0], pt[1], 'yx')

    ax[3].invert_yaxis()
    ax[3].set_xlabel('x (same in pygame)')
    ax[3].set_ylabel('y (same in pygame)')

    

    plt.show()

if __name__ == '__main__':
    main()