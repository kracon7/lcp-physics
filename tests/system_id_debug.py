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
from lcp_physics.physics.utils import Defaults, plot, reset_screen,  Recorder, get_tensor
from lcp_physics.physics.world import World, run_world
from lcp_physics.physics.action import build_mesh, random_action

DT = Defaults.DT
DEVICE = Defaults.DEVICE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
hand_radius = 30
particle_radius = 10

np.random.seed(1)

class Sim():
    def __init__(self, particle_radius, hand_radius, mass_gt=None, 
                bottom_fric_gt=None, mass_est=None, bottom_fric_est=None, mask=None,
                DT = Defaults.DT, DEVICE = Defaults.DEVICE):
        
        self.particle_radius = particle_radius
        self.DEVICE = DEVICE
        self.hand_radius = hand_radius

    def sample_action(self, offset):
        theta = np.random.uniform(0, 2*np.pi)
        direction = np.array([np.cos(theta), np.sin(theta)]).astype('float32')
        # init random action (start position and direction)
        action = [offset + direction*(self.particle_radius + self.hand_radius + 1), -direction]
        return action

    def make_world(self, offset, mass, action):
        bodies = []
        joints = []

        c0 = Circle(offset, self.particle_radius, mass=mass, fric_coeff_b=[0.001, 0.01])
        bodies.append(c0)
        
        # init hand object
        c1 = Circle(action[0], self.hand_radius, fric_coeff_b=[0.005, 0.45])
        bodies.append(c1)

        # init force and apply force
        f = 2 * action[1]
        initial_force = torch.FloatTensor([0, f[0], f[1]]).to(self.DEVICE)
        c1.add_force(ExternalForce(initial_force, 0.2))
        
        # init world
        world = World(bodies, joints, dt=Defaults.DT, extend=1, solver_type=1)

        return world, c0

def sys_id_demo(screen):
    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

    sim = Sim(particle_radius, hand_radius, mass_gt=[0.01])
    learning_rate = 1e-4
    max_iter = 10
    dist_hist = []
    mass_err_hist = []
    batch_size, TIME = 1, 2

    mass_gt = torch.tensor([0.5], dtype=torch.float32)
    mass_est = torch.tensor([0.1], dtype=torch.float32, requires_grad=True)

    for k in range(max_iter):

        offset = np.random.uniform(low=[450, 250], high=[550, 350], 
                                    size=(2)).astype('float32')

        action = sim.sample_action(offset)

        world_gt, c0_gt = sim.make_world(offset, mass_gt, action)
        run_world(world_gt, run_time=TIME, print_time=True, screen=screen)

        world_est, c0_est = sim.make_world(offset, mass_est, action)
        run_world(world_est, run_time=TIME, print_time=True, screen=screen)

        X0, X1 = c0_gt.pos, c0_est.pos
        
        # compute loss
        dist = torch.sum(torch.norm(X0 - X1, dim=-1))
        dist.backward()
        grad = mass_est.grad.data
        print(grad)
        grad.clamp_(1/learning_rate * -2e-3, 1/learning_rate * 2e-3)

        mass_est = Variable(mass_est.data - learning_rate * grad, 
                                        requires_grad=True)
        # print('\n bottom friction coefficient: ', mu.detach().cpu().numpy().tolist())
        learning_rate *= 0.99
        print(k, '/', max_iter, dist.data.item())
        
        print('=======\n\n')
        dist_hist.append(dist / batch_size)
        mass_err_hist.append(np.abs(mass_gt.item() - mass_est.item()))

        reset_screen(screen)

    fig, ax = plt.subplots(2,1)
    ax[0].plot(dist_hist)
    ax[1].plot(mass_err_hist)
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

    sys_id_demo(screen)