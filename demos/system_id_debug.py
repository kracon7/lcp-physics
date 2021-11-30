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
from lcp_physics.physics.world import World, run_world, run_world_batch, run_world_single
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

        self.mass_gt = get_tensor(mass_gt)
        self.bottom_fric_gt = torch.FloatTensor([[0.001, 0.1]]).to(DEVICE)

        self.mass_est = get_tensor([0.03])
        self.mass_est = Variable(self.mass_est, requires_grad=True)

        self.bottom_fric_est = torch.FloatTensor([[0.001, 0.1]]).to(DEVICE)
        self.bottom_fric_est = Variable(self.bottom_fric_est, requires_grad=True)

        self.hand_radius = hand_radius

    def random_rest_composite_pose(self, batch_size=1):
        offset = np.random.uniform(low=[450, 250], high=[550, 350], size=(batch_size, 2))
        return offset.astype('float32')

    def sample_action(self, composite_body):
        # get composite object particle pos
        p = composite_body.get_particle_pos().reshape(2)
        theta = np.random.uniform(0, 2*np.pi)
        direction = np.array([np.cos(theta), np.sin(theta)]).astype('float32')
        # init random action (start position and direction)
        action = [p + direction*(self.particle_radius + self.hand_radius + 1), -direction]
        return action

    def init_composite_object(self, particle_radius, mass_profile, 
                    bottom_fric_profile, offset=[0,0]):
        particle_pos = np.array(offset).reshape(1,2).astype('float32')

        composite_body = Composite(particle_pos, particle_radius, mass=mass_profile, 
                                    fric_coeff_b=bottom_fric_profile)
        return composite_body

    def make_world(self, composite_body, action):
        bodies = []
        joints = []
        bodies += composite_body.bodies
        joints += composite_body.joints
        
        # init hand object
        c1 = Circle(action[0], self.hand_radius, fric_coeff_b=[0.005, 0.45])
        bodies.append(c1)

        # init force and apply force
        f = 2 * action[1]
        initial_force = torch.FloatTensor([0, f[0], f[1]]).to(self.DEVICE)
        c1.add_force(ExternalForce(initial_force, 0.3))
        
        # init world
        world = World(bodies, joints, dt=Defaults.DT, extend=1, solver_type=1)

        return world

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

    for k in range(max_iter):

        offset = sim.random_rest_composite_pose(batch_size=batch_size)
        C, A, W = [], [], []
        for i in range(batch_size):
            composite_body_gt = sim.init_composite_object(sim.particle_radius, 
                        sim.mass_gt, sim.bottom_fric_gt, offset=offset[i])
            action = sim.sample_action(composite_body_gt)
            world_gt = sim.make_world(composite_body_gt, action)
            C.append(composite_body_gt)
            A.append(action)
            W.append(world_gt)    
        
        run_world_batch(W, run_time=TIME)

        X0, X1 = [], []
        for i in range(batch_size):
            X0.append(C[i].get_particle_pos())

            composite_body_est = sim.init_composite_object(sim.particle_radius, 
                        sim.mass_est, sim.bottom_fric_gt, offset=offset[i])
            world_est = sim.make_world(composite_body_est, A[i])
            run_world(world_est, run_time=TIME, print_time=True, screen=screen)
            X1.append(composite_body_est.get_particle_pos())

        # compute loss
        dist = torch.sum(torch.norm(torch.stack(X0) - torch.stack(X1), dim=-1))
        dist.backward()
        grad = sim.mass_est.grad.data
        print(grad)
        grad.clamp_(1/learning_rate * -2e-3, 1/learning_rate * 2e-3)

        sim.mass_est = Variable(sim.mass_est.data + learning_rate * grad, 
                                        requires_grad=True)
        # print('\n bottom friction coefficient: ', mu.detach().cpu().numpy().tolist())
        learning_rate *= 0.99
        print(k, '/', max_iter, dist.data.item())
        
        print('=======\n\n')
        dist_hist.append(dist / batch_size)
        mass_err_hist.append(sim.mass_gt.item()-sim.mass_est.item())

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