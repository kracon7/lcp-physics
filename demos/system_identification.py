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
from lcp_physics.physics.utils import Defaults, plot,    Recorder
from lcp_physics.physics.world import World, run_world


TIME = 5
DT = Defaults.DT
DEVICE = Defaults.DEVICE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def make_world(radius, fric_coeff_b):
    '''
    build world based on particle positions
    '''
    bodies = []
    joints = []

    pos = [500, 300]
    c1 = Circle(pos, radius, fric_coeff_b=[0.005, 0.45])
    bodies.append(c1)

    img = cv2.imread(os.path.join(ROOT, 'fig/T-Shape.png'))
    img = cv2.resize(img, (20, 15))
    x_cord, y_cord = np.where(img[:,:,0]<1)
    x_cord, y_cord = x_cord - x_cord.min(), y_cord - y_cord.min()
    particle_pos = 2 * radius * np.stack([x_cord, y_cord]).T + np.array([300, 300])
    composite_body = Composite(particle_pos, particle_radius)
    bodies += composite_body.bodies
    joints += composite_body.joints

    target = Circle([650, 300], radius, fric_coeff_b=fric_coeff_b)
    bodies.append(target)

    initial_force = torch.FloatTensor([0, 0.5, 0]).to(DEVICE)
    initial_force = Variable(initial_force, requires_grad=True)

    c1.add_no_contact(target)
    c2.add_no_contact(target)

    # Initial demo
    learned_force = lambda t: initial_force if t < 2 else ExternalForce.ZEROS
    c1.add_force(ExternalForce(learned_force))

    world = World(bodies, joints, dt=DT, extend=1, solver_type=1)#, post_stab=True, strict_no_penetration=False)
    return world, c2, target
    

def sys_id_demo(screen):
    radius = 30

    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

    mu = torch.FloatTensor([0, 0.44]).to(DEVICE)
    mu = Variable(mu, requires_grad=True)
    world, c2, target = make_world(radius, mu)
    recorder = None
    # recorder = Recorder(DT, screen)

    run_world(world, run_time=TIME, screen=screen, recorder=recorder)
    
    learning_rate = 0.001
    max_iter = 30

    dist_hist = []
    last_dist = 1e10
    for i in range(max_iter):
        world, c2, target = make_world(radius, mu)
        run_world(world, run_time=TIME, screen=screen, recorder=recorder)

        dist = (target.pos - c2.pos).norm()
        dist.backward()
        grad = mu.grad.data
        grad.clamp_(1/learning_rate * torch.tensor([-5e-4, -1e-2]), 
                    1/learning_rate * torch.tensor([5e-4, 1e-2]))

        mu = Variable(mu.data + learning_rate * grad, requires_grad=True)
        print('\n bottom friction coefficient: ', mu.detach().cpu().numpy().tolist())
        learning_rate *= 0.9
        print(i, '/', max_iter, dist.data.item())
        print(grad)
        print('=======')
        # if abs((last_dist - dist).data.item()) < 1e-5:
        #     break
        last_dist = dist
        dist_hist.append(dist)

    world, c2, target = make_world(radius, mu)
    rec = None
    # rec = Recorder(DT, screen)
    run_world(world, run_time=TIME, screen=screen, recorder=rec)
    dist = (target.pos - c2.pos).norm()
    print(dist.data.item())

    plot(dist_hist)



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

    sys_id_demo(screen)