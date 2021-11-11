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
from lcp_physics.physics.action import build_mesh


TIME = 5
DT = Defaults.DT
DEVICE = Defaults.DEVICE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def init_composite_object(particle_radius, mass_img, fric_img):
    img = cv2.imread(os.path.join(ROOT, mass_img))
    mask = img[:,:,0] < 255
    x_cord, y_cord = np.where(mask)
    x_cord, y_cord = x_cord - x_cord.min(), y_cord - y_cord.min()
    particle_pos = 2 * particle_radius * np.stack([x_cord, y_cord]).T + np.array([500, 300])

    mass_profile = img[:,:,0][mask].astype('float') / 1e3

    img = cv2.imread(os.path.join(ROOT, fric_img)).astype('float') / 255
    bottom_fric_profile = np.stack([img[:,:,2][mask]/100, img[:,:,1][mask]], axis=-1)

    composite_body = Composite(particle_pos, particle_radius, mass=mass_profile, 
                                fric_coeff_b=bottom_fric_profile)
    return composite_body

def make_world():
    
    initial_force = torch.FloatTensor([0, 0.5, 0]).to(DEVICE)
    initial_force = Variable(initial_force)
    push_force = lambda t: initial_force if t < 2 else ExternalForce.ZEROS

    hand_pos = [300, 330]
    hand_radius = 30

    bodies = []
    joints = []

    c1 = Circle(hand_pos, hand_radius, fric_coeff_b=[0.005, 0.45])
    bodies.append(c1)

    particle_radius = 10

    composite_body = init_composite_object(particle_radius, 'fig/hammer_mass.png', 'fig/hammer_fric.png')
    
    bodies += composite_body.bodies
    joints += composite_body.joints

    # target = Circle([650, 300], radius, fric_coeff_b=fric_coeff_b)
    # bodies.append(target)

    c1.add_force(ExternalForce(push_force))

    world = World(bodies, joints, dt=DT, extend=1, solver_type=1)#, post_stab=True, strict_no_penetration=False)
    return world
    

def sys_id_demo(screen):

    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

    world = make_world()
    recorder = None
    # recorder = Recorder(DT, screen)

    run_world(world, run_time=TIME, screen=screen, recorder=recorder)
    
    learning_rate = 0.001
    max_iter = 30

    dist_hist = []
    last_dist = 1e10
    for i in range(max_iter):
        world = make_world(radius, mu)
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

    world = make_world(radius, mu)
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