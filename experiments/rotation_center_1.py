'''
Rod with 3 unknown mass. Define rotation center at make-world()
Rod composite is loaded from fig/rot_center_rod.png, it's a 12 x 1 grid
Regress mass with lcp-physics
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

from lcp_physics.physics.bodies import Circle, Rect, Hull, Composite
from lcp_physics.physics.constraints import TotalConstraint, FixedJoint, Joint
from lcp_physics.physics.forces import ExternalForce, Gravity, vert_impulse, hor_impulse
from lcp_physics.physics.utils import Defaults, plot, reset_screen, Recorder
from lcp_physics.physics.world import World, run_world
from lcp_physics.physics.action import build_mesh, random_action
from lcp_physics.physics.sim import SimSingle


TIME = 2
STOP_DIFF = 1e-3
DT = Defaults.DT
DEVICE = Defaults.DEVICE
DTYPE = Defaults.DTYPE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM_P = 12
P_RADIUS = 10

np.random.seed(10)
torch.random.manual_seed(10)

def make_world(mass, rot_center):
    particle_pos = torch.stack([torch.zeros(NUM_P), 
                                torch.linspace(0, 2*P_RADIUS*(NUM_P-1), NUM_P)]
                              ).t().type(DTYPE) + torch.tensor([500, 300]).type(DTYPE)
    
    bodies = []
    joints = []
    N = particle_pos.shape[0]

    center = Circle(rot_center, 1, mass=0.01)
    bodies.append(center)
    joints.append(Joint(center, None, rot_center))

    for i in range(N):
        m = mass[i // int(N/3)]
        c = Rect(particle_pos[i], [2*P_RADIUS, 2*P_RADIUS], mass=m, fric_coeff_s=0, 
                    fric_coeff_b=[0.0, 0.1])
        bodies.append(c)
        joints += [FixedJoint(c, center)]

    for i in range(len(bodies)):
        for j in range(len(bodies)):
            bodies[i].add_no_contact(bodies[j])

    # init force and apply force
    initial_force = torch.FloatTensor([5000, 0, 0]).type(DTYPE).to(DEVICE)
    push_force = lambda t: initial_force if t < 0.5 else ExternalForce.ZEROS
    center.add_force(ExternalForce(push_force))

    world = World(bodies, joints, extend=0, solver_type=1)
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

    return positions

def plot_mass_error(m1, m2, save_path=None):
    err = (m1 - m2).reshape(1,-1)

    ax = plt.subplot()
    im = ax.imshow(err, vmin=-0.08, vmax=0.08, cmap='plasma')
    # im = ax.imshow(err, cmap='plasma')

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    if save_path:
        plt.savefig(save_path)

    plt.clf()
    ax.cla()

def main(screen):

    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

    mass_gt = torch.tensor([0.3, 0.2, 0.4]).type(DTYPE)
    mass_est = torch.rand_like(mass_gt, requires_grad=True)
    print(mass_est)
    rot_centers = torch.tensor([[0, -10],[0,70],[0,150],[0,230]]).type(DTYPE) + torch.tensor([500, 300]).type(DTYPE)
    
    learning_rate = 0.05
    max_iter = 40
    loss_hist = []
    last_loss = 1e10

    # setup optimizer
    optim = torch.optim.Adam([mass_est], lr=learning_rate)

    batch_size, batch_gt_pos = 4, []
    for b in range(batch_size):
        world = make_world(mass_gt, rot_centers[b])
        recorder = None
        # recorder = Recorder(DT, screen)
        ground_truth_pos = positions_run_world(world, run_time=TIME, screen=screen, recorder=recorder)
        ground_truth_pos = [p.data for p in ground_truth_pos]
        ground_truth_pos = torch.cat(ground_truth_pos)

        batch_gt_pos.append(ground_truth_pos)

    mass_est_hist = []
    mass_est_hist.append(mass_est.clone().detach().numpy())
    for i in range(max_iter):

        plot_mass_error(mass_gt, mass_est.detach().numpy(), 'tmp/mass_err_%03d.png'%i)

        loss = 0
        optim.zero_grad()
        for b in range(batch_size):
            world = make_world(mass_est, rot_centers[b])
            recorder = None
            # recorder = Recorder(DT, screen)
            estimated_pos = positions_run_world(world, run_time=TIME, screen=screen, recorder=recorder)
            estimated_pos = torch.cat(estimated_pos)
            ground_truth_pos = batch_gt_pos[b]
            estimated_pos = estimated_pos[:len(ground_truth_pos)]
            clipped_ground_truth_pos = ground_truth_pos[:len(estimated_pos)]

            loss += MSELoss()(estimated_pos, clipped_ground_truth_pos)

        loss.backward()
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

    with open('mass_est_hist.pkl', 'wb') as f:
	    pickle.dump(mass_est_hist, f)

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