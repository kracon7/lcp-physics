import math
import sys

import pygame

from lcp_physics.physics.bodies import Circle, Rect, Hull
from lcp_physics.physics.constraints import Joint, YConstraint, XConstraint, RotConstraint, TotalConstraint
from lcp_physics.physics.forces import ExternalForce, Gravity, vert_impulse, hor_impulse
from lcp_physics.physics.utils import Defaults, Recorder
from lcp_physics.physics.world import World, run_world

TIME = 6
DT = Defaults.DT


def debug_demo(screen):
    bodies = []
    joints = []
    # Ball hitting object constrained by 1 joint
    for i in range(1, 3):
        c = Circle([150, 150 + 80 * (i - 1)], 20)
        if i == 1:
            c.add_force(ExternalForce(vert_impulse, multiplier=500))
        bodies.append(c)
    joints.append(Joint(bodies[-1], None, [140, 220]))

    # Ball bouncing on body fixed in place
    for i in range(1, 3):
        c = Circle([300 + 1 * (i - 1), 150 + 80 * (i - 1)], 20)
        if i == 1:
            c.add_force(Gravity(g=100))
        # else:
        #     c.add_force(ExternalForce(neg_gravity, multiplier=100))
        bodies.append(c)
    joints.append(TotalConstraint(bodies[-1]))

    # 2 free ball collision angled
    for i in range(1, 3):
        c = Circle([225 - 10 * (i - 1), 300 + 80 * (i - 1)], 20)
        if i == 1:
            c.add_force(Gravity(g=100))
        bodies.append(c)

    # 2 free ball collision straight
    for i in range(1, 3):
        c = Circle([375, 300 + 80 * (i - 1)], 20)
        if i == 1:
            c.add_force(ExternalForce(vert_impulse, multiplier=500))
        bodies.append(c)

    r = Rect([300, 500], [40, 40], vel=[-1, 0, 0])
    r.add_force(Gravity(g=-100))
    bodies.append(r)

    clock = Circle([975, 575], 20, vel=[1, 0, 0])
    bodies.append(clock)

    world = World(bodies, joints, dt=DT, post_stab=True)
    run_world(world, run_time=10, screen=screen)


def chain_demo(screen):
    bodies = []
    joints = []
    restitution = 0.9

    # make chain of rectangles
    r = Rect([300, 50], [20, 60], restitution=restitution)
    bodies.append(r)
    joints.append(XConstraint(r))
    joints.append(YConstraint(r))
    for i in range(1, 10):
        r = Rect([300, 50 + 50 * i], [20, 60], restitution=restitution)
        bodies.append(r)
        joints.append(Joint(bodies[-1], bodies[-2], [300, 25 + 50 * i]))
        bodies[-1].add_no_contact(bodies[-2])
        bodies[-1].add_force(Gravity(g=100))

    # make projectile
    c = Circle([50, 500], 20, restitution=restitution)
    bodies.append(c)
    c.add_force(ExternalForce(hor_impulse, multiplier=2000))

    clock = Circle([975, 575], 20, vel=[1, 0, 0])
    bodies.append(clock)

    recorder = None
    # recorder = Recorder(DT, screen)
    world = World(bodies, joints, dt=DT, post_stab=True)
    run_world(world, run_time=TIME * 2, screen=screen, recorder=recorder)


def slide_demo(screen):
    bodies = []
    joints = []
    # restitution = Defaults.RESTITUTION
    restitution = 0
    fric_coeff_s = 0.15

    inclination = math.pi / 32
    r = Rect([inclination, 500, 300], [900, 10],
             restitution=restitution, fric_coeff_s=fric_coeff_s)
    bodies.append(r)
    joints.append(TotalConstraint(r))

    r = Rect([100, 100], [60, 60],
             restitution=restitution, fric_coeff_s=fric_coeff_s)
    # r = Hull([100, 100], [[30, 30], [-30, 30], [-30, -30], [30, -30]])
    bodies.append(r)
    r.add_force(Gravity(g=100))

    recorder = None
    # recorder = Recorder(DT, screen)
    world = World(bodies, joints, dt=DT)
    run_world(world, run_time=TIME, screen=screen, recorder=recorder)


def fric_demo(screen):
    restitution = 0.1
    fric_coeff_s = 1

    bodies = []
    joints = []

    def timed_force(t):
        if 1 < t < 2:
            return ExternalForce.RIGHT
        else:
            return ExternalForce.ZEROS

    r = Rect([400, 400], [900, 10], restitution=restitution, fric_coeff_s=fric_coeff_s)
    bodies.append(r)
    r.add_force(ExternalForce(timed_force, multiplier=100))
    r.add_force(Gravity(g=100))

    c = Circle([200, 364], 30, restitution=restitution, fric_coeff_s=fric_coeff_s)
    bodies.append(c)
    c.add_force(Gravity(g=100))

    c = Circle([50, 436], 30, restitution=restitution, fric_coeff_s=fric_coeff_s)
    bodies.append(c)
    joints.append(XConstraint(c))
    joints.append(YConstraint(c))
    c = Circle([800, 436], 30, restitution=restitution, fric_coeff_s=fric_coeff_s)
    bodies.append(c)
    joints.append(XConstraint(c))
    joints.append(YConstraint(c))

    clock = Circle([975, 575], 20, vel=[1, 0, 0])
    bodies.append(clock)

    recorder = None
    # recorder = Recorder(DT, screen)
    world = World(bodies, joints, dt=DT, post_stab=True)
    run_world(world, run_time=10, screen=screen, recorder=recorder)


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

    slide_demo(screen)
    # fric_demo(screen)
    # chain_demo(screen)
    # debug_demo(screen)
