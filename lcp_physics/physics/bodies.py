from collections import defaultdict
import math

import ode
import pygame
import scipy.spatial as spatial
import numpy as np

import torch

from .utils import Indices, Defaults, get_tensor, cross_2d, rotation_matrix
from .constraints import FixedJoint

X = Indices.X
Y = Indices.Y
DIM = Defaults.DIM

class GeomCircle(object):
    '''base class for circle geom 
    '''
    def __init__(self, position, radius):
        self.position = position
        self.radius = radius

    def setPosition(self, pos):
        self.pos = pos

class Body(object):
    """Base class for bodies.
    """
    def __init__(self, pos, vel=(0, 0, 0), mass=1, restitution=Defaults.RESTITUTION,
                 fric_coeff_s=Defaults.FRIC_COEFF_S, fric_coeff_b=Defaults.FRIC_COEFF_B, 
                 eps=Defaults.EPSILON, col=(255, 0, 0), thickness=1):
        # get base tensor to define dtype, device and layout for others
        self._set_base_tensor(locals().values())

        self.eps = get_tensor(eps, base_tensor=self._base_tensor)
        # rotation & position vectors
        pos = get_tensor(pos, base_tensor=self._base_tensor)
        if pos.size(0) == 2:
            self.p = torch.cat([pos.new_zeros(1), pos])
        else:
            self.p = pos
        self.rot = self.p[0:1]
        self.pos = self.p[1:]

        # linear and angular velocity vector
        vel = get_tensor(vel, base_tensor=self._base_tensor)
        if vel.size(0) == 2:
            self.v = torch.cat([vel.new_zeros(1), vel])
        else:
            self.v = vel

        self.mass = get_tensor(mass, self._base_tensor)
        self.ang_inertia = self._get_ang_inertia(self.mass)
        # M can change if object rotates, not the case for now
        self.M = self.v.new_zeros(len(self.v), len(self.v))
        ang_sizes = [1, 1]
        self.M[:ang_sizes[0], :ang_sizes[1]] = self.ang_inertia
        self.M[ang_sizes[0]:, ang_sizes[1]:] = torch.eye(DIM).type_as(self.M) * self.mass

        self.fric_coeff_s = get_tensor(fric_coeff_s, base_tensor=self._base_tensor)
        self.fric_coeff_b = get_tensor(fric_coeff_b, base_tensor=self._base_tensor)
        self.restitution = get_tensor(restitution, base_tensor=self._base_tensor)
        self.forces = []

        self.col = col
        self.thickness = thickness

        self._create_geom()

    def _set_base_tensor(self, args):
        """Check if any tensor provided and if so set as base tensor to
           use as base for other tensors' dtype, device and layout.
        """
        if hasattr(self, '_base_tensor') and self._base_tensor is not None:
            return

        for arg in args:
            if isinstance(arg, torch.Tensor):
                self._base_tensor = arg
                return

        # if no tensor provided, use defaults
        self._base_tensor = get_tensor(0, base_tensor=None)
        return

    def _create_geom(self):
        raise NotImplementedError

    def _get_ang_inertia(self, mass):
        raise NotImplementedError

    def move(self, dt):
        new_p = self.p + self.v * dt
        self.set_p(new_p)

    def set_p(self, new_p):
        self.p = new_p
        # Reset memory pointers
        self.rot = self.p[0:1]
        self.pos = self.p[1:]

        self.geom.setPosition(self.pos)

    def apply_forces(self, t):
        if len(self.forces) == 0:
            return self.v.new_zeros(len(self.v))
        else:
            return sum([f.force(t) for f in self.forces])

    def add_no_contact(self, other):
        self.geom.no_contact.add(other.geom)
        other.geom.no_contact.add(self.geom)

    def add_force(self, f):
        self.forces.append(f)
        f.set_body(self)

    def draw(self, screen, pixels_per_meter=1):
        raise NotImplementedError


class Circle(Body):
    def __init__(self, pos, rad, vel=(0, 0, 0), mass=1, restitution=Defaults.RESTITUTION,
                 fric_coeff_s=Defaults.FRIC_COEFF_S, fric_coeff_b=Defaults.FRIC_COEFF_B, 
                 eps=Defaults.EPSILON, col=(255, 0, 0), thickness=1):
        self._set_base_tensor(locals().values())
        self.rad = get_tensor(rad, base_tensor=self._base_tensor)
        super().__init__(pos, vel=vel, mass=mass, restitution=restitution,
                         fric_coeff_s=fric_coeff_s, fric_coeff_b=fric_coeff_b,
                         eps=eps, col=col, thickness=thickness)

    def _get_ang_inertia(self, mass):
        return mass / 2 * self.rad * self.rad

    def _create_geom(self):
        self.geom = GeomCircle(None, self.rad.item() + self.eps.item())
        self.geom.setPosition(torch.cat([self.pos,
                                         self.pos.new_zeros(1)]))
        self.geom.no_contact = set()

    def move(self, dt):
        super().move(dt)

    def set_p(self, new_p):
        super().set_p(new_p)

    def draw(self, screen, pixels_per_meter=1, show_mass=True):
        center = (self.pos.detach().cpu().numpy() * pixels_per_meter).astype(int)
        rad = int(self.rad.item() * pixels_per_meter)
        if show_mass:
            thickness = 0
            col = (max(0, min(255, int(self.mass.item() * 1e3))), 
                   0, 0)
        else:
            thickness = self.thickness
            col = (255, 0, 0)

        # draw radius to visualize orientation
        r = pygame.draw.line(screen, (0, 0, 255), center,
                             center + [math.cos(self.rot.item()) * rad,
                                       math.sin(self.rot.item()) * rad],
                             self.thickness)
        # draw circle
        c = pygame.draw.circle(screen, col, center, rad, thickness)
        return [c, r]

class Composite():
    """rigid body based on particle formulation"""
    def __init__(self, particle_pos, radius, mass=0.01, 
                fric_coeff_s=Defaults.FRIC_COEFF_S, fric_coeff_b=Defaults.FRIC_COEFF_B):
        '''
        Input:
            particle_pos -- ndarray (N, 2) 2D position of particles
            radius -- radius of each particle
            mass -- float or ndarray (N,), mass of each particle
            fric_coeff_s -- side friction coefficient
            fric_coeff_b -- bottom friction coefficient
        '''
        # super(ClassName, self).__init__()
        # self.args = args

        bodies = []
        joints = []
        N = particle_pos.shape[0]
        if isinstance(mass, float):
            mass = mass * np.ones(N)

        if isinstance(fric_coeff_b, list):
            fric_coeff_b = np.ones((N, 2)) * np.array(fric_coeff_b)

        for i in range(N):
            c = Circle(particle_pos[i], radius, mass=mass[i], 
                        fric_coeff_s=fric_coeff_s, fric_coeff_b=fric_coeff_b[i])

            bodies.append(c)

        for i in range(N-1):
            joints += [FixedJoint(bodies[i], bodies[-1])]

        # # add contact exclusion
        # no_contact = self.find_neighbors(particle_pos, radius)
        # for i in range(N):
        #     neighbors = no_contact[i]
        #     for j in neighbors:
        #         bodies[i].add_no_contact(bodies[j])

        # add contact exclusion
        for i in range(N):
            for j in range(N):
                bodies[i].add_no_contact(bodies[j])

        self.bodies = bodies
        self.joints = joints
        self.radius = radius
        self.mass = mass
        self.fric_coeff_b = fric_coeff_b

    def get_particle_pos(self):
        pos = []
        for b in self.bodies:
            pos.append(b.pos)
        pos = torch.stack(pos)
        return pos

    def find_neighbors(self, particle_pos, radius):
        '''
        find neighbors of particles for contact exlusion
        '''
        point_tree = spatial.cKDTree(particle_pos)
        neighbors_list = point_tree.query_ball_point(particle_pos, 2.5*radius)

        no_contact = defaultdict(list)
        for i in range(particle_pos.shape[0]):
            neighbors = neighbors_list[i]
            neighbors.remove(i)
            no_contact[i] = neighbors

        return no_contact

