'''
Hidden state forward and backward mapping
'''

import os
import sys
import numpy as np
import sympy as sp

class HiddenStateMapping:
    def __init__(self, sim):
        self.sim = sim
        
        # number of particles
        self.n_particle = sim.num_particle
        
        # particle coordinate in conanical world frame
        self.particle_coord = self.sim.block_object.particle_coord

        # gravity
        self.gravity = 9.8

    def validate_mapping(self, mass_mapping, friction_mapping):
        friction_keys = set(friction_mapping)
        fric2mass, seen = {}, set()
        for k in friction_keys:
            fric2mass[k] = set(mass_mapping[friction_mapping == k])
            if len(fric2mass[k].intersection(seen)) > 0:
                raise Exception("Mapping invalid for friction mapping key %d"%k)
            seen = seen.union(fric2mass[k])

        return True
        
    def map_to_hidden_state(self, composite_mass, mass_mapping, composite_friction, 
                            friction_mapping):
        '''
        From particle mass/friction to hidden states
        '''
        geom_mass, geom_friction = np.zeros(self.n_particle), np.zeros(self.n_particle)
        for i in range(self.n_particle):
            geom_mass[i] = composite_mass[mass_mapping[i]]
            geom_friction[i] = composite_friction[friction_mapping[i]]

        geom_pos = self.particle_coord
        body_mass = np.sum(geom_mass)
        body_com = np.zeros(2)
        for i in range(self.n_particle):
            body_com += geom_pos[i] * geom_mass[i] / body_mass

        body_inertia = 0.
        for i in range(self.n_particle):
            body_inertia += geom_mass[i] * np.linalg.norm(geom_pos[i] - body_com)**2

        # Compute si_mapping from mass_mapping and friction_mapping
        if self.validate_mapping(mass_mapping, friction_mapping):
            si_mapping = mass_mapping

        composite_si = np.zeros(self.n_particle)
        n = len(set(mass_mapping))
        for i in range(n):
            j = np.where(mass_mapping==i)[0][0]
            composite_si[i] = composite_mass[i] * geom_friction[j] * self.gravity
        
        hidden_state = {"body_mass": body_mass,
                        "body_inertia": body_inertia,
                        "body_com": body_com,
                        "composite_si": composite_si,
                        "si_mapping": si_mapping}
        return hidden_state

    def map_to_explicit_state(self, hidden_state, mass_mapping, friction_mapping):
        geom_pos = self.particle_coord

        # Get mass to friction mapping
        if not self.validate_mapping(mass_mapping, friction_mapping):
            return None

        mass2fric = {}
        for k in set(mass_mapping):
            # The first index where mass_mass has k
            i = np.where(mass_mapping == k)[0][0]
            mass2fric[k] = friction_mapping[i]

        n_mass, n_mu = len(set(mass_mapping)), len(set(friction_mapping))

        # mass of each particle in i-th mass partition
        m = sp.symbols('m:%d'%n_mass)
        # friction coefficient in i-th friction partition
        mu = sp.symbols('mu:%d'%n_mu)

        # expressions for total mass, center of mass and moment of inertia
        M = 0
        for i in range(self.n_particle):
            M = M + m[mass_mapping[i]]

        cx, cy = 0, 0
        for i in range(self.n_particle):
            cx = cx + geom_pos[i, 0] * m[mass_mapping[i]] / M
            cy = cy + geom_pos[i, 1] * m[mass_mapping[i]] / M

        I = 0
        for i in range(self.n_particle):
            I = I + m[mass_mapping[i]] * ((geom_pos[i, 0] - cx)**2 + 
                                          (geom_pos[i, 1] - cy)**2)
        I = sp.simplify(I)                                          

        si = []
        for i in range(n_mass):
            si.append(self.gravity * m[i] * mu[mass2fric[i]])

        equations = []
        equations.append(sp.Eq(M, hidden_state["body_mass"]))  
        equations.append(sp.Eq(cx, hidden_state["body_com"][0]))
        equations.append(sp.Eq(cy, hidden_state["body_com"][1]))
        equations.append(sp.Eq(I, hidden_state["body_inertia"]))
        for i in range(n_mass):
            equations.append(sp.Eq(si[i], hidden_state["composite_si"][i]))
        res = sp.solve(equations, m+mu, dict=True)

        composite_mass = np.zeros(self.n_particle)
        composite_friction = np.zeros(self.n_particle)
        for i in range(n_mass):
            composite_mass[i] = res[0][m[i]]
        for i in range(n_mu):
            composite_friction[i] = res[0][mu[i]]

        explicit_state = {"composite_mass": composite_mass,
                          "composite_friction": composite_friction}
        return explicit_state