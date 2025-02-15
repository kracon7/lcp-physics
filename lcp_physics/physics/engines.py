"""
Author: Filipe de Avila Belbute Peres
Based on: M. B. Cline, Rigid body simulation with contact and constraints, 2002
"""

import torch

from lcp_physics.lcp.lcp import LCPFunction, LCPOptions


class PdipmEngine():
    """Engine that uses the primal dual interior point method LCP solver.
    """
    def __init__(self, max_iter, verbose, extend, solver_type):
        self.lcp_options = LCPOptions(max_iter=max_iter, 
                                      verbose=verbose, 
                                      extend=extend, 
                                      solver_type=solver_type)
        self.lcp_solver = LCPFunction
        self.cached_inverse = None
        self.max_iter = max_iter

    # @profile
    def solve_dynamics(self, world, dt):
        t = world.t
        Je = world.Je()
        neq = Je.size(0) if Je.ndimension() > 0 else 0
        nbody = len(world.bodies)

        f = world.apply_forces(t)
        u = torch.matmul(world.M(), world.get_v()) + dt * f
        if neq > 0:
            u = torch.cat([u, u.new_zeros(neq)])

        extend = self.lcp_options.extend

        if not extend and not world.contacts:
            # No contact constraints, no complementarity conditions
            if neq > 0:
                P = torch.cat([torch.cat([world.M(), -Je.t()], dim=1),
                               torch.cat([Je, Je.new_zeros(neq, neq)],
                                         dim=1)])
            else:
                P = world.M()
            if self.cached_inverse is None:
                inv = torch.inverse(P)
                if world.static_inverse:
                    self.cached_inverse = inv
            else:
                inv = self.cached_inverse
            x = torch.matmul(inv, u)  # Kline Eq. 2.41
        elif not extend and world.contacts:
            # Solve Mixed LCP (Kline 2.7.2)
            Jc = world.Jc()
            v = torch.matmul(Jc, world.get_v()) * world.restitutions()
            M = world.M().unsqueeze(0)
            if neq > 0:
                b = Je.new_zeros(Je.size(0)).unsqueeze(0)
                Je = Je.unsqueeze(0)
            else:
                b = torch.tensor([])
                Je = torch.tensor([])
            Jc = Jc.unsqueeze(0)
            u = u[:world.M().size(0)].unsqueeze(0)
            v = v.unsqueeze(0)
            E = world.E().unsqueeze(0)
            mu_s = world.mu_s().unsqueeze(0)
            Js = world.Js().unsqueeze(0)
            G = torch.cat([Jc, Js,
                           Js.new_zeros(Js.size(0), mu_s.size(1), Js.size(2))], dim=1)
            F = G.new_zeros(G.size(1), G.size(1)).unsqueeze(0)
            F[:, Jc.size(1):-E.size(2), -E.size(2):] = E
            F[:, -mu_s.size(1):, :mu_s.size(2)] = mu_s
            F[:, -mu_s.size(1):, mu_s.size(2):mu_s.size(2) + E.size(1)] = \
                -E.transpose(1, 2)
            h = torch.cat([v, v.new_zeros(v.size(0), Js.size(1) + mu_s.size(1))], 1)   # m in Eq.(2)
            x = -self.lcp_solver.apply(M, u, G, h, Je, b, F, self.lcp_options)
        elif extend:
            Jc = world.Jc()
            ncon = Jc.size(0)
            v = torch.matmul(Jc, world.get_v()) * world.restitutions()
            M = world.M().unsqueeze(0)
            if neq > 0:
                b = Je.new_zeros(Je.size(0)).unsqueeze(0)
                Je = Je.unsqueeze(0)
            else:
                b = torch.tensor([])
                Je = torch.tensor([])
            Jc = Jc.unsqueeze(0)
            u = u[:world.M().size(0)].unsqueeze(0)
            v = v.unsqueeze(0)
            E = world.E().unsqueeze(0)
            mu_s = world.mu_s().unsqueeze(0)
            mu_b = world.mu_b().unsqueeze(0)
            Js = world.Js().unsqueeze(0)
            Jb = world.Jb().unsqueeze(0)
            G = torch.cat([Jc, 
                           Js,
                           Js.new_zeros(Js.size(0), mu_s.size(1), Js.size(2)),
                           Jb,
                           Jb.new_zeros(M.size(0), 2*nbody, 3*nbody)
                        ], dim=1)
            F = G.new_zeros(G.size(1), G.size(1)).unsqueeze(0)
            F[:,   ncon:3*ncon, 3*ncon:4*ncon] = E
            F[:, 3*ncon:4*ncon,       :  ncon] = mu_s
            F[:, 3*ncon:4*ncon,   ncon:3*ncon] = -E.transpose(1, 2)
            F[:,         4*ncon:4*ncon+2*nbody, 4*ncon+2*nbody:4*ncon+4*nbody] = \
                                    torch.diag(G.new_ones(2*nbody)).unsqueeze(0)
            F[:, 4*ncon+2*nbody:4*ncon+4*nbody,         4*ncon:4*ncon+2*nbody] = \
                                   -torch.diag(G.new_ones(2*nbody)).unsqueeze(0)
            h = torch.cat([v, 
                           v.new_zeros(v.size(0), 3*ncon+2*nbody),
                           (world.mu_b_diag_M().unsqueeze(0))
                          ], dim=1)   # m in Eq.(2)

            x = -self.lcp_solver.apply(M, u, G, h, Je, b, F, self.lcp_options)
        new_v = x[:world.vec_len * len(world.bodies)].squeeze(0)
        return new_v

    def post_stabilization(self, world):
        v = world.get_v()
        M = world.M()
        Je = world.Je()
        Jc = None
        if world.contacts:
            Jc = world.Jc()
        ge = torch.matmul(Je, v)
        gc = None
        if Jc is not None:
            gc = torch.matmul(Jc, v) + torch.matmul(Jc, v) * -world.restitutions()

        u = torch.cat([Je.new_zeros(Je.size(1)), ge])
        if Jc is None:
            neq = Je.size(0) if Je.ndimension() > 0 else 0
            if neq > 0:
                P = torch.cat([torch.cat([M, -Je.t()], dim=1),
                               torch.cat([Je, Je.new_zeros(neq, neq)], dim=1)])
            else:
                P = M
            if self.cached_inverse is None:
                inv = torch.inverse(P)
            else:
                inv = self.cached_inverse
            x = torch.matmul(inv, u)
        else:
            v = gc
            Je = Je.unsqueeze(0)
            Jc = Jc.unsqueeze(0)
            h = u[:M.size(0)].unsqueeze(0)
            b = u[M.size(0):].unsqueeze(0)
            M = M.unsqueeze(0)
            v = v.unsqueeze(0)
            F = Jc.new_zeros(Jc.size(1), Jc.size(1)).unsqueeze(0)
            x = self.lcp_solver.apply(M, h, Jc, v, Je, b, F, self.lcp_options)
        dp = -x[:M.size(0)]
        return dp
