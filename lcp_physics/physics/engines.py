"""
Author: Filipe de Avila Belbute Peres
Based on: M. B. Cline, Rigid body simulation with contact and constraints, 2002
"""

import torch

from lcp_physics.lcp.lcp import LCPFunction


class Engine:
    """Base class for stepping engine."""
    def solve_dynamics(self, world, dt):
        raise NotImplementedError


class PdipmEngine(Engine):
    """Engine that uses the primal dual interior point method LCP solver.
    """
    def __init__(self, max_iter=10):
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
        if not world.contacts:
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
        else:
            # Solve Mixed LCP (Kline 2.7.2)
            Jc = world.Jc()
            ncon = Jc.size(0)   # number of contacts
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
            G = torch.cat([Jc, 
                           Js,
                           Js.new_zeros(Js.size(0), ncon, 3*nbody)
                        ], dim=1)
            
            # G = torch.cat([Jc, 
            #                Js,
            #                Js.new_zeros(Js.size(0), mu_s.size(1), Js.size(2)),
            #                Jb,
            #                Jb.new_zeros(M.size(0), M.size(1), M.size(2))
            #             ], dim=1)
            F = G.new_zeros(G.size(1), G.size(1)).unsqueeze(0)
            F[:,   ncon:3*ncon, 3*ncon:4*ncon] = E
            F[:, 3*ncon:4*ncon,       :  ncon] = mu_s
            F[:, 3*ncon:4*ncon,   ncon:3*ncon] = -E.transpose(1, 2)
            h = torch.cat([v, v.new_zeros(v.size(0), 3*ncon)], 1)   # m in Eq.(2)

            x = -self.lcp_solver(max_iter=self.max_iter, verbose=-1)(M, u, G, h, Je, b, F)
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
            x = self.lcp_solver()(M, h, Jc, v, Je, b, F)
        dp = -x[:M.size(0)]
        return dp
