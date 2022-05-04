import torch
from torch.autograd import Function

from .solvers import pdipm
from .solvers.pdipm import KKTSolvers
from .util import bger, extract_batch_size


class LCPOptions():
    def __init__(self, eps=1e-12, verbose=-1, not_improved_lim=3,
                 max_iter=10, extend=0, solver_type=1):
        self.eps = eps
        self.verbose = verbose
        self.not_improved_lim = not_improved_lim
        self.max_iter = max_iter
        self.extend = extend
        self.solver = KKTSolvers(solver_type)

class LCPFunction(Function):
    # """A differentiable LCP solver, uses the primal dual interior point method
    #    implemented in pdipm.
    # """
    # def __init__(self, eps=1e-12, verbose=-1, not_improved_lim=3,
    #              max_iter=10):
    #     super().__init__()
    #     self.eps = eps
    #     self.verbose = verbose
    #     self.not_improved_lim = not_improved_lim
    #     self.max_iter = max_iter
    #     self.Q_LU = self.S_LU = self.R = None

    @staticmethod
    def forward(ctx, Q, p, G, h, A, b, F, lcp_options):
        _, nineq, nz = G.size()
        neq = A.size(1) if A.ndimension() > 1 else 0
        assert(neq > 0 or nineq > 0)
        ctx.neq, ctx.nineq, ctx.nz = neq, nineq, nz

        ctx.Q_LU, ctx.S_LU, ctx.R = pdipm.pre_factor_kkt(Q, G, F, A)
        zhats, nus, ctx.lams, ctx.slacks = pdipm.forward(
            Q, p, G, h, A, b, F, ctx.Q_LU, ctx.S_LU, ctx.R,
            eps=lcp_options.eps, max_iter=lcp_options.max_iter, verbose=lcp_options.verbose,
            not_improved_lim=lcp_options.not_improved_lim, solver=lcp_options.solver)

        ctx.save_for_backward(zhats, Q, p, G, h, A, b, F)
        return zhats, nus

    # @staticmethod
    # def backward(ctx, dl_dzhat):
    #     zhats, Q, p, G, h, A, b, F = ctx.saved_tensors
    #     batch_size = extract_batch_size(Q, p, G, h, A, b)

    #     neq, nineq, nz = ctx.neq, ctx.nineq, ctx.nz

    #     # D = torch.diag((ctx.lams / ctx.slacks).squeeze(0)).unsqueeze(0)
    #     d = ctx.lams / ctx.slacks

    #     pdipm.factor_kkt(ctx.S_LU, ctx.R, d)
    #     dx, _, dlam, dnu = pdipm.solve_kkt(ctx.Q_LU, d, G, A, ctx.S_LU,
    #                                        dl_dzhat, G.new_zeros(batch_size, nineq),
    #                                        G.new_zeros(batch_size, nineq),
    #                                        G.new_zeros(batch_size, neq))

    #     dps = dx
    #     dGs = (bger(dlam, zhats) + bger(ctx.lams, dx))
    #     dFs = -bger(dlam, ctx.lams)
    #     dhs = -dlam
    #     if neq > 0:
    #         dAs = bger(dnu, zhats) + bger(ctx.nus, dx)
    #         dbs = -dnu
    #     else:
    #         dAs, dbs = None, None
    #     dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))

    #     return dQs, dps, dGs, dhs, dAs, dbs, dFs, None

