"""
solve a vector reaction-diffusion equation:

phi_t = kappa phi_{xx} + frhs(phi)
where phi = [u, v, w]

using operator splitting, with implicit diffusion

This is the CORE program.

B. Sarels based on code from M. Zingale
"""

import numpy as np
from scipy.integrate import ode
from scipy.linalg import solve_banded

class Grid(object):

    def __init__(self, nx, xmin, xmax, ng = 1, vars = None):
        """ grid class initialization """

        self.nx = nx
        self.ng = ng

        self.xmin = xmin
        self.xmax = xmax

        self.dx = (xmax - xmin) / (nx + ng)
        self.x  = (np.arange(ng + nx + ng)) * self.dx + xmin

        self.ilo = ng - 1
        self.ihi = ng + nx

        self.data = {}

        for v in vars:
            self.data[v] = np.zeros((3*(ng + nx + ng)), dtype = np.float64)

    def scratch_array(self):
        return np.zeros((3*(self.ihi + self.ng)), dtype = np.float64)

    def initialize(self):
        """ initial condition """

        phi = self.data["phi"]
        phi[0*self.ihi + 0*self.ng : 1*self.ihi + 1*self.ng] = (self.x < 50.)
        phi[1*self.ihi + 1*self.ng : 2*self.ihi + 2*self.ng] = 0.
        phi[2*self.ihi + 2*self.ng : 3*self.ihi + 3*self.ng] = 0.

def react(gr, phi, frhs, dt):
    """ react phi through timestep dt """

    phinew = gr.scratch_array()

    for i in range(gr.ilo, gr.ihi + 1):
#        r = ode(frhs, jac)
        r = ode(frhs)
        phi0 = [phi[0*gr.ihi + 0*gr.ng + i], phi[1*gr.ihi + 1*gr.ng + i], phi[2*gr.ihi + 2*gr.ng + i]]
        t0 = 0.
        r.set_initial_value(phi0, t0)
# “vode”: real-valued Variable-coefficient Ordinary Differential Equation solver,
#         with fixed-leading-coefficient implementation.
#         It provides
# - implicit Adams method (for non-stiff problems) (method = "adams")
# - a method based on backward differentiation formulas (for stiff problems) (method = "bdf")
        r.set_integrator("vode", method = "adams")
        r.set_f_params()
        r.integrate(r.t + dt)
        phinew[0*gr.ihi + 0*gr.ng + i] = r.y[0]
        phinew[1*gr.ihi + 1*gr.ng + i] = r.y[1]
        phinew[2*gr.ihi + 2*gr.ng + i] = r.y[2]

    return phinew

def diffuse(gr, phi, kappa, dt):
    """ diffuse phi implicitly (Crank - Nicolson) through timestep dt """

    phinew = gr.scratch_array()

    alpha = kappa*dt/gr.dx**2

    # create the RHS of the matrix
    u = phi[0*gr.ihi + 0*gr.ng : 1*gr.ihi + 1*gr.ng]
    v = phi[1*gr.ihi + 1*gr.ng : 2*gr.ihi + 2*gr.ng]
    w = phi[2*gr.ihi + 2*gr.ng : 3*gr.ihi + 3*gr.ng]
    diag     = -2.0 * np.eye(gr.ihi + gr.ng)
    surdiag  = np.eye(gr.ihi + gr.ng, k = 1)
    sousdiag = np.eye(gr.ihi + gr.ng, k = -1)
    A  = diag + surdiag + sousdiag
    A[0, 1] = 2.0
    A[-1, -2] = 2.0
    I  = np.eye(gr.ihi + gr.ng)
    TR = I + alpha / 2.0 * A # Tridiagonal matrix in the Right hand side
    R1 = TR @ u
    R2 = TR @ v
    R3 = TR @ w
    #TL = I - alpha / 2.0 * A # Tridiagonal matrix in the Left hand side
    # create the diagonal, upper and lower parts of the matrix
    D = (1.0 + alpha) * np.ones(gr.ihi + gr.ng)
    U = -0.5 * alpha  * np.ones(gr.ihi + gr.ng)
    L = -0.5 * alpha  * np.ones(gr.ihi + gr.ng)
    U[0]  = 0.0 # Valeur de toute façon inutile pour solve_banded
    L[-1] = 0.0 # Valeur de toute façon inutile pour solve_banded
    # set the boundary conditions by changing the matrix elements
    # homogeneous Neumann BC
    U[1]  = -1.0*alpha
    L[-2] = -1.0*alpha
    BL = np.matrix([U, D, L]) # Banded matrix in the Left hand side

    # solve
    phinew[0*gr.ihi + 0*gr.ng : 1*gr.ihi + 1*gr.ng] = solve_banded((1,1), BL, R1)
    phinew[1*gr.ihi + 1*gr.ng : 2*gr.ihi + 2*gr.ng] = solve_banded((1,1), BL, R2)
    phinew[2*gr.ihi + 2*gr.ng : 3*gr.ihi + 3*gr.ng] = solve_banded((1,1), BL, R3)
    return phinew

def est_dt(gr, kappa, sigma = 1.):
    """ estimate the timestep """
# In the old context of solving phi_t = kappa phi_{xx} + sigma R(phi)
# the speed is proportional to the square root of kappa * sigma
    s = np.sqrt(kappa * sigma)
    dt = gr.dx/s

    return dt

def evolve(nx, xmin, xmax, tmin, tmax, kappa, frhs):
    """
    the main evolution loop.  Evolve

     phi_t = kappa phi_{xx} + frhs(phi)

    from t = tmin to tmax
    """

    # create the grid
    gr = Grid(nx, xmin, xmax, vars = ["phi", "phi1", "phi2"])

    # create the time vector
    dt = est_dt(gr, kappa)
    T  = np.arange(tmin, tmax, dt)

    # pointers to the data at various stages
    phi  = gr.data["phi"]
    phi1 = gr.data["phi1"]
    phi2 = gr.data["phi2"]

    sol  = np.zeros((T.size, 3*(gr.ihi + gr.ng)), dtype = np.float64)

    # initialize
    gr.initialize()
    sol[0] = phi

    for n in range(1, T.size):

        # react for dt/2
        phi1[:] = react(gr, phi, frhs, dt/2)
        # diffuse for dt
        phi2[:] = diffuse(gr, phi1, kappa, dt)
        # react for dt/2 -- this is the updated solution
        phi[:] = react(gr, phi2, frhs, dt/2)

        sol[n] = phi

    else:
        return sol, gr.x, T, gr.ng
