"""
solve a vector reaction-diffusion equation:

phi_t = kappa phi_{xx} + frhs(phi)
where phi = [u1, u2, ..., uk, ..., uN]

using operator splitting, with implicit diffusion

This is the CORE program.

B. Sarels based on code from M. Zingale
"""

import numpy as np
from scipy.integrate import ode
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.linalg import solve_banded

class Grid(object):

    def __init__(self, nx, systemsize, xmin, xmax, ng = 1, vars = None):
        """ grid class initialization """

        self.nx = nx	# number of interior points
        self.systemsize = systemsize
        self.ng = ng	# number of points to handle a boundary, 1 in 1D

        self.xmin = xmin
        self.xmax = xmax

        self.dx = (xmax - xmin) / (nx + ng)
        self.x  = np.arange(ng + nx + ng) * self.dx + xmin

        self.ilo = ng - 1
        self.ihi = ng + nx

        self.data = {}	# data est un dictionnaire

        for v in vars:
            self.data[v] = np.zeros((systemsize*(ng + nx + ng)), dtype = np.float64)

    def scratch_array(self):
        return np.zeros((self.systemsize*(self.ihi + self.ng)), dtype = np.float64)

    def initialize(self):
        """ initial condition """

        phi = self.data["phi"]
# provide an initial condition such that y1>=0, y2>=0, y3>=0 and y1+y2+y3<=1
        SA = 0.1
        SB = 0.1
        p = 1./(1.+np.exp(np.sqrt(SA)*(self.x-35.)))
        q = 1./(1.+np.exp(np.sqrt(SB)*(self.x-45.)))
        phi[0 * (self.ihi + self.ng) : 1 * (self.ihi + self.ng)] = p     *q
        phi[1 * (self.ihi + self.ng) : 2 * (self.ihi + self.ng)] = p     *(1.-q)
        phi[2 * (self.ihi + self.ng) : 3 * (self.ihi + self.ng)] = (1.-p)*q

def react(gr, phi, systemsize, frhs, dt):
    """ react phi through timestep dt """

    phinew = gr.scratch_array()

    for i in range(gr.ilo, gr.ihi + gr.ng):
#        r = ode(frhs, jac)
        r = ode(frhs)
        phi0 = phi[i : : (gr.ihi + gr.ng)]
        t0 = 0.
        r.set_initial_value(phi0, t0)
# “vode”: real-valued Variable-coefficient Ordinary Differential Equation solver,
#         with fixed-leading-coefficient implementation.
#         It provides
# - implicit Adams method (for non-stiff problems) (method = "adams")
# - a method based on backward differentiation formulas (for stiff problems) (method = "bdf")
        r.set_integrator("vode", method = "adams")
        r.set_f_params(gr.x[i])
        r.integrate(r.t + dt)
        for k in range(0, systemsize):
            phinew[k * (gr.ihi + gr.ng) + i] = r.y[k]

    return phinew

def diffuse(gr, phi, systemsize, kappa, dt):
    """ diffuse phi implicitly (Crank - Nicolson) through timestep dt """

    phinew = gr.scratch_array()

    alpha = kappa*dt/gr.dx**2

    u = np.zeros((systemsize, (gr.ihi + gr.ng)))
    R = np.zeros((systemsize, (gr.ihi + gr.ng)))
    diag     = -2.0 * np.eye(gr.ihi + gr.ng)
    surdiag  = np.eye(gr.ihi + gr.ng, k = 1)
    sousdiag = np.eye(gr.ihi + gr.ng, k = -1)
    A  = diag + surdiag + sousdiag
    # set the boundary conditions by changing the matrix elements
    # homogeneous Neumann BC
    A[0, 1] = 2.0
    A[-1, -2] = 2.0
    I  = np.eye(gr.ihi + gr.ng)

    for k in range(0, systemsize):
        # 1. define right hand side
        u[k, : ] = phi[k * (gr.ihi + gr.ng) : (k+1) * (gr.ihi + gr.ng)]
        TR = I + alpha[k] / 2.0 * A # Tridiagonal matrix in the right hand side
        R[k, : ] = TR @ u[k, : ]

        # 2. define left hand side
        #TL = I - alpha[k] / 2.0 * A # Tridiagonal matrix in the left hand side
        # create the diagonal, upper and lower parts of the matrix
        D = (1.0 + alpha[k]) * np.ones(gr.ihi + gr.ng)
        U = -0.5 * alpha[k]  * np.ones(gr.ihi + gr.ng)
        L = -0.5 * alpha[k]  * np.ones(gr.ihi + gr.ng)
        U[0]  = 0.0 # Valeur de toute façon inutile pour solve_banded
        L[-1] = 0.0 # Valeur de toute façon inutile pour solve_banded
        # set the boundary conditions by changing the matrix elements
        # homogeneous Neumann BC
        U[1]  = -1.0*alpha[k]
        L[-2] = -1.0*alpha[k]
        BL = np.matrix([U, D, L]) # Banded matrix in the left hand side

        # 3. solve
        phinew[k * (gr.ihi + gr.ng) : (k+1) * (gr.ihi + gr.ng)] = solve_banded((1,1), BL, R[k, : ])

    return phinew

def est_dt(gr, kappa, sigma = 1.):
    """ estimate the timestep """
# In the old context of solving phi_t = kappa phi_{xx} + sigma R(phi)
# the speed is proportional to the square root of (kappa * sigma)
    kappamin = kappa[kappa > 0.].min()
    s  = np.sqrt(kappamin * sigma)
    dt = gr.dx/s

    return dt

def evolve(nx, systemsize, xmin, xmax, tmin, tmax, kappa, frhs):
    """
    the main evolution loop.  Evolve

    phi_t = kappa phi_{xx} + frhs(phi)

    from t = tmin to tmax
    """

    # create the grid
    gr = Grid(nx, systemsize, xmin, xmax, vars = ["phi", "phi1", "phi2"])

    # create the time vector
    dt = est_dt(gr, kappa, 2.)
    T  = np.arange(tmin, tmax, dt) # note that more often than not, tmax is not a point in T

    # pointers to the data at various stages
    phi  = gr.data["phi"]
    phi1 = gr.data["phi1"]
    phi2 = gr.data["phi2"]

    sol  = np.zeros((T.size, systemsize*(gr.ihi + gr.ng)), dtype = np.float64)

    # initialize
    gr.initialize()
    sol[0] = phi

    for n in range(1, T.size):
        # react for dt/2
        phi1[:] = react(gr, phi, systemsize, frhs, dt/2)
        # diffuse for dt
        phi2[:] = diffuse(gr, phi1, systemsize, kappa, dt)
        # react for dt/2 -- this is the updated solution
        phi[:] = react(gr, phi2, systemsize, frhs, dt/2)

        # debug
        y1 = phi[0 * (gr.ihi + gr.ng) : 1 * (gr.ihi + gr.ng)]
        y2 = phi[1 * (gr.ihi + gr.ng) : 2 * (gr.ihi + gr.ng)]
#        print (T[n], y1.max(), y2.max())
#        print (n, T[n])

        # incident front at t = 5
        if n == int(5/dt):
            # record incident front
            y1ref = phi[0 * (gr.ihi + gr.ng) : 1 * (gr.ihi + gr.ng)]
            # shift incident front
            i = 0
            while y1ref[i] > 0.5:
                i += 1
            imilieu = i
            # shift incident front - autre méthode
            sorter = np.arange(y1ref.size)[::-1]
            imilieu2 = y1ref.size - np.searchsorted(y1ref, 0.5, sorter = sorter)
            imilieu3 = y1ref.size - np.searchsorted(y1ref[::-1], 0.5)
            x0pourx1ref = imilieu*gr.dx + xmin
            x1ref = gr.x - x0pourx1ref
#            print (n, T[n])
#            print (y1ref)
#            print ('imilieu', imilieu, y1ref[imilieu-1], y1ref[imilieu])
#            print ('imilieu2', imilieu2)
#            print ('imilieu3', imilieu3)
#            print (x0pourx1ref)
#            print (x1ref)
            # positions to inter/extrapolate
            x = np.linspace(-100., 100., 5000)
            # spline order: 1 linear, 2 quadratic, 3 cubic ... 
            order = 1
            # do inter/extrapolation
            s = InterpolatedUnivariateSpline(x1ref, y1ref, k = order)
            y = s(x)

            # example showing the interpolation for linear, quadratic and cubic interpolation
#            plt.figure()
#            plt.grid(True)
#            plt.plot(x1ref, y1ref, linestyle=':')
#            for order in range(1, 4):
#                s = InterpolatedUnivariateSpline(x1ref, y1ref, k = order)
#                y = s(x)
#                plt.plot(x, y)
#            plt.show()

        # write to sol
        sol[n] = phi

    return sol, T, gr
