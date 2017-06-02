"""
solve a vector reaction-diffusion equation:

phi_t = kappa phi_{xx} + sigma R(phi)
where phi = [u, v, w]

using operator splitting, with implicit diffusion

B. Sarels based on code from M. Zingale
"""

import numpy as np
from scipy.integrate import ode
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt

class Grid(object):

    def __init__(self, nx, ng=1, xmin=0.0, xmax=1.0, vars=None):
        """ grid class initialization """

        self.nx = nx
        self.ng = ng

        self.xmin = xmin
        self.xmax = xmax

        self.dx = (xmax - xmin)/nx
        self.x  = (np.arange(nx+ng+1) + 0.5 - ng)*self.dx + xmin

        self.ilo = ng # intérêt ???
        self.ihi = ng+nx-1

        self.data = {}

        for v in vars:
            self.data[v] = np.zeros((3*(nx+ng+1)), dtype=np.float64)

    def scratch_array(self):
        return np.zeros((3*(self.nx+self.ng+1)), dtype=np.float64)

    def initialize(self):
        """ initial condition """

        phi = self.data["phi"]
        longueur1 = 10.
        longueur2 = 13.
        epsilon   = longueur2 - longueur1
        phi[0*self.ihi+0*self.ng+0 : 1*self.ihi+1*self.ng+1] = (self.x < 50.)
        phi[1*self.ihi+1*self.ng+1 : 2*self.ihi+2*self.ng+2] = 0.
        phi[2*self.ihi+2*self.ng+2 : 3*self.ihi+3*self.ng+3] = 0.

def frhs(t, phi, sigma):
    """ reaction ODE righthand side """

    u, v, w = phi
    D  = u * ( 1. - u - v - w ) - v * w

    dudt = u * ( - S + S * ( u + v ) * ( 3. - 2. * ( u + v ) ) + sB * ( u + w - 1. ) ) - r * D
    dvdt = v * ( - S + S * ( u + v ) * ( 3. - 2. * ( u + v ) ) + sB * ( u + w ) ) + r * D
    dwdt = w * ( S * ( u + v ) * ( 1. - 2. * ( u + v ) ) + sB * ( u + w - 1. ) ) + r * D

    return [dudt, dvdt, dwdt]

def jac(t, phi):

    return None

def react(gr, phi, sigma, dt):
    """ react phi through timestep dt """

    phinew = gr.scratch_array()

    for i in range(gr.ilo-1, gr.ihi+2):
        r = ode(frhs, jac).set_integrator("vode", method="adams", with_jacobian=False)
        phi0 = [phi[0*gr.ihi+0*gr.ng+0+i], phi[1*gr.ihi+1*gr.ng+1+i], phi[2*gr.ihi+2*gr.ng+2+i]]
        t0 = 0.
        r.set_initial_value(phi0, t0).set_f_params(sigma)
        r.integrate(r.t+dt)
        phinew[0*gr.ihi+0*gr.ng+0+i] = r.y[0]
        phinew[1*gr.ihi+1*gr.ng+1+i] = r.y[1]
        phinew[2*gr.ihi+2*gr.ng+2+i] = r.y[2]

    return phinew

def diffuse(gr, phi, kappa, dt):
    """ diffuse phi implicitly (Crank - Nicolson) through timestep dt """

    phinew = gr.scratch_array()

    alpha = kappa*dt/gr.dx**2

    # create the RHS of the matrix
    u = phi[0*gr.ihi+0*gr.ng+0 : 1*gr.ihi+1*gr.ng+1]
    v = phi[1*gr.ihi+1*gr.ng+1 : 2*gr.ihi+2*gr.ng+2]
    w = phi[2*gr.ihi+2*gr.ng+2 : 3*gr.ihi+3*gr.ng+3]
    diag     = -2.0 * np.eye(gr.nx+gr.ng+1)
    surdiag  = np.eye(gr.nx+gr.ng+1, k = 1)
    sousdiag = np.eye(gr.nx+gr.ng+1, k = -1)
    A  = diag + surdiag + sousdiag
    A[0, 1] = 2.0
    A[-1, -2] = 2.0
    I  = np.eye(gr.nx+gr.ng+1)
    TR = I + alpha / 2.0 * A # Tridiagonal matrix in the Right hand side
    R1 = TR @ u
    R2 = TR @ v
    R3 = TR @ w
    #TL = I - alpha / 2.0 * A # Tridiagonal matrix in the Left hand side
    # create the diagonal, upper and lower parts of the matrix
    D = (1.0 + alpha)*np.ones(gr.nx+gr.ng+1)
    U = -0.5*alpha*np.ones(gr.nx+gr.ng+1)
    L = -0.5*alpha*np.ones(gr.nx+gr.ng+1)
    U[0] = 0.0 # Valeur de toute façon inutile pour solve_banded
    L[-1] = 0.0 # Valeur de toute façon inutile pour solve_banded
    # set the boundary conditions by changing the matrix elements
    # homogeneous Neumann BC
    U[1] = -1.0*alpha
    L[-2] = -1.0*alpha
    BL = np.matrix([U, D, L]) # Banded matrix in the Left hand side

    # solve
    phinew[0*gr.ihi+0*gr.ng+0 : 1*gr.ihi+1*gr.ng+1] = solve_banded((1,1), BL, R1)
    phinew[1*gr.ihi+1*gr.ng+1 : 2*gr.ihi+2*gr.ng+2] = solve_banded((1,1), BL, R2)
    phinew[2*gr.ihi+2*gr.ng+2 : 3*gr.ihi+3*gr.ng+3] = solve_banded((1,1), BL, R3)

    return phinew

def est_dt(gr, kappa, sigma):
    """ estimate the timestep """

    # the speed is proportional to the square root of kappa * sigma
    s = np.sqrt(kappa * sigma)
    dt = gr.dx/s

    return dt

def evolve(nx, kappa, sigma, tmax, dovis=1, return_initial=0):
    """
    the main evolution loop.  Evolve

     phi_t = kappa phi_{xx} + sigma R(phi)

    from t = tmin to tmax
    """

    # create the grid
    gr = Grid(nx, ng = 1, xmin = 0.0, xmax = 100.0,
              vars = ["phi", "phi1", "phi2"])

    # pointers to the data at various stages
    phi  = gr.data["phi"]
    phi1 = gr.data["phi1"]
    phi2 = gr.data["phi2"]

    # initialize
    gr.initialize()

    if return_initial == 1:
        phi_init = phi.copy()

    t = tmin

    # runtime plotting
    if dovis == 1:
        u = phi[0*gr.ihi+0*gr.ng+0 : 1*gr.ihi+1*gr.ng+1]
        v = phi[1*gr.ihi+1*gr.ng+1 : 2*gr.ihi+2*gr.ng+2]
        w = phi[2*gr.ihi+2*gr.ng+2 : 3*gr.ihi+3*gr.ng+3]
        x = 1. - u - v - w
        figure1 = plt.figure(figsize=(12, 9), dpi=80)
        figure1.canvas.set_window_title('My title')
        plt.ion()
        plt.suptitle("Reaction-Diffusion, $t = {:3.2f}$".format(t))
        plt.subplot(2, 2, 1)
        plt.plot(gr.x, u, label='u')
        plt.plot(gr.x, v, label='v')
        plt.plot(gr.x, w, label='w')
        plt.plot(gr.x, x, label='x')
        plt.xlim(gr.xmin, gr.xmax)
        plt.ylim(0.0, 1.0)
        plt.grid(True)
        plt.legend(loc='best')

        plt.subplot(2, 2, 3)
        plt.plot(gr.x, w, label='w')
        plt.plot(gr.x, x, label='x')
        plt.xlim(gr.xmin, gr.xmax)
#        plt.ylim(0.0, 1.0)
        plt.grid(True)
        plt.legend(loc='best')

        plt.subplot(2, 2, 2)
        plt.plot(gr.x, w + x, label='p')
        plt.plot(gr.x, v + x, label='q')
        plt.plot(gr.x, u * x - v * w, label='D')
        plt.xlim(gr.xmin, gr.xmax)
        plt.ylim(0.0, 1.0)
        plt.grid(True)
        plt.legend(loc='best')
        plt.draw()
        plt.pause(0.1)

    while t < tmax:

        dt = est_dt(gr, kappa, sigma)

        if t + dt > tmax:
            dt = tmax - t

        # react for dt/2
        phi1[:] = react(gr, phi, sigma, dt/2)
        # diffuse for dt
        phi2[:] = diffuse(gr, phi1, kappa, dt)
        # react for dt/2 -- this is the updated solution
        phi[:] = react(gr, phi2, sigma, dt/2)

        t += dt

        # runtime plotting
        if dovis == 1:
            u = phi[0*gr.ihi+0*gr.ng+0 : 1*gr.ihi+1*gr.ng+1]
            v = phi[1*gr.ihi+1*gr.ng+1 : 2*gr.ihi+2*gr.ng+2]
            w = phi[2*gr.ihi+2*gr.ng+2 : 3*gr.ihi+3*gr.ng+3]
            p = w + x
            q = v + x
            dpdx = ( p[1 : ] - p[ : -1] ) / gr.dx
            dqdx = ( q[1 : ] - q[ : -1] ) / gr.dx

            x = 1. - u - v - w
            plt.clf()
            plt.suptitle("Reaction-Diffusion, $t = {:3.2f}$".format(t))
            plt.subplot(2, 2, 1)
            plt.plot(gr.x, u, label='u')
            plt.plot(gr.x, v, label='v')
            plt.xlim(gr.xmin, gr.xmax)
            plt.ylim(0.0, 1.0)
            plt.grid(True)
            plt.legend(loc='best')

            plt.subplot(2, 2, 3)
            plt.plot(gr.x, w, '--', label='w')
#            plt.plot(gr.x, x, label='x')
            plt.xlim(gr.xmin, gr.xmax)
#            plt.ylim(0.0, 1.0)
            plt.grid(True)
            plt.legend(loc='best')

            plt.subplot(2, 2, 2)
            plt.plot(gr.x, p, label='p')
            plt.plot(gr.x, q, label='q')
            plt.xlim(gr.xmin, gr.xmax)
            plt.ylim(0.0, 1.0)
            plt.grid(True)
            plt.legend(loc='best')

            plt.subplot(2, 2, 4)
            plt.plot(gr.x, u * x - v * w, label='D')
            plt.plot(gr.x[ : -1], 2 * kappa / r * dpdx * dqdx, label='estim')
            plt.xlim(gr.xmin, gr.xmax)
#            plt.ylim(0.0, 1.0)
            plt.grid(True)
            plt.legend(loc='best')

            plt.draw()
            plt.pause(0.1)

    if return_initial == 1:
        return phi, gr.x, phi_init
    else:
        return phi, gr.x

kappa = 0.025
S     = 0.05
sB    = 0.05
r     = 0.0001
sigma = 0.1 # intérêt ??? -> faire une estimation pour borner le terme de réaction

nx    = 256
tmin  = 0.
tmax1 = 1000.

phi1, x1 = evolve(nx, kappa, sigma, tmax1)
