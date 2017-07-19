"""
solve a vector reaction-diffusion equation:

phi_t = kappa phi_{xx} + frhs(phi)
where phi = [u, v, w]

using operator splitting, with implicit diffusion

This is the DRIVER program.

B. Sarels
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

from STRANGcore import evolve

# Parameters for diffusion
kappa = 0.025

# Parameters for reaction
# They should be put in the function itself
def frhs(t, phi):
    """ reaction ODE righthand side """

    u, v, w = phi
    D  = u * ( 1. - u - v - w ) - v * w

    S  = 0.05
    sB = 0.05
    r0 = 0.0001

    dudt = u * ( - S + S * ( u + v ) * ( 3. - 2. * ( u + v ) ) + sB * ( u + w - 1. ) ) - r0 * D
    dvdt = v * ( - S + S * ( u + v ) * ( 3. - 2. * ( u + v ) ) + sB * ( u + w ) ) + r0 * D
    dwdt = w * ( S * ( u + v ) * ( 1. - 2. * ( u + v ) ) + sB * ( u + w - 1. ) ) + r0 * D

    return [dudt, dvdt, dwdt]

# Parameters for the problem
nx   = 198
xmin = 0.
xmax = 100.
tmin = 0.
tmax = 400.

phi, x, T, ng = evolve(nx, xmin, xmax, tmin, tmax, kappa, frhs)

u = phi[ : , 0*nx + 0*ng + 0 : 1*nx + 1*ng + 1]
v = phi[ : , 1*nx + 1*ng + 1 : 2*nx + 2*ng + 2]
w = phi[ : , 2*nx + 2*ng + 2 : 3*nx + 3*ng + 3]
z = 1. - u - v - w # attention aux notations z est la 4eme variable
p = w + z
q = v + z
D = u * z - v * w
dx = (xmax - xmin)/nx # mettre ça ailleurs : dans la sortie de evolve ?
dpdx = np.gradient(p, dx)
dqdx = np.gradient(q, dx)

# Parameters for the visualisation
ymin = 0.
ymax = 1.1

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax  = plt.axes(xlim = (xmin, xmax), ylim = (ymin, ymax))
linep, = ax.plot([], [], lw = 2)
lineq, = ax.plot([], [], lw = 2)
lineD, = ax.plot([], [], lw = 2)
time_text  = ax.text(0.02, 0.95, '', transform = ax.transAxes)

def indice():
    """ indice function. It will give me a nice iterable """
    for i in range(T.size):
        yield i

def animate(i):
    """ animate function. It is called following the indice iterable """
    linep.set_data(x, p[i, : ])
    lineq.set_data(x, q[i, : ])
    lineD.set_data(x, D[i, : ])
    time_text.set_text('time = %.1f' % T[i])
    ax.set_title('time = %.1f' % T[i])
#    return linep, lineq, lineD, time_text

# call the animator. 'blit = True' means only re-draw the parts that have changed.
# but pay attention! If using 'blit = True', then animate has to have a return statement and then I cannot have the title to change...
courbe = animation.FuncAnimation(fig, animate, indice, interval = 200)

# save the animation as a gif/mp4
#courbe.save('genet_pop.gif', fps = 30, writer = 'imagemagick')
#courbe.save('genet_pop.mp4', fps = 30, extra_args = ['-vcodec', 'libx264'])

plt.show()
