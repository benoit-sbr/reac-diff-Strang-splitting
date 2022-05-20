"""
solve a vector reaction-diffusion equation:

phi_t = kappa phi_{xx} + frhs(phi)
where phi = [y1, y2, y3]

using operator splitting, with implicit diffusion

This is the DRIVER program.

B. Sarels
"""
# Standard imports
import os
#import time

# Scientific imports
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from constants import xp0, xq0
from constants import kappa, reac_params
from constants import nx, xmin, xmax, space_bounds, time_bounds
from functions_animation import animate, init_my_axes
from functions import frhs
from STRANGcore import evolve
from theory import cp0, cp1

# Necessary runtime configuration if in jupyterlab
#plt.rcParams['animation.html'] = 'jshtml'

sA, SA, sB, SB, rAB = reac_params

# The main computation is done in the following line
print('Run the main computation...')
phi, T, grille = evolve(nx, 3, xmin, xmax, time_bounds, xp0, xq0, kappa, reac_params, frhs)
print('Done!')
y1 = phi[ : , 0*phi.shape[1]//3 : 1*phi.shape[1]//3 ]
y2 = phi[ : , 1*phi.shape[1]//3 : 2*phi.shape[1]//3 ]
y3 = phi[ : , 2*phi.shape[1]//3 : ]
y4 = 1. - y1 - y2 - y3
p = y1 + y2
q = y1 + y3
D = y1 * y4 - y2 * y3
#dpdx = np.gradient(p, grille.dx)
#dqdx = np.gradient(q, grille.dx)

def kink(x, centre):
    return 1./(1.+np.exp(np.sqrt(SA)*(x-centre)))

# The fit to the known kink solutions is done in the following block
centres = np.zeros((phi.shape[0], 2))
for n in range(phi.shape[0]): # fit at each time step
    # initial guess
    centrep = centres[n-1, 0]
    centreq = centres[n-1, 1]
    # call to curve_fit
    poptp, pcovp = curve_fit(kink, grille.x, p[n , : ], p0 = centrep)
    poptq, pcovq = curve_fit(kink, grille.x, q[n , : ], p0 = centreq)
    # write to centres
    centres[n, 0] = poptp
    centres[n, 1] = poptq

dcentrespdt = np.gradient(centres[ : , 0], T[1]-T[0])
dcentresqdt = np.gradient(centres[ : , 1], T[1]-T[0])

# Parameters for the visualisation
petit = (dcentrespdt[2 : ].max() - dcentresqdt[2 : ].min())/20.
ymin0 = p.min() - petit
ymax0 = p.max() + petit
ymin1 = cp0(sA, SA) - petit
ymax1 = cp1(sA, SA, rAB) + petit

# Create a figure
figure1, axes = plt.subplots(1, 2, dpi = 80, figsize = (16, 12), sharex = 'col')
# Initialize the line objects
linep,  = axes[0].plot([], [], lw = 2, label = 'p')
lineq,  = axes[0].plot([], [], lw = 2, label = 'q')
lineD,  = axes[0].plot([], [], lw = 2, label = 'D')
linevp, = axes[1].plot([], [], lw = 2, label = 'speed of p')
color1p  = linevp.get_color()
axes[1].hlines(cp0(sA, SA), *time_bounds, lw = 1, colors = color1p, linestyles = 'dashed')
axes[1].hlines(cp1(sA, SA, rAB), *time_bounds, lw = 1, colors = color1p, linestyles = 'dashed')
linevq, = axes[1].plot([], [], lw = 2, label = 'speed of q')
lines = (linep, lineq, lineD, linevp, linevq)
# Initialize the axes
init_my_axes(axes, lines, space_bounds, time_bounds, (ymin0, ymax0), (ymin1, ymax1))

def init():
    """Initialization function: plot the background of each frame.

    Arguments:
    none
    """
    linep.set_data([], [])
    lineq.set_data([], [])
    lineD.set_data([], [])
    return linep, lineq, lineD

# Create animation
curve = animation.FuncAnimation(figure1, animate, frames = T.size, #init_func = init,
        fargs = (grille, p, q, D, dcentrespdt, dcentresqdt, T, axes, figure1, lines),
        repeat = False)
# Show animation
#curve         # if in jupyterlab
figure1.show()  # if in ipython
#plt.close(figure1) # necessary?

# in case of closer initial conditions
#FileNameRoot = 'sA'+str(sA)+'SA'+str(SA)+'sB'+str(sB)+'SB'+str(SB)+'rAB'+str(rAB)+'closerIC'
# in case of stacked initial conditions
FileNameRoot = 'sA'+str(sA)+'SA'+str(SA)+'sB'+str(sB)+'SB'+str(SB)+'rAB'+str(rAB)+'stackedIC'
FolderName = 'data/data' + FileNameRoot + '/'
if not os.path.exists(FolderName):
    os.makedirs(FolderName)

for n in range(0, 1, 3):
#for n in range(0, phi.shape[0], 3):
    figure2 = plt.figure()
    plt.title('Allele frequencies at time $t = {:3.1f}$'.format(T[n]))
    plt.grid(True)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin0, ymax0)
#    plt.xlabel('Space')
#    plt.ylabel('')
    plt.plot(grille.x, p[n, : ], label = 'p')
    plt.plot(grille.x, q[n, : ], label = 'q')
    plt.plot(grille.x, D[n, : ], label = 'D')
    plt.legend(loc = 'best')
#    figure2.show()
#    plt.pause(0.1)
    FileNameTimestepN = FileNameRoot+'timestep'+str(n)+'.png'
    plt.savefig(FolderName+FileNameTimestepN, bbox_inches='tight')
    plt.close(figure2)

figure3 = plt.figure(figsize = (12, 8), dpi = 80)
plt.title('Instantaneous wave speed')
line3vp, = plt.plot(T[2 : ], dcentrespdt[2 : ], label = 'speed of p')
color3p = line3vp.get_color()
plt.hlines(cp0(sA, SA), *time_bounds, lw = 1, colors = color3p, linestyles = 'dashed')
plt.hlines(cp1(sA, SA, rAB), *time_bounds, lw = 1, colors = color3p, linestyles = 'dashed')
plt.plot(T[2 : ], dcentresqdt[2 : ], label = 'speed of q')
plt.grid(True)
plt.yticks([i/500.0 for i in range(0, 50)])
plt.xlim(time_bounds)
plt.ylim(ymin1, ymax1)
plt.xlabel('Time')
plt.ylabel('Wave speed')
plt.legend(loc = 'best')
figure3.show()
FileNameSpeeds = FileNameRoot+'speeds.png'
plt.savefig(FolderName+FileNameSpeeds, bbox_inches='tight')
plt.close(figure3)


# Save the animation as a gif file
print('Save the animation...')
curve.save(FolderName+FileNameRoot+'.mp4')#, writer = 'imagemagick')
print('Done!')

# Save the data as a npy file
print('Save the data...')
np.save(FolderName+FileNameRoot+'.npy', phi)
print('Done!')
