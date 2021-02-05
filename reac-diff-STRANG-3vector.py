"""
solve a vector reaction-diffusion equation:

phi_t = kappa phi_{xx} + frhs(phi)
where phi = [y1, y2, y3]

using operator splitting, with implicit diffusion

This is the DRIVER program.

B. Sarels
"""

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
#import time

from STRANGcore import evolve

# Parameters for diffusion
kappa = np.array([1., 1., 1.])

# Parameters for reaction
sA = 0.
SA = 0.1
cp = sA/np.sqrt(SA)
sB = 0.
SB = 0.1
rAB = 0.1
# Should they be put in the function itself?
def frhs(t, phi, x):
    """ reaction ODE righthand side """

    y1, y2, y3 = phi
    y4 = 1. - y1 - y2 - y3
    D  = y1 * y4 - y2 * y3

    dy1dt  = -2.*(SA       *(y1+y2)**2. + SB        *(y1+y3)**2.)
    dy1dt +=     (3.*SA-sA)*(y1+y2)     + (3.*SB-sB)*(y1+y3)
    dy1dt += sA - SA + sB - SB
    dy1dt  = y1 * dy1dt
    dy1dt += - rAB * D

    dy2dt  = -2.*(SA       *(y1+y2)**2. + SB        *(y1+y3)**2.)
    dy2dt +=     (3.*SA-sA)*(y1+y2)     + (SB   -sB)*(y1+y3)
    dy2dt += sA - SA
    dy2dt  = y2 * dy2dt
    dy2dt += rAB * D

    dy3dt  = -2.*(SA       *(y1+y2)**2. + SB        *(y1+y3)**2.)
    dy3dt +=     (SA   -sA)*(y1+y2)     + (3.*SB-sB)*(y1+y3)
    dy3dt += sB - SB
    dy3dt  = y3 * dy3dt
    dy3dt += rAB * D

    return np.array([dy1dt, dy2dt, dy3dt])

# Parameters for the problem
nx   = 299
xmin = 0.
xmax = 100.
tmin = 0.
tmax = 300.

phi, T, grille = evolve(nx, 3, xmin, xmax, tmin, tmax, kappa, frhs)

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
    S = 0.1
    return 1./(1.+np.exp(np.sqrt(S)*(x-centre)))

def energie(centre, x, y, S):
    erreur = 0.
    for i in range(0, y.size):
        erreur += (y[i] - 1./(1.+np.exp(np.sqrt(S)*(x[i]-centre))))**2.
    return erreur/y.size

def gradenergie(centre, x, y, S):
    grad = 0.
    for i in range(0, y.size):
        grad += - (y[i] - 1./(1.+np.exp(np.sqrt(S)*(x[i]-centre)))) * np.sqrt(S) * \
        np.exp(np.sqrt(S)*(x[i]-centre)) * (1.+np.exp(np.sqrt(S)*(x[i]-centre)))**-2.
    return 2.*grad/y.size

centres = np.zeros((phi.shape[0], 2))
for n in range(phi.shape[0]): # fit
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
petit = 0.05
ymin0 = p.min() - petit
ymax0 = p.max() + petit
ymin1 = dcentresqdt[2 : ].min() - petit
ymax1 = dcentrespdt[2 : ].max() + petit

# 1. create a figure with several subplots
figure1, axes = plt.subplots(1, 2, dpi = 80, figsize = (16, 12), sharex = 'col') # faire attention aux dpi si les figures sont pour latex

# 2. initialize the line objects (one in each axes)
line1p,  = axes[0].plot([], [], lw = 2, label = 'p')
line1q,  = axes[0].plot([], [], lw = 2, label = 'q')
line1D,  = axes[0].plot([], [], lw = 2, label = 'D')
line1vp, = axes[1].plot([], [], lw = 2, label = 'speed of p')
color1p  = line1vp.get_color()
axes[1].hlines(cp, tmin, tmax, lw = 1, colors = color1p, linestyles = 'dashed')
line1vq, = axes[1].plot([], [], lw = 2, label = 'speed of q')

# 3. axes initializations
for ax in axes:
    ax.grid()
axes[0].set_xlim(xmin, xmax)
axes[1].set_xlim(tmin, tmax)
axes[0].set_ylim(ymin0, ymax0)
axes[1].set_ylim(ymin1, ymax1)
axes[0].set_xlabel('space')
axes[1].set_xlabel('time')
axes[0].legend(handles = [line1p, line1q, line1D])
axes[1].legend(handles = [line1vp, line1vq])

def animate(i):
    """ animate function """
    line1p.set_data (grille.x, p[i, : ])
    line1q.set_data (grille.x, q[i, : ])
    line1D.set_data (grille.x, D[i, : ])
    line1vp.set_data(T[2 : i], dcentrespdt[2 : i])
    line1vq.set_data(T[2 : i], dcentresqdt[2 : i])
    figure1.suptitle('time = %.1f' % T[i])
    axes[0].set_title('p, q, D at time = %.1f' % T[i])
    axes[1].set_title('speed of p and q until time = %.1f' % T[i])

# call animation
courbe = animation.FuncAnimation(figure1, animate, frames = T.size, repeat = False)
figure1.show()



#figure2 = plt.figure(figsize = (12, 8), dpi = 80)
#for n in range(0, phi.shape[0], 3):
#    plt.clf()
#    plt.title('Allele frequencies at time $t = {:3.1f}$'.format(T[n]))
#    plt.grid(True)
#    plt.xlim(xmin, xmax)
#    plt.ylim(ymin0, ymax0)
#    plt.xlabel('Space')
#    plt.ylabel('')
#    plt.plot(grille.x, p[n, : ], label = 'p')
#    plt.plot(grille.x, q[n, : ], label = 'q')
#    plt.plot(grille.x, D[n, : ], label = 'D')
#    plt.legend(loc = 'best')
#    figure2.show()
#    plt.pause(0.1)
#    FileNameTimestepN = 'sA'+str(sA)+'SA'+str(SA)+'sB'+str(sB)+'SB'+str(SB)+'rAB'+str(rAB)+'timestep'+str(n)+'.png'
##    plt.savefig('data/'+FileNameTimestepN, bbox_inches='tight')
#
#figure3 = plt.figure(figsize = (12, 8), dpi = 80)
#plt.title('Instantaneous wave speed')
#line3vp, = plt.plot(T[2 : ], dcentrespdt[2 : ], label = 'speed of p')
#color3p = line3vp.get_color()
#plt.hlines(cp, tmin, tmax, lw = 1, colors = color3p, linestyles = 'dashed')
#plt.plot(T[2 : ], dcentresqdt[2 : ], label = 'speed of q')
#plt.grid(True)
#plt.xlim(tmin, tmax)
#plt.ylim(ymin1, ymax1)
#plt.xlabel('Time')
#plt.ylabel('Wave speed')
#plt.legend(loc = 'best')
#figure3.show()
#FileNameSpeeds = 'sA'+str(sA)+'SA'+str(SA)+'sB'+str(sB)+'SB'+str(SB)+'rAB'+str(rAB)+'speeds.png'
#plt.savefig('data/'+FileNameSpeeds, bbox_inches='tight')
#
#
#
#FileNameGlobal = 'sA'+str(sA)+'SA'+str(SA)+'sB'+str(sB)+'SB'+str(SB)+'rAB'+str(rAB)
#
# save the animation as a gif file
#print ('debut de sauvegarde')
#start1 = time.time()
#courbe.save('data/'+FileNameGlobal+'.gif', dpi = 80, writer = 'imagemagick')
#print ('temps de sauvegarde', time.time() - start1)
#print ('fin de sauvegarde')
#
# save the data as an npy file
#np.save('data/'+FileNameGlobal+'.npy', phi)

