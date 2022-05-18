import numpy as np
from matplotlib import pyplot as plt

from theory import cp0, cp1

FileNameRoot = 'sA0.0SA0.1sB0.0SB0.1rAB0.1closerIC'
FolderName = 'data' + FileNameRoot + '/'
phi = np.load(FolderName + FileNameRoot + '.npy')
y1 = phi[ : , 0*phi.shape[1]//3 : 1*phi.shape[1]//3 ]
y2 = phi[ : , 1*phi.shape[1]//3 : 2*phi.shape[1]//3 ]
y3 = phi[ : , 2*phi.shape[1]//3 : ]
y4 = 1. - y1 - y2 - y3

nx = 299	# number of interior points
ng = 1		# number of points to handle a boundary, 1 in 1D
xmin = 0.
xmax = 100.
dx = (xmax - xmin) / (nx + ng)
x  = np.arange(ng + nx + ng) * dx + xmin

ymin = 0.
ymax = 1.

n  = 900
time = 0.2357 * n

figure1 = plt.figure()
plt.title('Genotype frequencies at time $t = {:3.1f}$'.format(time))
plt.grid(True)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
#    plt.xlabel('Space')
#    plt.ylabel('')
plt.plot(x, y1[n, : ], label = 'u')
plt.plot(x, y2[n, : ], label = 'v')
plt.plot(x, y3[n, : ], label = 'w')
plt.plot(x, y4[n, : ], label = 'z')
plt.legend(loc = 'best')
figure1.show()
#plt.pause(5.)
FileNameTimestepN = FileNameRoot + 'timestep' + str(n) + 'uvwz.png'
plt.savefig(FolderName + FileNameTimestepN, bbox_inches = 'tight')
plt.close(figure1)
