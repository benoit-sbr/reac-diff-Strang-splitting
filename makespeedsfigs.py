import numpy as np
from matplotlib import pyplot as plt

from theory import cp0, cp1

FileName = 's0.01S0.05speeds'
FigName = FileName + '.png'
numerical_speeds = np.loadtxt(FileName)
sA = float(FileName[1:5])
SA = float(FileName[6:10])
numerical_abscissae = numerical_speeds[ : , 0]
#theoretical_abscissae = np.linspace(numerical_abscissae[0], numerical_abscissae[-1])
theoretical_abscissae = np.linspace(0.01, numerical_abscissae[-1], 200) # plus joli

theoretical_speeds = cp1(sA, SA, theoretical_abscissae)

figure1, axes = plt.subplots(1, 1, sharex = 'col') # faire attention aux dpi si les figures sont pour latex
plt.title('Speeds for $s = {0}$ and $S = {1}$'.format(sA, SA))

#$t = {:3.1f}$'.format(T[n])

plt.xlabel('Parameter r')
plt.ylabel('Speed')
plt.ylim(0.043, 0.076)
plt.grid(True)
plt.plot(numerical_abscissae, numerical_speeds[ : , 1], 'bo', label = 'numeric')
plt.plot(theoretical_abscissae, theoretical_speeds, label = 'theory')
plt.hlines(cp0(sA, SA), numerical_abscissae[0], numerical_abscissae[-1], lw = 1, linestyles = 'dashed', label = '$s/\sqrt{S}$')
plt.hlines(cp0(2.*sA, 2.*SA), numerical_abscissae[0], numerical_abscissae[-1], lw = 1, linestyles = 'dashdot', label = '$2s/\sqrt{2S}$')
plt.legend(loc = 'best')
figure1.show()
plt.pause(5.)
plt.savefig(FigName, bbox_inches='tight')
plt.close(figure1)
