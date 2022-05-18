import numpy as np

# Theoretical results
def cp0(sA, SA):
    return sA/np.sqrt(SA)

def cp1(sA, SA, rAB):
    return cp0(sA, SA) + 4.*sA*np.sqrt(SA)/15./rAB
