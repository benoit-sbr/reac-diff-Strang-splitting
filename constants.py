#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:36:50 2022

@author: sarels
"""

import numpy as np

# Parameters for initial condition
xp0 = 40.
xq0 = 60.

# Parameters for diffusion
kappa = np.array([1., 1., 1.])

# Parameters for reaction
sA = 0.01
SA = 0.03
sB = 0.01
SB = 0.03
rAB = 0.03
reac_params = (sA, SA, sB, SB, rAB)

# Parameters for the problem
nx   = 399
xmin = 0.
xmax = 200.
space_bounds = (xmin, xmax)
tmin = 0.
tmax = 25.
time_bounds = (tmin, tmax)
