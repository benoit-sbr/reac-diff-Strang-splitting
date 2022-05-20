#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:48:28 2022

@author: sarels
"""

import numpy as np

def frhs(t, phi, x, reac_params):
    """ reaction ODE righthand side """

    sA, SA, sB, SB, rAB = reac_params
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
