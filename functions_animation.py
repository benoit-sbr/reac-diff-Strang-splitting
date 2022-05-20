#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 11:05:11 2022

@author: sarels
"""
def animate(i, grille, p, q, D, dcentrespdt, dcentresqdt, T, axes, figure, lines):
    """Animate function
    """
    linep, lineq, lineD, linevp, linevq = lines
    linep.set_data (grille.x, p[i, : ])
    lineq.set_data (grille.x, q[i, : ])
    lineD.set_data (grille.x, D[i, : ])
    linevp.set_data(T[2 : i], dcentrespdt[2 : i])
    linevq.set_data(T[2 : i], dcentresqdt[2 : i])
    figure.suptitle('time = %.1f' % T[i])
    axes[0].set_title('p, q, D at time = %.1f' % T[i])
    axes[1].set_title('speed of p and q until time = %.1f' % T[i])


def init_my_axes(axes, lines, xlim_axes0, xlim_axes1, ylim_axes0, ylim_axes1):
    """Init axes function
    """
    linep, lineq, lineD, linevp, linevq = lines
    for ax in axes:
        ax.grid()
    axes[0].set_xlim(xlim_axes0)
    axes[1].set_xlim(xlim_axes1)
    axes[0].set_ylim(ylim_axes0)
    axes[1].set_ylim(ylim_axes1)
    axes[0].set_xlabel('space')
    axes[1].set_xlabel('time')
    axes[0].legend(handles = [linep, lineq, lineD])
    axes[1].legend(handles = [linevp, linevq])
