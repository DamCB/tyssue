#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import IPython.core.display as disp
import json

import sys, os
curdir = os.path.abspath(os.path.curdir)
#print os.path.dirname(curdir)
sys.path.append(os.path.dirname(curdir))

import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (8.0,6.0)    #(6.0,4.0)
mpl.rcParams['font.size'] = 12                #10
mpl.rcParams['savefig.dpi'] = 100             #72
mpl.rcParams['figure.subplot.bottom'] = .1    #.125

import graph_tool.all as gt
#import matplotlib.pyplot as plt
import leg_joint as lj
import numpy as np

def before_after(func):
    def new_func(eptm, *args, **kwargs):
        import matplotlib.pyplot as plt
        import leg_joint as lj
        #eptm.update_gradient()
        fig, axes = plt.subplots(1,4, figsize=(12,4))
        axes_before = lj.plot_2pannels(eptm, axes[0:2])
        lj.plot_2pannels_gradients(eptm, axes_before, scale=10.)
        foutput = func(eptm, *args, **kwargs)
        #eptm.update_gradient()
        axes_after = lj.plot_2pannels(eptm, axes[2:])
        lj.plot_2pannels_gradients(eptm, axes_after, scale=10.)
        return foutput
    return new_func

@before_after
def show_optimisation(eptm, **kwargs):
    #eptm.update_gradient()
    pos0, pos1 = lj.find_energy_min(eptm, **kwargs)
    return pos0, pos1


def local_optimum(*arg, **kwargs):

    return show_optimisation(*arg, **kwargs)