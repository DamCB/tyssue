#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:49:53 2019

@author: georgecourcoubetis
"""

import time
start = time.time()

from tyssue import stores
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
#%matplotlib inline
#from tyssue.generation import hexa_grid2d
from tyssue import config, Sheet
from tyssue import SheetGeometry, PlanarGeometry
from tyssue.io import hdf5

from tyssue.draw.plt_draw import quick_edge_draw
from tyssue.draw import sheet_view

from tyssue.topology.sheet_topology import remove_face, cell_division
from tyssue.solvers.quasistatic import QSSolver
from tyssue.dynamics.planar_vertex_model import PlanarModel as model




dsets = hdf5.load_datasets(Path(stores.stores_dir)/'planar_periodic8x8.hf5')
specs = config.geometry.planar_sheet()
specs['settings']['boundaries'] =  {'x': [-0.1, 8.1], 'y': [-0.1, 8.1]}
sheet = Sheet('periodic', dsets, specs)
coords=['x','y']
draw_specs = config.draw.sheet_spec()
PlanarGeometry.update_all(sheet)
# seems like we need to call this twice for cells to be labeled out of boundary
PlanarGeometry.update_all(sheet)
# solver
solver = QSSolver(with_collisions=False, with_t1=True, with_t3=False)
nondim_specs = config.dynamics.quasistatic_plane_spec()
dim_model_specs = model.dimensionalize(nondim_specs)
sheet.update_specs(dim_model_specs, reset=True)
    


fig, ax = sheet_view(sheet, coords, **draw_specs)
fig.set_size_inches(12, 5)



#arbitrarily choose a cell to divide
div_cell=sheet.face_df.index[(sheet.face_df['at_x_boundary'] == True)&(sheet.face_df['at_y_boundary']== False)].tolist()[0]
print("test cell division on boundary begins (rember point is that in the original cell_div, cells on the boundary cannot divide) ")
daughter = cell_division(sheet,div_cell,PlanarGeometry,angle=0.6)
sheet.face_df.loc[div_cell, "prefered_area"]=1.0
sheet.face_df.loc[daughter, "prefered_area"]=1.0
solver.find_energy_min(sheet, PlanarGeometry, model)
fig, ax = sheet_view(sheet, coords, **draw_specs)
fig.set_size_inches(12, 5)
print("test_end")


print("test cell division on both x and y boundary")
div_cell=sheet.face_df.index[(sheet.face_df['at_x_boundary'] == True)&(sheet.face_df['at_y_boundary'] == True)].tolist()[0]
daughter = cell_division(sheet,div_cell,PlanarGeometry,angle=0.6)
sheet.face_df.loc[div_cell, "prefered_area"]=1.0
sheet.face_df.loc[daughter, "prefered_area"]=1.0
solver.find_energy_min(sheet, PlanarGeometry, model)
fig, ax = sheet_view(sheet, coords, **draw_specs)
fig.set_size_inches(12, 5)
print("test_end")

print("test cell division on bulk")
div_cell=sheet.face_df.index[(sheet.face_df['at_x_boundary'] == False)&(sheet.face_df['at_y_boundary']== False)].tolist()[0]
daughter = cell_division(sheet,div_cell,PlanarGeometry,angle=0.6)
sheet.face_df.loc[div_cell, "prefered_area"]=1.0
sheet.face_df.loc[daughter, "prefered_area"]=1.0
solver.find_energy_min(sheet, PlanarGeometry, model)
fig, ax = sheet_view(sheet, coords, **draw_specs)
fig.set_size_inches(12, 5)
print("test_end")


plt.show()
