# tyssue-demo

Small demo notebooks for tyssue


## Usage

1. Install tyssue from conda-forge in a python >= 3.6 environement:
```bash
conda install -c conda-forge tyssue
```

2. Leasurely explore the notebooks


3. You can even try it in MyBinder without installing anything:

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/DamCB/tyssue-demo/master)


## What you find in notebooks

00-Basic_Usage : Introduction to generation of tissue in 2D, data structure and input/output + specification
 
01-Geometry : Presentation of geometries include in tyssue.

02-Visualisation : Explanation of some plot functions.

03-Dynamics : Explanation of models and associated specs.

04-Solvers: Presentation of quasistatic and dynamic solver. Presentation of the object `history`. 

05-Rearangments : Explanation of how rearangments works.

06-Cell_Division : Explanation of of how cell division works.

07-EventManager : Presentation of the `eventManager` object which manage events during simulation.


A-ReversibleNetworkReconnection : Reproduce the reversible network reconnection model from Okuda et al. 

B-Apoptosis : Example of apoptotic cell on apical 3D epithelium.

C-2DMigration : Reproduce the model of Mapeng Bi et al. of cell migration in 2D epithelium.  

D-FarhadifarModel : Exemple of the Farhadifar model from his paper in 2007. 

