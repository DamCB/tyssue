from tyssue import stores
from pathlib import Path
import numpy as np
from tyssue import config, Sheet
from tyssue import PlanarGeometry
from tyssue.io import hdf5
from tyssue.solvers.quasistatic import QSSolver
from tyssue.dynamics.planar_vertex_model import PlanarModel as model


def test_relaxation_convergance():
    # Here we relax a pre-counstructed periodic tissue object to its "periodic equilibrium" configuration
    # this tissue object that is loaded is far away from equilibrium 6x6 is the "periodic" equilibrium
    dsets = hdf5.load_datasets(Path(stores.stores_dir) / "planar_periodic8x8.hf5")
    specs = config.geometry.planar_sheet()
    specs["settings"]["boundaries"] = {"x": [-0.1, 8.1], "y": [-0.1, 8.1]}
    initial_box_size = 8.2
    sheet = Sheet("periodic", dsets, specs)
    coords = ["x", "y"]
    draw_specs = config.draw.sheet_spec()
    PlanarGeometry.update_all(sheet)
    solver = QSSolver(with_collisions=False, with_t1=True, with_t3=False)
    nondim_specs = config.dynamics.quasistatic_plane_spec()
    dim_model_specs = model.dimensionalize(nondim_specs)
    sheet.update_specs(dim_model_specs, reset=True)
    # epsilon is deviation of boundary between iterations
    epsilon = 1.0
    # max_dev is the max epsilon allowed for configuration to be in equilibrium
    max_dev = 0.001
    # i counts the number of solver iterations
    i = 0
    # loop ends if box size variation between iterations drops 10^-3
    # loaded tissue is far away from periodic equilibrium L=8 and equilibrium is reached around 6.06
    while np.abs(epsilon) > max_dev:
        i += 1
        if i == 1:
            previous_box_size = 0
        else:
            previous_box_size = solution_result["x"][-1]
        solution_result = solver.find_energy_min(
            sheet, PlanarGeometry, model, periodic=True
        )
        epsilon = solution_result["x"][-1] - previous_box_size
        # print(epsilon)
    final_box_size = (
        sheet.settings["boundaries"]["x"][1] - sheet.settings["boundaries"]["x"][0]
    )
    print("number of iterations  " + str(i))
    print("final box size  " + str(final_box_size))
    assert 6.06 - 0.1 < final_box_size < 6.06 + 0.1
