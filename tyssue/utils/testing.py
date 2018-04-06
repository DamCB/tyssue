import numpy as np

from ..dynamics.factory import model_factory
from ..dynamics.effectors import AbstractEffector
from ..generation import three_faces_sheet


def effector_test(eptm, effector):

    for elem, spec in effector.specs.items():

        for col in spec:
            if col not in eptm.datasets[elem].columns:
                eptm.datasets[elem][col] = 1.

    energy = effector.energy(eptm)
    assert energy.shape == (eptm.datasets[effector.element].shape[0],)
    assert np.all(np.isfinite(energy.values))

    grad_s, grad_t = effector.gradient(eptm)
    assert np.all(np.isfinite(grad_s))

    if grad_s.shape[0] == eptm.Nv:
        assert grad_t is None
    elif grad_s.shape[0] == eptm.Ne:
        assert grad_t.shape[0] == eptm.Ne
        assert np.all(np.isfinite(grad_t))



def model_test(eptm, model):

    nondim_specs = {elem: {k : 1 for k in spec}
                    for elem, spec in model.specs.items()}
    dim_specs = model.dimensionalize(nondim_specs)
    for elem, spec in model.specs.items():
        assert set(dim_specs[elem].keys()) == spec

    nondim_specs = model.dimensionalize(dim_specs)
    for elem, spec in model.specs.items():
        assert set(nondim_specs[elem].keys()) == spec

    eptm.update_specs(dim_specs, reset=False)
    energies = model.compute_energy(eptm, full_output=True)
    assert len(energies) == len(model.labels)
    energy = model.compute_energy(eptm, full_output=False)
    assert np.isfinite(energy)

    gradients = model.compute_gradient(eptm, components=True)
    assert len(gradients) == len(model.labels)
    gradient = model.compute_gradient(eptm, components=False)
    assert gradient.shape == eptm.vert_df[eptm.coords].shape
    assert np.all(np.isfinite(gradient))
