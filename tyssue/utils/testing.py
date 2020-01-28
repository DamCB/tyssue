import numpy as np


def effector_tester(eptm, effector):

    for u in eptm.ucoords:
        eptm.edge_df[u] = 0

    for elem, spec in effector.specs.items():

        for col in spec:
            if col not in eptm.datasets[elem].columns:
                eptm.datasets[elem][col] = 1.0

    energy = effector.energy(eptm)
    assert energy.shape == (eptm.datasets[effector.element].shape[0],)
    assert np.all(np.isfinite(energy.values))

    grad_s, grad_t = effector.gradient(eptm)
    assert np.all(np.isfinite(grad_s))

    if grad_s.shape == (eptm.Nv, eptm.dim):
        assert grad_t is None
    elif grad_s.shape == (eptm.Ne, eptm.dim):
        assert grad_t.shape[0] == eptm.Ne
        assert np.all(np.isfinite(grad_t))
    else:
        raise ValueError(
            f"""
            The computed gradients for effector {effector.label}
            should have shape {(eptm.Ne, eptm.dim)} or {(eptm.Nv, eptm.dim)},
            found {grad_s.shape}.
            """
        )


def model_tester(eptm, model):

    for u in eptm.ucoords:
        eptm.edge_df[u] = 0

    nondim_specs = {elem: {k: 1 for k in spec} for elem, spec in model.specs.items()}
    dim_specs = model.dimensionalize(nondim_specs)
    for elem, spec in model.specs.items():
        assert set(dim_specs[elem].keys()) == set(spec.keys())

    nondim_specs = model.dimensionalize(dim_specs)
    for elem, spec in model.specs.items():
        assert set(nondim_specs[elem].keys()) == set(spec.keys())

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
