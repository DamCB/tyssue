import numpy as np


def update_spherical(eptm):
    """ Computes the spherical coordinates (rho, theta, phi)
    of an epithelium.

    rho is the distance to the coordinate system's origin, theta
    is the co-latitude (0 ≤ θ < π) and phi is the longitude (0 ≤ ϕ < 2π).
    """
    for element in ["vert", "face", "cell"]:
        if eptm.datasets.get(element) is None:
            continue
        eptm.datasets[element]["rho"] = np.linalg.norm(
            eptm.datasets[element][["x", "y", "z"]], axis=1
        )
        eptm.datasets[element]["theta"] = np.arccos(
            eptm.datasets[element].eval("z / rho")
        )
        eptm.datasets[element]["phi"] = np.arctan2(
            eptm.datasets[element].eval("y / (rho * sin(theta))"),
            eptm.datasets[element].eval("x / (rho * sin(theta))"),
        )


def rotation_matrix(angle, direction):
    # Copyright (c) 2006-2015, Christoph Gohlke
    # Copyright (c) 2006-2015, The Regents of the University of California
    # Produced at the Laboratory for Fluorescence Dynamics
    # All rights reserved.
    #
    # Redistribution and use in source and binary forms, with or without
    # modification, are permitted provided that the following conditions
    #  are met:
    #
    # * Redistributions of source code must retain the above copyright
    #   notice, this list of conditions and the following disclaimer.
    # * Redistributions in binary form must reproduce the above copyright
    #   notice, this list of conditions and the following disclaimer in the
    #   documentation and/or other materials provided with the distribution.
    # * Neither the name of the copyright holders nor the names of any
    #   contributors may be used to endorse or promote products derived
    #   from this software without specific prior written permission.
    #
    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    # "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    # LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    # FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
    # IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    # LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    # CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    # SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    # INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    # CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    # ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
    # OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    Returns the 3X3 rotation matrix around `direction` by `angle`

    adapted from http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
    """
    sina = np.sin(angle)
    cosa = np.cos(angle)
    direction = direction / np.linalg.norm(direction)
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        [
            [0.0, -direction[2], direction[1]],
            [direction[2], 0.0, -direction[0]],
            [-direction[1], direction[0], 0.0],
        ]
    )
    return R


def rotation_matrices(angle, direction):
    """ Return an (N, 3, 3) array of rotation matrices
    along N angles and N directions



    Parameters
    ----------
    angle : np.ndarray of shape (N,)
        array of rotation angles

    directions : np.ndarray of shape (N, 3)
        array of rotation vectors

    Returns
    -------
    rots : np.ndarray of shape (N, 3, 3)
        the array of rotation matrices
    """

    sint, cost = np.sin(angle), np.cos(angle)

    rots = np.zeros((cost.size, 3, 3))
    for i in range(3):
        rots[:, i, i] = cost
    rots += np.einsum("ij, ik -> ijk", direction, direction) * (1 - cost[:, None, None])

    direction *= sint[:, None]

    rots[:, 0, 1] -= direction[:, 2]
    rots[:, 0, 2] += direction[:, 1]
    rots[:, 1, 0] += direction[:, 2]
    rots[:, 1, 2] -= direction[:, 0]
    rots[:, 2, 0] -= direction[:, 1]
    rots[:, 2, 1] += direction[:, 0]

    return rots
