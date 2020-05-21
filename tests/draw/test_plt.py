import matplotlib
import pytest

matplotlib.use("Agg")
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from tyssue.generation import three_faces_sheet
from tyssue import Sheet, config
from tyssue.draw.plt_draw import quick_edge_draw, sheet_view
from tyssue.draw.plt_draw import _face_color_from_sequence


class TestsPlt:

    sheet = Sheet("test", *three_faces_sheet())
    draw_specs = config.draw.sheet_spec()

    def test_quick_edge_draw(self):

        fig, ax = quick_edge_draw(self.sheet)
        assert ax.lines[0].get_xydata().shape == (54, 2)

    def test_sheet_view(self):
        self.sheet = Sheet("test", *three_faces_sheet())
        self.sheet.vert_df["rand"] = np.linspace(
            0.0, 1.0, num=self.sheet.vert_df.shape[0]
        )
        cmap = plt.cm.get_cmap("viridis")
        color_cmap = cmap(self.sheet.vert_df.rand)
        self.draw_specs["vert"]["visible"] = True
        self.draw_specs["vert"]["color"] = color_cmap
        self.draw_specs["vert"]["alpha"] = 0.5
        self.draw_specs["vert"]["s"] = 500
        self.sheet.face_df["col"] = np.linspace(
            0.0, 1.0, num=self.sheet.face_df.shape[0]
        )
        self.draw_specs["face"]["color"] = self.sheet.face_df["col"]

        self.draw_specs["face"]["visible"] = True
        self.draw_specs["face"]["alpha"] = 0.5

        self.sheet.edge_df["rand"] = np.linspace(
            0.0, 1.0, num=self.sheet.edge_df.shape[0]
        )[::-1]

        self.draw_specs["edge"]["visible"] = True
        self.draw_specs["edge"]["color"] = self.sheet.edge_df["rand"]  # [0, 0, 0, 1]
        self.draw_specs["edge"]["alpha"] = 1.0
        self.draw_specs["edge"]["color_range"] = 0, 3
        self.draw_specs["edge"]["width"] = 1.0 * np.linspace(
            0.0, 1.0, num=self.sheet.edge_df.shape[0]
        )

        fig, ax = sheet_view(self.sheet, ["x", "y"], **self.draw_specs)
        assert len(ax.collections) == 3
        assert ax.collections[0].get_edgecolors().shape == (13, 4)
        assert ax.collections[1].get_edgecolors().shape == (18, 4)
        assert ax.collections[2].get_edgecolors().shape == (0, 4)
        assert ax.collections[2].get_facecolors().shape == (3, 4)

        self.draw_specs["edge"]["head_width"] = 1.0
        fig, ax = sheet_view(self.sheet, ["x", "y"], **self.draw_specs)
        assert len(ax.collections) == 3
        assert ax.collections[0].get_edgecolors().shape == (13, 4)
        assert ax.collections[1].get_edgecolors().shape == (18, 4)
        assert ax.collections[2].get_edgecolors().shape == (0, 4)
        assert ax.collections[2].get_facecolors().shape == (3, 4)

    def test_sheet_view_color_string(self):
        self.draw_specs["edge"]["color"] = "k"
        self.draw_specs["face"]["color"] = "red"
        fig, ax = sheet_view(self.sheet, ["x", "y"], **self.draw_specs)
        assert ax.collections[1].get_edgecolors().shape == (1, 4)
        assert ax.collections[2].get_facecolors().shape == (1, 4)

    def test_sheet_view_color_partial_visibility(self):
        self.draw_specs["face"]["color"] = np.random.rand(3, 4)
        self.sheet.face_df["visible"] = False
        self.sheet.face_df.loc[0, "visible"] = True
        fig, ax = sheet_view(self.sheet, ["x", "y"], **self.draw_specs)
        assert ax.collections[2].get_facecolors().shape == (1, 4)

    def test_sheet_view_color_null_visibility(self):
        self.draw_specs["face"]["color"] = np.random.rand(3, 4)
        self.sheet.face_df["visible"] = False
        with pytest.warns(UserWarning):
            fig, ax = sheet_view(self.sheet, ["x", "y"], **self.draw_specs)
        assert ax.collections[2].get_facecolors().shape == (3, 4)

    def test_sheet_view_homogenous_color(self):
        with pytest.warns(UserWarning):
            self.draw_specs["face"]["color"] = np.ones(3)
            fig, ax = sheet_view(self.sheet, ["x", "y"], **self.draw_specs)

        with pytest.warns(UserWarning):
            self.draw_specs["edge"]["color"] = np.ones(self.sheet.Ne)
            fig, ax = sheet_view(self.sheet, ["x", "y"], **self.draw_specs)

        with pytest.warns(UserWarning):
            self.draw_specs["edge"]["color"] = np.ones(self.sheet.Nv)
            fig, ax = sheet_view(self.sheet, ["x", "y"], **self.draw_specs)

    def test_sheet_view_invalid_color_array(self):
        with pytest.raises(ValueError):
            self.draw_specs["face"]["color"] = np.ones(5)
            fig, ax = sheet_view(self.sheet, ["x", "y"], **self.draw_specs)

    def test_per_vertex_edge_colors(self):

        self.draw_specs["face"]["color"] = "red"
        self.draw_specs["edge"]["color"] = np.random.random(self.sheet.Nv)
        fig, ax = sheet_view(self.sheet, ["x", "y"], **self.draw_specs)

    def test_sheet_view_callable(self):
        with pytest.raises(ValueError):
            self.draw_specs["face"]["color"] = lambda sheet: np.ones(5)
            self.draw_specs["edge"]["color"] = lambda sheet: np.ones(5)
            fig, ax = sheet_view(self.sheet, ["x", "y"], **self.draw_specs)
