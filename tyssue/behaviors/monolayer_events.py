import logging
from .sheet_events import SheetEvents

# from ..topology.sheet_topology import (remove_face,
#                                        type1_transition,
#                                        cell_division)
logger = logging.getLogger(__name__)


class MonoLayerEvents(SheetEvents):

    def __init__(self, monolayer, model, geom):
        self.monolayer = monolayer
        self.model = model
        self.geom = geom

    @property
    def events(self):
        return {
            'shrink': self.shrink,
            'grow': self.grow,
            'contract': self.contract,
            'ab_pull': self.ab_pull,
            }

    def shrink(self, cell, *args):

        factor = args[0]
        faces = self.monolayer.edge_df[
            self.monolayer.edge_df['cell'] == cell]['face']
        new_vol = self.monolayer.cell_df.loc[cell, 'prefered_vol'] * factor
        self.monolayer.cell_df.loc[cell, 'prefered_vol'] = new_vol
        new_areas = self.monolayer.face_df.loc[
            faces, 'prefered_area'] * factor**(2/3)
        self.monolayer.face_df.loc[faces, 'prefered_area'] = new_areas

    def grow(self, cell, *args):
        self.shrink(cell, *args)

    def ab_pull(self, cell, *args):

        cell_edges = self.monolayer.edge_df[
            self.monolayer.edge_df['cell'] == cell]
        sagittal_edges = cell_edges[
            cell_edges['segment'] == 'sagittal']
        srce_segment = self.monolayer.upcast_srce(
            self.monolayer.vert_df['segment']).loc[sagittal_edges.index]
        trgt_segment = self.monolayer.upcast_trgt(
            self.monolayer.vert_df['segment']).loc[
                sagittal_edges.index]

        ab_edges = sagittal_edges[(srce_segment == 'apical') &
                                  (trgt_segment == 'basal')].index
        ba_edges = sagittal_edges[(trgt_segment == 'apical') &
                                  (srce_segment == 'basal')].index
        factor = args[0]
        new_tension = self.monolayer.specs['edge']['line_tension'] * factor
        self.monolayer.edge_df.loc[ab_edges, 'line_tension'] += new_tension
        self.monolayer.edge_df.loc[ba_edges, 'line_tension'] += new_tension

    def contract(self, face, *args):

        factor = args[0]
        new_contractility = self.monolayer.specs['face'][
            'contractility'] * factor
        self.monolayer.face_df.loc[face, 'contractility'] += new_contractility
