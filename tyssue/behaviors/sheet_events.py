"""
Event management module
=======================


"""

import logging
import pandas as pd
import warnings
import random
from collections import deque
from ..topology.sheet_topology import (remove_face,
                                       type1_transition,
                                       cell_division)
logger = logging.getLogger(__name__)


class SheetEvents():

    def __init__(self, sheet, model, geom):
        self.sheet = sheet
        self.model = model
        self.geom = geom
        self.current_deque = deque()
        self.next_deque = deque()

    def idx_lookup(self, face_id):
        return self.sheet.face_df[
            self.sheet.face_df.id == face_id].index[0]

    @property
    def events(self):
        events = {
            'shrink': self.shrink,
            'grow': self.grow,
            'contract': self.contract,
            'type1_at_shorter': self.type1_at_shorter,
            'type3': self.type3,
            'divide': self.divide,
            'ab_pull': self.ab_pull,
            'type1_at_shorter_pull': self.type1_at_shorter_pull,
        }
        return events

    def add_events(self, face_event):
        """
        Add new face with their event.

        Parameters
        ----------
        face_event : list of tuples (face, event)

        """

        self.current_deque.extend(face_event)

    def execute_behaviors(self):
        """
        Execute event present in current_deque
        Complete the next deque with the corresponding event for each cell.
        Replace current_deque by next_deque and clear it.
        """

        while self.current_deque:
            (face, event_name) = self.current_deque.popleft()
            print('face: {}, event: {}'.format(face, event_name))
            self.events[event_name](face)

        random.shuffle(self.next_deque)
        self.current_deque = self.next_deque.copy()
        self.next_deque.clear()

    def contract(self, face):
        """
        Contract the face by multiplying the contractility by a factor.

        Parameters
        ----------
        face : id face
        """
        if face not in self.sheet.face_df['id'].values:
            return
        face_line = self.sheet.face_df.loc[self.idx_lookup(face)]
        settings = self.sheet.settings
        if face_line['area'] > settings['delamination']['critical_area']:
            factor = self.sheet.settings[
                'delamination']['contractile_increase']
            new_contractility = self.sheet.specs[
                'face']['contractility'] * factor
            self.sheet.face_df.loc[self.idx_lookup(face),
                                   'contractility'] += new_contractility

            self.next_deque.append((face, 'contract'))
        else:
            self.next_deque.append((face, 'type1_at_shorter_pull'))

    def type1_at_shorter_pull(self, face):
        """
        Put an apico-basal tension and execute a type1 transition
        for the minimal edge of the face.

        Parameters
        ----------
        face : id face
        """
        if face not in self.sheet.face_df['id'].values:
            return
        # pull event
        verts = self.sheet.edge_df[self.sheet.edge_df['face'] ==
                                   self.idx_lookup(face)]['srce'].unique()
        factor = self.sheet.settings['delamination']['radial_tension']
        new_tension = self.sheet.specs['vert']['radial_tension'] * factor
        self.sheet.vert_df.loc[verts, 'radial_tension'] += new_tension

        # type1 event
        edges = self.sheet.edge_df[self.sheet.edge_df['face'] ==
                                   self.idx_lookup(face)]
        shorter = edges.length.idxmin()
        type1_transition(self.sheet, shorter)
        self.geom.update_all(self.sheet)

        if face not in self.sheet.face_df['id'].values:
            return
        if self.sheet.face_df.loc[self.idx_lookup(face)]['num_sides'] > 4:
            self.next_deque.append((face, 'type1_at_shorter_pull'))

    def type1_at_shorter(self, face):
        """
        Execute a type1 transition for the minimal edge of the face.

        Parameters
        ----------
        face : id face
        """
        if face not in self.sheet.face_df['id'].values:
            return
        edges = self.sheet.edge_df[self.sheet.edge_df['face'] ==
                                   self.idx_lookup(face)]
        shorter = edges.length.idxmin()
        type1_transition(self.sheet, shorter)

        self.geom.update_all(self.sheet)

        if face not in self.sheet.face_df['id'].values:
            return
        if self.sheet.face_df.loc[self.idx_lookup(face)]['num_sides'] > 4:
            self.next_deque.append((face, 'type1_at_shorter'))

    def type3(self, face, *args):
        """
        Execute a type3 transition by removing the face when it has 3 edges.

        Parameters
        ----------
        face : id face
        """
        if face not in self.sheet.face_df['id'].values:
            return
        try:
            if self.sheet.face_df.loc[self.idx_lookup(face)]['num_sides'] == 3:
                remove_face(self.sheet, face)
                self.geom.update_all(self.sheet)
            elif self.sheet.face_df.loc[self.idx_lookup(face)]['num_sides'] > 3:
                self.events['type1_at_shorter'](face)
                #self.next_deque.append((face, 'type3'))
        except ValueError:
            print(
                'failed: face: {}, error: {Try to remove face with less \
                            than 3 faces}'.format(face))
            raise ValueError

    def shrink(self, face, *args):
        warnings.simplefilter("always")
        warnings.warn("La fonction shrink n'est pas a jour avec \
                            collections.deque.", DeprecationWarning)

        factor = args[0]
        new_vol = self.sheet.specs['face']['prefered_vol'] * factor
        self.sheet.face_df.loc[self.idx_lookup(face),
                               'prefered_vol'] = new_vol

    def grow(self, face, *args):
        warnings.simplefilter("always")
        warnings.warn("La fonction shrink n'est pas a jour avec \
                            collections.deque.", DeprecationWarning)
        self.shrink(face, *args)

    def ab_pull(self, face, *args):
        warnings.simplefilter("always")
        warnings.warn("La fonction shrink n'est pas a jour avec \
                            collections.deque.", DeprecationWarning)
        verts = self.sheet.edge_df[self.sheet.edge_df['face'] ==
                                   self.idx_lookup(face)]['srce'].unique()
        factor = args[0]
        new_tension = self.sheet.specs['vert']['radial_tension'] * factor
        self.sheet.vert_df.loc[verts, 'radial_tension'] += new_tension

    def divide(self, face, *args):
        warnings.simplefilter("always")
        warnings.warn("La fonction shrink n'est pas a jour avec \
                            collections.deque.", DeprecationWarning)
        cell_division(self.sheet, self.idx_lookup(face), self.geom, *args)
