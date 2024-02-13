from .objects import get_prev_edges
from .sheet import Sheet


class LateralSheet(Sheet):
    """
    2D lateral model
    """

    def segment_index(self, segment, element):
        df = getattr(self, "{}_df".format(element))
        return df[df["segment"] == segment].index

    @property
    def lateral_edges(self):
        return self.segment_index("lateral", "edge")

    @property
    def apical_edges(self):
        return self.segment_index("apical", "edge")

    @property
    def basal_edges(self):
        return self.segment_index("basal", "edge")

    @property
    def apical_verts(self):
        return self.segment_index("apical", "vert")

    @property
    def basal_verts(self):
        return self.segment_index("basal", "vert")

    def reset_topo(self):
        super().reset_topo()
        self.edge_df["prev"] = get_prev_edges(self)

    @staticmethod
    def get_line_edges(self, sub_edges):
        sub_edges = sub_edges.reset_index()
        srces, trgts, edge = sub_edges[["srce", "trgt", "edge"]].to_numpy().T

        # Beginning of the line
        start_e = sub_edges.iloc[0]["srce"]
        for v in sub_edges["srce"].to_numpy():
            if v not in sub_edges["trgt"].to_numpy():
                start_e = sub_edges[sub_edges['srce'] == v].index[0]

        srce, trgt, edge_ = srces[start_e], trgts[start_e], edge[start_e]
        edges = [[srce, trgt, edge_]]

        for i in range(len(srces) - 1):
            srce, trgt = trgt, trgts[srces == trgt][0]
            edge_ = sub_edges[(sub_edges["srce"] == srce)]["edge"].to_numpy()[0]
            edges.append([srce, trgt, edge_])
        return edges

    def get_apical_surface(self):
        sub_edges = self.edge_df.loc[self.apical_edges]
        return self.get_line_edges(sub_edges)

    def get_basal_surface(self):
        sub_edges = self.edge_df.loc[self.basal_edges]
        return self.get_line_edges(sub_edges)

