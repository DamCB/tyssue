from ..json_parser import load_spec
import os

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


def sheet_spec():
    """Default specification for drawing/graphical output
    functions for a sheet - also suitable for general faceted tissue

    .. code-block::

        {
            "edge": {
            "visible": true,
                "width": 0.5,
                "head_width": 0.2,
                "length_includes_head": true,
                "shape": "right",
                "color": "#2b5d0a",
                "alpha": 0.8,
                "zorder": 1
            },
            "vert": {
                "visible": true,
                "s": 100,
                "color": "#000a4b",
                "alpha": 0.3,
                "zorder": 2
            },
            "grad": {
                "color":"#000a4b",
                "alpha":0.5,
                "width":0.04
            },
            "face": {
                "visible": false,
                "color":"#8aa678",
                "alpha": 1.0,
                "zorder": -1
            }
        }

    """
    specfile = os.path.join(CURRENT_DIR, "sheet.json")
    return load_spec(specfile)
