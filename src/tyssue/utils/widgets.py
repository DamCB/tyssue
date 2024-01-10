import ipywidgets as ipw


def parameter_settings(eptm):
    specs = eptm.specs
    elements = []

    for element in ["edge", "vert", "face"]:
        if element not in specs:
            continue

        spec = specs[element]
        fts = []
        for param, val in spec.items():

            def update_param(change):
                specs[element][param] = change["new"]
                print(change)
                print("{} {} changed to {}".format(element, param, change["new"]))

            w = ipw.FloatText(val, description=param)
            w.observe(update_param, names="value")
            fts.append(w)
        elements.append(ipw.VBox(fts))

    return ipw.HBox(elements)
