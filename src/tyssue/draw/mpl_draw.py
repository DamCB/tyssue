import matplotlib.pyplot as plt


def draw_tyssue(eptm, xx, yy, **kwargs):
    fig, ax = plt.subplots()
    ax.plot(xx, yy, **kwargs)
