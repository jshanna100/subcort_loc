import matplotlib.pyplot as plt
import numpy as np

def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True, color="blue",
                  area_fill=True, alpha=1, dot_size=2):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=area_fill, linewidth=1, alpha=alpha,
                     color=color)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches, radius.max()

def circ_hist_norm(ax, x, points=None, vecs=None, bins=16, density=True, offset=0,
                   gaps=True, color="blue", area_fill=True, alpha=1,
                   points_col=None, dot_size=2):
    n, bins, patches, r_max = circular_hist(ax, x, bins=bins, density=density,
                                            offset=offset, gaps=gaps, color=color,
                                            area_fill=area_fill, alpha=alpha,
                                            dot_size=dot_size)
    if points is not None:
        full_points = np.array([(x, r_max) for x in np.nditer(points)])
        ax.scatter(full_points[:,0], full_points[:,1], s=dot_size,
                   facecolors="None", edgecolors=points_col, linewidths=3.5)
    if vecs is not None:
        for vec in vecs:
            r = np.arange(0, r_max*vec[0][1], 0.01)
            theta = np.ones_like(r)*vec[0][0]
            ax.plot(theta, r, **vec[1])
