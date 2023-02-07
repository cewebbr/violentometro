import matplotlib.pyplot as pl
import numpy as np
import pandas as pd


def build_alternating_divergent_series(first_instance=0):
    """
    Return a generator of the sequence
    0, 1, -1, 2, -2, 3, -3, ...
    """
    y = 1 + first_instance
    while True:
        sign = (-1) ** (y % 2)
        yield sign * int( y / 2)
        y = y + 1   

        
def split_sequences(sorted_x, sorted_x_bin, max_diff):
    """
    Split instances of clusters `sorted_x_bin` (array)
    positioned at `sorted_x` (array) when they get are 
    farther than `max_diff` from the cluster leader.
    """
    
    # Get cluster leader label:
    cluster_leader = np.insert(np.diff(sorted_x_bin), 0, 1).astype(bool)
    # Check if instance is far from leader:
    far_from_leader = (np.subtract.outer(sorted_x, sorted_x[cluster_leader]) > max_diff)[range(len(sorted_x)), sorted_x_bin]
    # Create offset for new clusters:
    far_switch = np.cumsum(np.insert(np.diff(far_from_leader.astype(int)), 0, 0) > 0)
    # New clusters:
    new_sorted_bins = sorted_x_bin + far_switch
    
    return new_sorted_bins


def gen_beeswarm_y(x, crowd_frac=0.01, y_offset=0, delta_y=1, unsort=False):
    """
    Generate a dummy variable for a 1D 
    dataset to be used to make a beeswarm 
    plot (if `x` is the horizontal coordinate, 
    then return the vertical coordinate).
    
    If `unsort` is True, return x and y in the
    original `x` position. Else, return x and y 
    sorted by `x` value.
    """
    
    # Compute x stats:
    min_x = np.min(x)
    max_x = np.max(x)
    max_diff = crowd_frac * (max_x - min_x)
    
    # Bin the values:
    sort_idx = np.argsort(x)
    sorted_x = np.array(x)[sort_idx]
    sorted_x_bin = np.insert(np.cumsum(np.diff(sorted_x) > max_diff), 0, 0)
    # Split clusters if they are long chains in the X direction: 
    sorted_x_bin = split_sequences(sorted_x, sorted_x_bin, max_diff)
    sorted_x_bin = split_sequences(sorted_x, sorted_x_bin, max_diff)
    sorted_x_bin = split_sequences(sorted_x, sorted_x_bin, max_diff)
        
    # Assign bins to unsorted x, if required:
    if unsort:
        unsort = np.empty_like(sort_idx)
        unsort[sort_idx] = np.arange(len(x))
        x_bin = sorted_x_bin[unsort]
        out_x = x
    else:
        x_bin = sorted_x_bin
        out_x = sorted_x
    
    # Build y generators:
    bin_idx = np.unique(x_bin)
    bin_gen = {k:build_alternating_divergent_series(y_offset) for k in bin_idx}
    
    # Generate ys:
    y = np.array(list(map(lambda x: next(bin_gen[x]), x_bin))) * delta_y
    
    return out_x, y


def beeswarm(x, marker_size=100, alpha=0.5, color=None, label=None, y_offset=0, vertical=False, unsort=False, fig_width=10, height_range=[2, 10]):
    """
    Make a beeswarm plot.
    
    Input
    -----
    
    x : list of array of numbers.
        Values to plot.
    
    marker_size : float
        Size of the plot marker (as specified by pl.scatter).
    
    alpha : float
        Opacity of the marker, from 0.0 to 1.0.
        
    color : str or None
        Color of the marker (or None to assign it)
        automatically.
        
    label : str or None
        Label of the points, to appear in legend.
        
    y_offset : int
        Start index of the sequence used to generate the
        position perpendicular to the plot direction.
    
    vertical : bool
        Whether to plot the `x` along the vertical
        or the horizontal axis.
    
    unsort : bool
        Whether the point positions perpendicular to the 
        plot direction should follow the original order 
        of `x` or not (i.e. it should be sorted).
        
    fig_width : float
        Width of the final figure.
    
    height_range : tuple/list of two floats
        Min. and max. heights of the plot (the 
        exact height is set accordingly to the 
        number of points on similar `x` values).
    """
    
    # Scaling factor:
    scale = np.sqrt(marker_size)
 
    if len(x) > 0:
        # Generate ys:
        uncluster_scale = 0.015 * scale / fig_width
        x, y = gen_beeswarm_y(x, uncluster_scale, y_offset, unsort=unsort)

        # Get y info:
        y_max = np.max(y)
        y_min = np.min(y)
        n_points = y_max - y_min + 1
    else:
        # Empty X safety option:
        y = []
        y_min = 0
        y_max = 1
        n_points = 1
    
    # Generate plot:
    min_height = height_range[0]
    max_height = height_range[1]
    fig_height = max(0.01875 * n_points * scale + 0.3, min_height)
    fig_height = min(fig_height, max_height)
    
    if vertical:
        fig = pl.figure(num=1, figsize=(fig_height, fig_width))
        pl.scatter(y, x, s=marker_size, alpha=alpha, c=color, label=label)
        pl.xlim([y_min - 1.5, y_max + 1.5])
        pl.gca().get_xaxis().set_visible(False)
    else:
        fig = pl.figure(num=1, figsize=(fig_width, fig_height))
        pl.scatter(x, y, s=marker_size, alpha=alpha, c=color, label=label)
        pl.ylim([y_min - 1.5, y_max + 1.5])
        pl.gca().get_yaxis().set_visible(False)
        
    return fig


def multiple_bars_plot(df, colors=None, alpha=None, err=None, width=0.8, rotation=0, horizontal=False):
    """
    Create a bar plot with bars from different columns 
    in `df` side by side.
    
    Parameters
    ----------
    df : DataFrame
        Data to plot as bars. Each row corresponds to a 
        different entry, translating to bar positions, 
        and each column correponds to a different data 
        series, each with a different color. The series
        labels are the column names and the bar location
        labels are the index. The data is plotted in the
        order set in `df`.
    colors : list, str or None.
        Colors for the data series (`df` columns).    
    alpha : list, float or None.
        Transparency of the columns.
    err : DataFrame, tuple of two DataFrames, or None    
        Errors for the data in `df`, to be plot as error 
        bars. They should have the same structure as `df`.
        If one DataFrame, use simmetrical error bars. If 
        a tuple of DataFrames, they are the lower and upper
        errors. If None, do not draw error bars.
    width : float
        Total column width formed by all summing the 
        widths of each data series.
    rotation : float
        Rotation of the column axis labels, given 
        in degrees.
    horizontal : bool
        Whether to use horizontal bar plot or not.
    """
    
    # Count number of columns (associated to bar colors):
    cols   = df.columns
    n_cols = len(cols)
    # Count number of rows (associated to bar positions):
    rows   = df.index
    n_rows = len(rows)

    # Standardize input:
    if type(colors) != list:
        colors = [colors] * n_cols
    if type(alpha) != list:
        alpha = [alpha] * n_cols

    # Organize errors, if provided:
    if err is not None:
        
        # Standardize one error to simmetrical errors:
        if type(err) == pd.core.frame.DataFrame:
            err = (err, err)
        assert type(err) == tuple, '`err` must be a DataFrame or a tuple of DataFrames.'
        assert len(err) == 2, '`err` must have two elements.'

        # Check that the errors table have the same structure as the data:
        for i, e in enumerate(err):
            assert set(e.columns) == set(df.columns), 'Columns in `err` {} are not the same as in `df`.'.format(i + 1)
            assert set(e.index) == set(df.index), 'Index in `err` {} are not the same as in `df`.'.format(i + 1)
            
        # Order the errors like the data:
        err0_df = err[0].loc[df.index]
        err1_df = err[1].loc[df.index]
    
    # Set plotting x position:
    ind = np.arange(n_rows)
    # Set width of columns:
    wid = width / n_cols
    
    # Loop over columns:
    for i, col in enumerate(cols):
        
        # Select the columns' errors:
        if err is not None:
            err0 = err0_df[col].values
            err1 = err1_df[col].values
            errs = np.stack([err0, err1])
        else:
            errs = None
        
        # Bar plot:
        if horizontal:
            pl.barh(ind - wid / 2 * (n_cols - 1) + wid * i, df[col], height=wid, xerr=errs, ecolor=colors[i], color=colors[i], alpha=alpha[i], label=col)
        else:
            pl.bar(ind - wid / 2 * (n_cols - 1) + wid * i, df[col], width=wid, yerr=errs, ecolor=colors[i], color=colors[i], alpha=alpha[i], label=col)

    # Set tick labels:
    ax = pl.gca()
    if horizontal:
        ax.set_yticks(ind)
        ax.set_yticklabels(rows, rotation=rotation)
    else:
        ax.set_xticks(ind)
        ax.set_xticklabels(rows, rotation=rotation)


def hist_err_val(a, bins=10, density=False):
    """
    Compute the input for a plot of an histogram (vertical) error bars.
    
    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars or str, optional
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a
        sequence, it defines a monotonically increasing array of bin edges,
        including the rightmost edge, allowing for non-uniform bin widths.
    density : bool, default: False
        If ``True``, draw and return a probability density: each bin
        will display the bin's raw count divided by the total number of
        counts *and the bin width*
        (``density = counts / (sum(counts) * np.diff(bins))``),
        so that the area under the histogram integrates to 1
        (``np.sum(density * np.diff(bins)) == 1``).

    Returns
    -------
    x : array
        The center of the bins; the X position of the errorbars.
    y : array
        The histogram heights; the Y position of the center of the errorbars.
    e : array
        The length of the error bars.
    """
    
    # Count the values in the bins:
    counts, edges = np.histogram(a, bins)
    # Get bins' properties:
    x  = (edges[1:] + edges[:-1]) / 2
    dx = (edges[1:] - edges[:-1])
    
    # Compute uncertainty:
    n  = counts.sum()
    prob = counts / n
    dev  = np.sqrt(counts * (1 - prob))
    
    # Normalize to density if requested:
    if density is True:
        y = prob / dx
        e = dev / n / dx
    else:
        y = counts
        e = dev 
    
    return x, y, e


def hist_err_plot(a, bins=10, density=False, ealpha=None, ecolor=None):
    """
    Plot the (vertical) error bars of an histogram.
    
    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars or str, optional
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a
        sequence, it defines a monotonically increasing array of bin edges,
        including the rightmost edge, allowing for non-uniform bin widths.
    density : bool, default: False
        If ``True``, draw and return a probability density: each bin
        will display the bin's raw count divided by the total number of
        counts *and the bin width*
        (``density = counts / (sum(counts) * np.diff(bins))``),
        so that the area under the histogram integrates to 1
        (``np.sum(density * np.diff(bins)) == 1``).
    ealpha : float or None
        Opacity of the error bars.
    ecolor : color, default: None
        The color of the errorbar lines.
    """   
    
    # Compute the values of the parameters of the error bars:
    center, h, dev = hist_err_val(a, bins, density=density)
    # Plot:
    pl.errorbar(center, h, yerr=dev, linestyle='none', alpha=ealpha, ecolor=ecolor)

    
def errorbar_hist(x, bins=None, density=False, histtype='bar', rwidth=None, color=None, label=None, alpha=0.6, ealpha=None, ecolor=None):
    """
    Plot a histogram with multinomial error bars.
    
    Parameters
    ----------
    x : (n,) array or sequence of (n,) arrays
        Input values, this takes either a single array or a sequence of
        arrays which are not required to be of the same length.

    bins : int or sequence or str, default: :rc:`hist.bins`
        If *bins* is an integer, it defines the number of equal-width bins
        in the range.

        If *bins* is a sequence, it defines the bin edges, including the
        left edge of the first bin and the right edge of the last bin;
        in this case, bins may be unequally spaced.  All but the last
        (righthand-most) bin is half-open.  In other words, if *bins* is::

            [1, 2, 3, 4]

        then the first bin is ``[1, 2)`` (including 1, but excluding 2) and
        the second ``[2, 3)``.  The last bin, however, is ``[3, 4]``, which
        *includes* 4.

        If *bins* is a string, it is one of the binning strategies
        supported by `numpy.histogram_bin_edges`: 'auto', 'fd', 'doane',
        'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
        
    density : bool, default: False
        If ``True``, draw and return a probability density: each bin
        will display the bin's raw count divided by the total number of
        counts *and the bin width*
        (``density = counts / (sum(counts) * np.diff(bins))``),
        so that the area under the histogram integrates to 1
        (``np.sum(density * np.diff(bins)) == 1``).

        If *stacked* is also ``True``, the sum of the histograms is
        normalized to 1.

    histtype : {'bar', 'step', 'stepfilled'}, default: 'bar'
        The type of histogram to draw.

        - 'bar' is a traditional bar-type histogram.  If multiple data
          are given the bars are arranged side by side.
        - 'barstacked' is a bar-type histogram where multiple
          data are stacked on top of each other.
        - 'step' generates a lineplot that is by default unfilled.
        - 'stepfilled' generates a lineplot that is by default filled.
        
    rwidth : float or None, default: None
        The relative width of the bars as a fraction of the bin width.  If
        ``None``, automatically compute the width.

        Ignored if *histtype* is 'step' or 'stepfilled'.

    color : color or array-like of colors or None, default: None
        Color or sequence of colors, one per dataset.  Default (``None``)
        uses the standard line color sequence.

    label : str or None, default: None
        String, or sequence of strings to match multiple datasets.  Bar
        charts yield multiple patches per dataset, but only the first gets
        the label, so that `~.Axes.legend` will work as expected.
        
    alpha : float or None
        Opacity of the histogram bars.
    
    ealpha : float or None
        Opacity of the error bars. If None, set it to a little more than 
        `alpha`.
    ecolor : color or array-like of colors or None, default: None
        Color of the error bars. If None, set to the same as `color`.
    """
    
    # Plot the histogram:
    pl.hist(x, bins=bins, density=density, histtype=histtype, rwidth=rwidth, color=color, label=label, alpha=alpha)
    
    # Set color and opacity of the error bars:
    if alpha is not None and ealpha is None:
        ealpha = min(alpha + 0.15, 1.0)
    if ecolor is None:
        ecolor = color
    
    # Plot error bars:
    hist_err_plot(x, bins=bins, density=density, ealpha=ealpha, ecolor=ecolor)