"""
A collection of utils for plotting.
"""
import numpy as np

from scipy.signal import savgol_filter


def tsplot(x, y, ax,
           ci=68,
           color=None,
           deriv=0,
           polyorder=3,
           resample=None,
           window_length=None):
    """Plot the data after applying Savitzky-Golay filter.

    Arguments:
    ----------
        x : 2D np.array
        y : 2D np.array
            Each row should correspond to a time series.
        ax : matplotlib axis object
        ci : float or tuple of 2 floats (default: 68)
            Quantile values to be plotted.
        color : matplotlib color
        deriv : int (default: 0)
        polyorder : int (default :3)
        resample : int or None (default: None)
            Total number of points to resample.
            If None, it is equal to `np.min(np.max(x, axis=1))`.
    """
    assert len(x.shape) == len(y.shape) == 2

    resample = resample or int(np.min(np.max(x, axis=1)))
    x_interp = np.arange(1, resample)
    y_interp = np.stack([
        np.interp(x_interp, x_1d, y_1d)
        for x_1d, y_1d in zip(x, y)
    ])

    if isinstance(ci, float) or isinstance(ci, int):
        ci = (100. - ci) / 2.
        ci = (ci, 100. - ci)
    assert len(ci) == 2

    if window_length is None:
        window_length = int(0.01 * len(x))
        window_length += 1 - (window_length % 2)

    # Filter
    y_filt = savgol_filter(y_interp,
                           window_length=window_length,
                           polyorder=polyorder,
                           deriv=deriv,
                           axis=-1)
    y_low = np.percentile(y_filt, ci[0], axis=0)
    y_high = np.percentile(y_filt, ci[1], axis=0)
    y_mean = np.mean(y_filt, axis=0)

    # Plot
    ax.plot(x_interp, y_mean, color=color)
    ax.fill_between(x_interp, y_low, y_high, color=color, alpha=0.3)

    return ax
