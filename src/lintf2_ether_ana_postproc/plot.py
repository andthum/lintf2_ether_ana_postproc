"""Module containing functions for plotting."""

# First-party libraries
import lintf2_ether_ana_postproc as leap


atom_type2display_name = {
    "Li": r"Li",
    "NBT": r"N_{TFSI}",
    "OBT": r"O_{TFSI}",
    "OE": r"O_{PEO}",
}


def plot_elctrd_left(ax, offset=0, **kwargs):
    """
    Plot the position of the electrode layers of the left electrode as
    vertical lines into an :class:`matplotlib.axes.Axes`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` into which to plot the
        electrode positions.
    offset : float, optional
        x value at which to position the rightmost layer of the left
        electrode.
    kwargs : dict
        Keyword arguments to parse to
        :func:`matplotlib.axes.Axes.axvline`.  See there for possible
        options.  By default, `color` is set to ``"tab:gray"`` and
        `linestyle` is set to ``"dashed"``.
    """
    kwargs.setdefault("color", "tab:gray")
    kwargs.setdefault("linestyle", "dashed")

    Elctrd = leap.simulation.Electrode()
    elctrd_pos_z = offset
    for _ in range(Elctrd.GRA_LAYERS_N):
        ax.axvline(x=elctrd_pos_z, **kwargs)
        elctrd_pos_z -= Elctrd.GRA_LAYER_DIST / 10  # nm -> Angstrom


def plot_elctrd_right(ax, offset=0, **kwargs):
    """
    Plot the position of the electrode layers of the right electrode as
    vertical lines into an :class:`matplotlib.axes.Axes`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` into which to plot the
        electrode positions.
    offset : float, optional
        x value at which to position the leftmost layer of the right
        electrode.
    kwargs : dict
        Keyword arguments to parse to
        :func:`matplotlib.axes.Axes.axvline`.  See there for possible
        options.  By default, `color` is set to ``"tab:gray"`` and
        `linestyle` is set to ``"dashed"``.
    """
    kwargs.setdefault("color", "tab:gray")
    kwargs.setdefault("linestyle", "dashed")

    Elctrd = leap.simulation.Electrode()
    elctrd_pos_z = offset
    for _ in range(Elctrd.GRA_LAYERS_N):
        ax.axvline(x=elctrd_pos_z, **kwargs)
        elctrd_pos_z += Elctrd.GRA_LAYER_DIST / 10  # nm -> Angstrom


def plot_elctrds(ax, offset_right, offset_left=0, **kwargs):
    """
    Plot the position of the electrode layers of the left and right
    electrode as vertical lines into an :class:`matplotlib.axes.Axes`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` into which to plot the
        electrode positions.
    offset_left, offset_right : float
        x values at which to position the rightmost layer of the left
        electrode and the leftmost layer of the right electrode.
    kwargs : dict
        Keyword arguments to parse to
        :func:`lintf2_ether_ana_postproc.plot.plot_elctrd_left` and
        :func:`lintf2_ether_ana_postproc.plot.plot_elctrd_right`.  See
        there for possible options.  By default, `color` is set to
        ``"tab:gray"`` and `linestyle` is set to ``"dashed"``.
    """
    kwargs.setdefault("color", "tab:gray")
    kwargs.setdefault("linestyle", "dashed")
    leap.plot.plot_elctrd_left(ax, offset_left, **kwargs)
    leap.plot.plot_elctrd_right(ax, offset_right, **kwargs)
