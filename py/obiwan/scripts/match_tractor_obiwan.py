import logging
import numpy as np
from ..catalog import SimCatalog
from .. import utils

logger = logging.getLogger('Matching')

def match(output_dir,brickname,base='input',radius_in_degree=1.5/3600.,**kwargs_file):
    """
    Match and merge random input catalog to **Tractor** output.

    Input columns ``field`` are relabelled ``'sim_'+field``.

    Parameters
    ----------
    output_dir : string
        Obiwan output directory.

    brickname : string
        Brick name.

    base : string, default=`input`
        If `input`, returns input catalog with tractor measurements.
        If `output`, returns **Tractor** catalog with input values.
        If 'inter', returns the intersection of input and **Tractor** catalogs.

    radius_in_degree : float, default=1.5/3600.
        Radius to sphere-match both catalogs.

    kwargs_file : dict, default={}
        Other Arguments for file paths (``fileid``, ``rowstart``, ``skipid``).

    Returns
    -------
    match : SimCatalog
        Matched catalog.
    """
    input_fn = utils.get_output_file(output_dir,'obiwan-randoms',brickname=brickname,**kwargs_file)
    tractor_fn = utils.get_output_file(output_dir,'tractor',brickname=brickname,**kwargs_file)
    logger.info('Reading input %s' % input_fn)
    logger.info('Reading Tractor %s' % tractor_fn)

    cat_input = SimCatalog(input_fn)
    cat_tractor = SimCatalog(tractor_fn)

    index_input,index_tractor = cat_input.match_radec(cat_tractor,nearest=True,radius_in_degree=radius_in_degree)

    for field in cat_input.fields:
        cat_input.rename(field,'sim_%s' % field)

    if base == 'input':
        cat_input.merge(cat_tractor,index_self=index_input,index_other=index_tractor)
        return cat_input
    else:
        cat_tractor.merge(cat_input,index_self=index_tractor,index_other=index_input)
    if base == 'output':
        return cat_tractor
    return cat_tractor[index_tractor]

@utils.saveplot()
def scatter_match(ax,values1=None,values2=None,match=None,field=None,xlabel=None,ylabel=None,square=False,regression=False,diagonal=False,kwargs_scatter={},kwargs_regression={},kwargs_diagonal={}):
    """
    Plot ``values2`` v.s. ``values1`` or ``match[field]`` v.s. ``match['sim_'+field]``.

    Parameters
    ----------
    ax : plt.axes
        Where to plot.

    values1 : array-like, default=None
        x values.

    values2 : array-like, default=None
        y values.

    match : SimCatalog
        If ``values1`` is None, ``values1`` and ``values2`` taken from ``match``.

    field : string, default=None
        Field of ``match`` to plot.

    xlabel : string, default=None
        x label, if ``None`` and ``match`` is provided, defaults to ``'sim_'+field``.

    ylabel : string, default=None
        y label, if ``None`` and ``match`` is provided, defaults to ``field``.

    square : bool, default=False
        Whether to enforce square plot.

    regression : bool, default=False
        Whether to plot regression line.

    diagonal : bool, default=False
        Whether to plot diagonal line.

    kwargs_scatter : dict, default={}
        Arguments for ``plt.scatter()``.

    kwargs_regression : dict, default={}
        Arguments for ``plt.plot()`` regression line.

    kwargs_diagonal : dict, default={}
        Arguments for ``plt.plot()`` diagonal line.
    """
    kwargs_scatter = utils.dict_default(kwargs_scatter,{'s':10,'marker':'.','alpha':1,'edgecolors':'none'})
    kwargs_diagonal = utils.dict_default(kwargs_diagonal,{'linestyle':'--','linewidth':2,'color':'k'})
    kwargs_regression = utils.dict_default(kwargs_regression,{'linestyle':'--','linewidth':2,'color':'r','label':''})
    label_regression = kwargs_regression.pop('label',None)

    if values1 is None:
        values1,values2 = match.get('sim_%s' % field),match.get(field)
        if xlabel is None: xlabel = 'sim_%s' % field
        if ylabel is None: ylabel = field

    ax.scatter(values1,values2,**kwargs_scatter)

    if square:
        xlim,ylim = ax.get_xlim(),ax.get_ylim()
        xylim = min(xlim[0],ylim[0]),max(xlim[1],ylim[1])
        ax.axis('square')
        ax.set_xlim(xylim)
        ax.set_ylim(xylim)

    xlim,ylim = [np.array(tmp) for tmp in [ax.get_xlim(),ax.get_ylim()]]
    a,b = np.polyfit(values1,values2,1)
    y = a*xlim + b
    r = np.corrcoef(values1,values2)[0,1]

    if label_regression is not None:
        label = '$\\rho = %.3f$' % r
        if label_regression:
            label = '%s %s' % (label_regression,label)
    else: label = None

    if regression:
        ax.plot(xlim,y,label=label,**kwargs_regression)
        ax.set_ylim(ylim)
    elif label:
        ax.text(0.95,0.05,label,horizontalalignment='right',verticalalignment='bottom',transform=ax.transAxes,color='k')
    if diagonal:
        ax.plot(xlim,xlim,**kwargs_diagonal)
        ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
