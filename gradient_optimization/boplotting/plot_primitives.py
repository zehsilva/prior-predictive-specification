
import numpy as np
import matplotlib.pyplot as plt

import warnings
from functools import reduce
from matplotlib.colors import LogNorm


def plot_err(x, y, yerr, color="salmon", alpha_fill=0.2, 
             ax=None, label="", lw=2, ls="-"):
    if len(y.shape)!=1: y, yerr = y.reshape(-1), yerr.reshape(-1)    
    ax = ax if ax is not None else plt.gca()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, label=label, lw=lw, ls=ls)
    ax.fill_between(x, ymax, ymin, 
            color=color, alpha=alpha_fill, linewidth=0.0)


def fix_grid(t):
    """Removes numeric issues"""
    t = np.round(t, 8) 
    return t


def sparsify_ticks(pos, t, max_count=10):
    if len(t)<=max_count: return t
    step = len(t) // (max_count//2)
    warnings.warn("[plot_2D] Sparsifying ticks to increase readibility!")
    return pos[::step], t[::step]


def plot_2D(xgrid1, xgrid2, g, kind="im", **kwargs):
    xgrid1, xgrid2 = fix_grid(xgrid1), fix_grid(xgrid2)
    plt.scatter([0.0], [0.0], marker="+", s=60, color="limegreen", zorder=100) # mark zero 
    
    g = g.copy()
    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)
    if not vmin is None: g[g<vmin] = vmin
    if not vmax is None: g[g>vmax] = vmax      

    norm = None
    if kwargs.get("norm", None)=="log": 
        norm = LogNorm(vmin=vmin, vmax=vmax)  
        if vmin is not None and vmin<=0.0:
            warnings.warn("[plot_2D] Vmin=%s <= 0!" % vmin)
            vmin = 1e-16
          
    if kind=="smooth":
        im = plt.contourf(xgrid1, xgrid2, g, 
                     kwargs.get("ygrid", 100), 
                     #vmin=kwargs.get("vmin", None), 
                     #vmax=kwargs.get("vmax", None), 
                     cmap=kwargs.get("cmap", "Reds"), norm=norm
                    )
    elif kind=="grid": # TODO axis proportion fails for very different scales of X1 vs X2
        g[range(len(g)),:] = g[range(len(g)-1,-1,-1),:] # invert y-axis  
        im = plt.imshow(g, 
                        vmin=kwargs.get("vmin", None), 
                        vmax=kwargs.get("vmax", None), 
                        cmap=kwargs.get("cmap", "Reds"),
                        extent=[xgrid1.min(), xgrid1.max(), xgrid2.min(), xgrid2.max()],
                        aspect='auto', norm=norm
                        );    
        
        h = reduce(lambda a,b: a and b, (xgrid1[i]-xgrid1[i+1]==xgrid1[i+1]-xgrid1[i+2] for i in range(len(xgrid1)-2)))
        if not h: warnings.warn("[plot_2D] Horizontal grid is non linear. Ticks' labels are incorrect!")
        v = reduce(lambda a,b: a and b, (xgrid2[i]-xgrid2[i+1]==xgrid2[i+1]-xgrid2[i+2] for i in range(len(xgrid2)-2)))
        if not v: warnings.warn("[plot_2D] Vertical grid is non linear. Ticks' labels are incorrect!")
    else: 
        im = plt.imshow(g, 
                        vmin=kwargs.get("vmin", None), 
                        vmax=kwargs.get("vmax", None), 
                        cmap=kwargs.get("cmap", "Reds"), norm=norm
                        ); 

        xpos = list(range(len(xgrid1)))
        ypos = list(range(len(xgrid2)))
        xpos, xgrid1 = sparsify_ticks(xpos, xgrid1, kwargs.get("numxticks", 15))
        ypos, xgrid2 = sparsify_ticks(ypos, xgrid2, kwargs.get("numyticks", 30))

        plt.xticks(xpos, xgrid1); 
        plt.yticks(ypos, xgrid2); 
        y1, y2 = plt.ylim(); plt.ylim( (y2, y1) ) # invert y-axis            
    return im

