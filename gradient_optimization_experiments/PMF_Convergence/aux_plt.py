import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import seaborn as sns; sns.set(); sns.set(font_scale=1.0); sns.set_style("white"); 

#import warnings
#warnings.filterwarnings("ignore")


GREEN = "limegreen"
BLUE = "dodgerblue"
RED = "salmon"
    
BLUES = ["dodgerblue", "#F0F8FF", "#E6E6FA", "#B0E0E6", "#ADD8E6", "#87CEFA", "#87CEEB", "#00BFFF", 
         "#B0C4DE", "#1E90FF", "#6495ED", "#4682B4", "#5F9EA0", "#7B68EE", "#6A5ACD", "#483D8B", "#4169E1", 
         "#0000FF", "#0000CD", "#00008B", "#000080", "#191970", "#8A2BE2", "#4B0082"]
REDS = ["salmon", "#FFA07A","#E9967A","#F08080","#CD5C5C","#DC143C","#B22222",
        "#FF0000","#8B0000","#800000","#FF6347","#FF4500","#DB7093"]    
COLORS = ['dodgerblue', 'salmon',  'limegreen', 'teal', 'mediumspringgreen', 'violet',  'crimson']



def _reset_mpl_config(font_size = 17*1.5, cmbright=True):
    mpl.rcParams.update(mpl.rcParamsDefault) #reset to defaults
        
    SMALL_SIZE = font_size-4
    MEDIUM_SIZE = font_size
    BIGGER_SIZE = font_size
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.rc('font',**{'family':'serif','serif':['Times'], "weight": "normal"})
    plt.rc('text', usetex=True)
    plt.rc('mathtext', fontset='stix')  #['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom']
    
    mpl.rcParams['text.latex.preamble'] = [
            r'\usepackage{mathtools}',
            r'\usepackage{amsmath}',
            r'\usepackage{amsfonts}', 
            r'\usepackage{microtype}',    
            r'\usepackage{arydshln}',              
    ] + ([r'\usepackage{cmbright}'] if cmbright else [])


def _create_fig(bottom=0.2, left=0.125, right=0.9, top=0.9):
    fig = plt.figure(figsize=(6.4, 4.8), dpi=72)
    fig.subplots_adjust(bottom=bottom, left=left, right=right, top=top) 
    
    
def start_plotting(cmbright=True, font_size=17*1.5, bottom=0.2, left=0.125, right=0.95, top=0.95):
    _reset_mpl_config(cmbright=cmbright, font_size=font_size)
    _create_fig(bottom=bottom, left=left, right=right, top=top)


def my_formatter3(x, pos):
    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
    formatted as -.4)."""
    val_str = "%.3f" % x
    if np.abs(x) > 0 and np.abs(x) < 1:
        return val_str.replace("0.", ".", 1)
    else:
        return val_str
    
    
def my_formatter2(x, pos):
    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
    formatted as -.4)."""
    val_str = "%.2f" % x
    if np.abs(x) > 0 and np.abs(x) < 1:
        return val_str.replace("0.", ".", 1)
    else:
        return val_str    
    

# Set up the formatter.
major_formatter3 = FuncFormatter(my_formatter3)    
major_formatter2 = FuncFormatter(my_formatter2)    


def fix_colors(bp, color):
    # boxplot style adjustments
    [item.set_linewidth(2) for item in bp['boxes']]
    [item.set_linewidth(2) for item in bp['fliers']]
    [item.set_linewidth(2) for item in bp['medians']]
    [item.set_linewidth(2) for item in bp['means']]
    [item.set_linewidth(0.5) for item in bp['whiskers']]
    [item.set_linewidth(0.5) for item in bp['caps']]

    [item.set_color(color) for item in bp['boxes']]
    [item.set_color("k") for item in bp['fliers']]
    [item.set_color(color) for item in bp['medians']]
    [item.set_color("k") for item in bp['means']]
    [item.set_color("k") for item in bp['whiskers']]
    [item.set_color("k") for item in bp['caps']]




def running_mean1(x):
    x = list(x)
    return [np.mean(x[i: min(i+2, len(x))]) for i in range(len(x))]


def running_mean11(x):
    x = list(x)
    return [np.mean(x[max(i-1,0): min(i+1, len(x))]) for i in range(len(x))]


def running_mean(x, N=0):
    x = list(x)
    if N==1: return running_mean1(x)
    if N==-1: return running_mean11(x)
    if N<=0: return x
    l = N//2    
    return [np.mean(x[max(i-l,0): min(i+l+1, len(x))]) for i in range(len(x))]


def errorfill(x, y, yerr, color=None, alpha_fill=0.2, ax=None, label="", lw=2, ls="-", smooth=0):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, running_mean(y, smooth), color=color, label=label, lw=lw, ls=ls)
    ax.fill_between(x, running_mean(ymax, smooth), running_mean(ymin, smooth), color=color, alpha=alpha_fill, linewidth=0.0)
    
    
def extract_mean_std(df, x, y):
    df = df.sort_values(x)
    return df.groupby(x).mean().index, df.groupby(x).mean()[y], df.groupby(x).std()[y]



