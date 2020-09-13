"""Plotting of 1D or 2D slices of precomputed function from pandas DataFrame.

   Convention: 
        df  Pandas DataFrame with x_features and a special response column 'obj' containing an objective.
        x_feature_axis / x_feature1, x_feature2  Names of the columns containing the values to be plotted along the axes.
        y_feature_name="obj"  Response (objective) column name.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import plot_primitives 
#from importlib import reload  
#reload(plot_primitives)

import warnings

###############################################################################


def find_closest_df_features(df, x_features, x_position):
    """ Matching the closest to x_position point from data frame df. 
        Ideally where x_features match values from x_position.
    """
    x_position_matched = []
    for dim_name, x_position_val in zip(x_features, x_position):
        if dim_name is None or x_position_val is None:
            x_position_matched.append(None)           
        else:
            dim_vals = df[dim_name].unique()
            matching_ix = np.argmin(abs(dim_vals-x_position_val))
            x_position_matched.append(dim_vals[matching_ix])   
    return x_position_matched


def extract_slice(df, x_features, x_position, ignore_features=[], verbose=False):
    """ Returns a data frame with values where x_features (approximately) match values from x_position
        everywhere apart from ignore_features and None values.
    """
    assert len(x_position)==len(x_features)

    x_position_matched = find_closest_df_features(df, x_features, x_position)
    if verbose and (np.array(x_position_matched)!=np.array(x_position)).any(): 
        warnings.warn("[extract slice] Warning: approximate slicing at %s (~= %s)" % (x_position_matched, x_position))

    df_slice = df
    for x_feature, x_feature_postion in zip(x_features, x_position_matched):
      if (x_features is not None) and (x_feature_postion is not None) and (x_feature not in ignore_features):
        df_slice = df_slice[df_slice[x_feature]==x_feature_postion]

    return df_slice, x_position_matched


###############################################################################


def plot_df1D(df_slice, x_feature_axis, y_feature_name="obj", sign=+1, **kwargs):
    """ Plots values from data frame in 1D.
        Args:
            df_slice       data frame slice with known function values (in column y_feature_name)
            x_feature_axis name of the feature that the plot will go along
    
    """
    # obtaining means and stds
    df_means = df_slice.groupby(x_feature_axis).mean().reset_index() 
    arg = df_means[x_feature_axis]
    means = sign*df_means[y_feature_name]
    df_stds = df_slice.groupby(x_feature_axis).std().reset_index()            
    stds = df_stds[y_feature_name]
    
    plot_primitives.plot_err(arg, means, stds, 
                              label=kwargs.get("label", ""), 
                              color=kwargs.get("color", "salmon"), 
                              lw=kwargs.get("lw", 2.5),
                              ls=kwargs.get("ls", "-"))    
    plt.xlabel(kwargs.get("xlabel", x_feature_axis)); 
    plt.ylabel(kwargs.get("ylabel", y_feature_name)); 
    plt.grid(); 



def plot_df2D(df_slice, x_feature1, x_feature2, y_feature_name="obj", sign=+1, show_nwarnings=10, **kwargs):
    """Plots 2D surface of mean values over x_feature1 x x_feature2."""
    xgrid1 = np.array( sorted(df_slice[x_feature1].unique()) )
    xgrid2 = np.array( sorted(df_slice[x_feature2].unique()) )

    ix2feature1 = dict(enumerate(xgrid1))
    ix2feature2 = dict(enumerate(xgrid2))    
    g = np.zeros((len(ix2feature2), len(ix2feature1)))
    dm = df_slice.groupby([x_feature1, x_feature2]).mean().reset_index() # take means
    for r in range(len(ix2feature2)):
        for c in range(len(ix2feature1)):
            i1 = (dm[x_feature1]==ix2feature1[c])
            i2 = (dm[x_feature2]==ix2feature2[r])

            if sum(i1 & i2)<=0:
                if show_nwarnings>0: warnings.warn( ("No rows matching %s=%s and %s=%s" % (x_feature1, ix2feature1[c], x_feature2, ix2feature2[r]) ) )
                else: warnings.warn( "The following warnings will be wrapped..." )
                show_nwarnings -= 1
                continue

            g[r,c] = dm[i1 & i2][y_feature_name].item() * sign

    plot_primitives.plot_2D(xgrid1, xgrid2, g, **kwargs)            
    plt.colorbar(label=kwargs.get("label", y_feature_name))   

    plt.xlabel(x_feature1);
    plt.ylabel(x_feature2);
    
  

###############################################################################


def slice_and_plot_df1D(df, x_feature_axis, x_position, x_features, 
                        y_feature_name="obj", sign=+1, **kwargs):
    """
        Args:
            x_features denotes the order of features in x_position.
    """
    x_position = x_position[:len(x_features)] # if there is task on the last postion: skip it

    df_slice, x_position_matched = extract_slice(df, x_features, x_position, [x_feature_axis])
    if len(df_slice)<=0:  warnings.warn("[slice_and_plot_df1D] Empty slice!")

    position_label = ", ".join("%s=%s" % (n,v1) for (n, v1) in 
                               zip(x_features, x_position_matched) if n!=x_feature_axis)    
    label = kwargs.pop("label", "true (%s)" % position_label)

    return plot_df1D(df_slice, x_feature_axis, y_feature_name=y_feature_name, sign=sign, label=label, **kwargs)


def slice_and_plot_df2D(df, plt_feature1, plt_feature2, x_position, x_features,
                        y_feature_name="obj", kind="grid", sign=+1, label="",
                        cmap="Reds", vmin=None, vmax=None, **kwargs):
    """ Creates and plots 2D slice of means (for example over different runs) 
        from precomputed data frame.

        Args:
            x_features denotes the order of features in x_position.
    """
    x_position = x_position[:len(x_features)] # if there is task on the last postion: skip it    

    df_slice, x_position_matched = extract_slice(df, x_features, x_position, [plt_feature1, plt_feature2])
    if len(df_slice)<=0:  warnings.warn("[slice_and_plot_df2D] Empty slice!")

    return plot_df2D(df_slice, plt_feature1, plt_feature2, 
              y_feature_name=y_feature_name, kind=kind, 
              sign=sign, label=label, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)



