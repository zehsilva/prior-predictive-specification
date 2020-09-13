# Learning hyperparameters of Bayesian models by matching moments of prior predictive distribution.


## Pre-installation Requirements

The code was tested using Python 3.7.4 from Anaconda 2019.10.
with TensorFlow 2.1 and TensorFlow Probability 0.9.0.
It uses numpy, pandas, seaborn, and matplotlib.


## Data

The hetrec-lastfm dataset along with train-test split can be found in the directory named ``data``.


## Code illustrating gradient-based optimization

The code illustrating gradient-based optimization can be found in the directory named ``gradient_optimization``.

### Main files 
  * ``pmf_sgd_optimization.ipynb``  - Jupter Notebook illustrating how priors matching requested values of prior predictive expectation and/or variance can be found for Poisson Matrix Factorization (PMF) model using SGD.
  * ``hpf_sgd_optimization.ipynb``  - Jupter Notebook illustrating how priors matching requested values of prior predictive expectation and/or variance can be found for Hierarchical Poisson Matrix Factorization (HPF) model using SGD.
  * ``pmf_estimators_analysis.ipynb``  - Jupter Notebook illustrating bias and variance of the estimators used in pmf_sgd_optimization.ipynb for PMF model.
  * ``pmf_surface_visualizations.ipynb``  - Jupter Notebook illustrating 1D and 2D projections of optimization space for the problem of matching Poisson Matrix Factorization (PMF) prior predicitve distribution variance (minimization of the discrepancy=(Variance-100)^2 ). We consider two parametrizations: abcd vs mu-sgima. 

### Sampling code 
  * ``pmf_model.py``  - Methods calculating E[Y] and E[Y^2] (and therefore also Var[Y]) over prior predictive distribution for Poisson Matrix Factorization.
  * ``hpf_model.py``  - Methods calculating E[Y] and E[Y^2] (and therefore also Var[Y]) over prior predictive distribution for Hierarchical Poisson Matrix Factorization.

### Additional files 
  * ``aux.py``, ``aux_plt.py``, ``boplotting/*``  - Auxiliary functions for tensor processing and plotting.


## Experiments illustrating convergance of gradient-based optimization

The code can be found in ``gradient_optimization_experiments``.
It contains two subfolders ``PMF_Convergence`` and ``HPF_Convergence``.


## Visualization of PMF posterior quality using PSIS-LOO

The code computing PSIS-LOO on the test subset of fitted PMF can be found in ``posterior_visualization``. The scripts pmf_precompute_objectives_posterior.py and python pmf_precompute_objectives_posterior2.py precompute certain set of configurations specified inside those files and write to respectively pmf_precompute_objectives_posterior.py.csv and pmf_precompute_objectives_posterior2.py.csv. The outputs can be then previewed with VISUALIZATION.ipynb and VISUALIZATION2.ipynb.
VISUALIZATION_K.ipynb plots PSIS-LOO on test subset for various K with a,b,c,d set to prior optimal values.


## Comparison of Bayesian optimization of PSIS-LOO 

The code can be found in ``bo_optimization``. To run the experiment use: ``RUN_EXPERIMENT_BO.sh``. 
It requires RoBO - a Robust Bayesian Optimization framework (https://github.com/automl/RoBO) to be preinstalled.
Results can be displayed using the Jupter Notebook VISUALIZATION_BO.ipynb.

## Sensitivity to Model Mismatch 

The code can be found in ``sensitivity analysis`` folder. 
To visualize the experiment results open the jupyter-notebook ``sensitivity_analysis.ipynb``.
To re-run the experiment run the python scripts
	- ``python poisson_prior_exp_negbin_async.py``: experiment sampling from a Negative Binomial
	- ``python poisson_prior_exp_binomial_async.py``: experiment sampling from a PMF but with a probability of randomly zeroing each of the entries of the matrix.
Both experiments will generate csv files with the results. The files with a suffix ``final_*.csv`` can be analyzed in the ``sensitivity_analysis.ipynb`` notebook simply by adding new cells, keeping the same code from previous cells and just adjusting the file name that is loaded.
