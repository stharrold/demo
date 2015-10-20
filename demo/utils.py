#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Utilities for predictive analytics demo.

"""


# Import standard packages.
# Import __future__ for Python 2x backward compatibility.
from __future__ import absolute_import, division, print_function
import os
import pdb
import subprocess
import time
# Import installed packages.
from IPython.display import display, SVG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.cross_validation as sklearn_cval
import sklearn.grid_search as sklearn_gs
import sklearn.metrics as sklearn_met
import sklearn.tree as sklearn_tree


def plot_feature_importances(model, train_features):
    r"""Plot feature importances.
    
    Args:
        model (sklearn): Model from `sklearn`.
            Model must have already been fit.
        train_features (pandas.DataFrame): The data that was used to train
            the model.
    
    Returns:
        None
        
    Notes:
        * Feature importance is determined by the expected fraction
            of samples that the feature contributes to [1]_.
    
    References:
        .. [1] http://scikit-learn.org/stable/modules/
               ensemble.html#feature-importance-evaluation
    
    """
    df_importances = pd.DataFrame.from_records(
        data=zip(model.feature_importances_, train_features.columns),
        columns=['importance', 'feature'])
    df_importances.sort_values(
        by=['importance'], ascending=False, inplace=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Feature vs importance")
    ax.set_xlabel("Importance (mean over all trees)")
    ax.set_ylabel("Feature")
    sns.barplot(
        x='importance', y='feature', data=df_importances,
        color=sns.color_palette()[0], ax=ax)
    plt.show()
    return None


def plot_actual_vs_predicted(y_true, y_pred, return_ax=False):
    r"""Plot actual values vs predicted values from fit.
    
    Args:
        y_true (pandas.DataFrame): The true target values.
        y_pred (pandas.DataFrame): The predicted target values.
        return_ax (bool, optional, default=False):
            If `False` (default), show the plot. Return `None`.
            If `True`, return a `matplotlib.axes` instance
            for additional modification.

    
    Returns:
        ax (matplotlib.axes): Returned if `return_ax` is `True`.
            Otherwise returns `None`.
    
    """
    # TODO: Plot binned percentiles; Q-Q plot
    # TODO: Z1,Z2 gaussianity measures
    # Check input.
    # TODO: limit number of points to plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Actual vs predicted values")
    ax.set_xlabel("Predicted values")
    ax.set_ylabel("Actual values")
    # TODO: Use hexbins for density.
    #sns.jointplot(
    #    x=y_pred, y=y_true, kind='hex', stat_func=None,
    #    label='(predicted, actual)')
    ax.plot(
        y_pred, y_true, color=sns.color_palette()[0],
        marker=',', linestyle='', alpha=0.5, label='(predicted, actual)')
    y_pred_extrema = [min(y_pred), max(y_pred)]
    ax.plot(
        y_pred_extrema, y_pred_extrema, color=sns.color_palette()[1],
        marker='', linestyle='-', linewidth=1, label='(predicted, predicted)')
    ax.legend(loc='upper left', title='values')
    if return_ax:
        return_obj = ax
    else:
        plt.show()
        return_obj = None
    return return_obj


def search_models(
    model, param_dists, train_features, train_pred_true, test_features,
    n_iter=10, scoring='r2', cv=5, n_jobs=-1, pred_column='prediction',
    file_path='', file_basename='', show_svg=False):
    r"""Search hyper-parameters for best regression model and report.
    
    Args:
        model (sklearn): Regression model from `sklearn`.
        param_dists (dict): Dictionary with keys as string parameter names
            and values as `scipy.stats` distributions [1]_ or lists.
        train_features (pandas.DataFrame): The data to be fit for training.
        train_pred_true (pandas.Series): The target relative to `train_features`
            for regression.
        test_features (pandas.DataFrame): The data to be fit for testing
            as a final evaluation.
        n_iter (int, optional, default=10): Number of parameter settings 
            that are sampled. Trades off runtime vs quality of the solution.
        scoring (str, optional, default='r2'): Scoring function.
            Default is R^2, coefficient of determination.
            See scikit-learn model evaluation documentation [2]_.
        cv (int, optional, default=5): Number of folds for K-fold cross-validation.
        n_jobs (int, optional, default=-1): The number of CPUs to use to do the
            computation. -1 is all CPUs.
        pred_column (str, optional, default='prediction'): Name for output
            prediction column in CSV.
        file_path (str, optional, default=''): Path for generated files.
        file_basename (str, optional, default=''): Base name for generated files.
        show_svg (bool, optional, default=False): Show SVG plot.
    
    Returns:
        None
    
    Raises:
        ValueError
    
    See Also:
        sklearn.grid_search.RandomizedSearchCV
        
    References:
    .. [1] http://docs.scipy.org/doc/scipy/reference/stats.html
    .. [2] http://scikit-learn.org/stable/modules/model_evaluation.html
    
    """
    # TODO: move to demo/main.py as a top-level script.
    # TODO: outliers by Bonferroni correcte p-values
    # TODO: outliers by prediction distribution
    # Check input.
    if not isinstance(train_features, pd.DataFrame):
        raise ValueError("`train_features` must be a `pandas.DataFrame`")
    if not isinstance(train_pred_true, pd.Series):
        raise ValueError("`train_pred_true` must be a `pandas.Series`")
    if not isinstance(test_features, pd.DataFrame):
        raise ValueError("`test_features` must be a `pandas.DataFrame`")
    # Search for best model and report.
    search = sklearn_gs.RandomizedSearchCV(
        estimator=model, param_distributions=param_dists,
        n_iter=n_iter, scoring=scoring, n_jobs=n_jobs, cv=cv)
    time_start = time.time()
    search.fit(X=train_features, y=train_pred_true)
    time_stop = time.time()
    print(("Elapsed search time (seconds) = {elapsed:.1f}").format(
        elapsed=time_stop-time_start))
    model_best = search.best_estimator_
    print(("Best params = {params}").format(params=search.best_params_))
    grid_best = max(
        search.grid_scores_,
        key=lambda elt: elt.mean_validation_score)
    if not np.isclose(search.best_score_, grid_best.mean_validation_score):
        raise AssertionError(
            "Program error. Max score from `search.grid_scores_` was not found correctly.")
    print(("Best score (R^2) = {mean:.4f} +/- {std:.4f}").format(
            mean=grid_best.mean_validation_score,
            std=np.std(grid_best.cv_validation_scores)))
    train_pred_best = sklearn_cval.cross_val_predict(
        estimator=model_best, X=train_features, y=train_pred_true,
        cv=cv, n_jobs=n_jobs)
    print("Score from best model training predictions (R^2) = {score:.4f}".format(
            score=sklearn_met.r2_score(y_true=train_pred_true, y_pred=train_pred_best)))
    train_pred_default = sklearn_cval.cross_val_predict(
        estimator=model, X=train_features, y=train_pred_true, cv=cv, n_jobs=n_jobs)
    print("Score from default model training predictions (R^2) = {score:.4f}".format(
            score=sklearn_met.r2_score(y_true=train_pred_true, y_pred=train_pred_default)))
    if hasattr(model_best, 'feature_importances_'):
        print("Plot feature importances from best model:")
        plot_feature_importances(
            model=model_best, train_features=train_features)
    print("Plot actual vs predicted values from best model:")
    plot_actual_vs_predicted(y_true=train_pred_true, y_pred=train_pred_best)
    # Create predictions for `test_features`.
    # Order by index, save as CSV, and graph.
    test_pred_best = model_best.predict(X=test_features)
    file_csv = r'predictions_{name}.csv'.format(name=file_basename)
    path_csv = os.path.join(file_path, file_csv)
    print("Predictions CSV file =\n{path}".format(path=path_csv))
    df_csv = pd.DataFrame(
        data=test_pred_best, index=test_features.index,
        columns=[pred_column]).sort_index()
    df_csv.to_csv(path_or_buf=path_csv, header=True, index=True, quoting=None)
    if hasattr(model_best.estimators_[0], 'tree_'):
        file_dot = r'graph_{name}.dot'.format(name=file_basename)
        path_dot = os.path.join(file_path, file_dot)
        print("Graphviz dot and SVG files =\n{path}\n{path}.svg".format(path=path_dot))
        sklearn_tree.export_graphviz(
            decision_tree=model_best.estimators_[0], out_file=path_dot,
            feature_names=test_features.columns)
        cmd = ['dot', '-Tsvg', path_dot, '-O']
        # Use pre-Python 3.5 subprocess API for backward compatibility.
        subprocess.check_call(args=cmd)
        if show_svg:
            display(SVG(filename=path_dot+'.svg'))
    return None
