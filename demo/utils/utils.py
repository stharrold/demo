#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Utilities for predictive analytics demo.

"""


# Import standard packages.
import collections
import inspect
import os
import shelve
import subprocess
import sys
import tempfile
import textwrap
import time
import warnings
# Import installed packages.
import astroML.density_estimation as astroML_dens
from IPython.display import display, SVG
import geopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn.cluster as sk_cl
import sklearn.cross_validation as sk_cv
import sklearn.grid_search as sk_gs
import sklearn.metrics as sk_met
import sklearn.tree as sk_tree


__all__ = [
    'calc_feature_importances',
    'calc_score_pvalue',
    'calc_silhouette_scores',
    'Container',
    'plot_actual_vs_predicted',
    'plot_feature_importances',
    'plot_stacked_timeseries_histogram',
    'rolling_window',
    'search_models']


def calc_feature_importances(
    estimator,
    df_features:pd.DataFrame, ds_target:pd.Series, ds_weight:pd.Series=None, 
    size_sub:int=None, replace:bool=True, dists:list=None, 
    show_progress:bool=False, show_plot:bool=False) -> pd.DataFrame:
    r"""Calculate feature importances and compare to random added features
    for weighted data sets.
    
    Args:
        estimator: Estimator model from `sklearn` with attributes `fit`
            and, after running `fit`, also `feature_importances_`.
            See 'Raises', 'Notes'.
        df_features (pandas.DataFrame): Data frame of feature values.
            Format: rows=records, cols=features.
        ds_target (pandas.Series): Data series of target values.
            Format: rows=records, col=target.
        ds_weight (pandas.Series, optional, None): Data series of record
            weights. Format: rows=records, col=weight.
            Default: `None`, then all set to 1.
        size_sub (int, optional, None): Number of records in subset for
            selecting features.
            Default: `None`, then min(10K, all) records are used.
            The larger `size_sub` the more features that will become
            significant. See 'Raises'.
        replace (bool, optional, True): Sample records with replacement
            (bootstrap sampling). Default: `True`.
            Use `replace=True` to reduce overfitting the feature importances.
        dists (list, optional, None): List of random distributions to add into
            `df_features` to compare significance of feature importance.
            List items are evaluated with `eval` then appended to `df_features`.
            Default: `None`, then uses distributions Laplace, logistic,
            lognormal, Pareto, Poisson, Rayleigh, standard Cauchy,
            standard exponential, standard normal, uniform.
        show_progress (bool, optional, False): Print status.
        show_plot (bool, optional, False): Show summary plot of max top 20 
            significant feature importances and the random importance.
        
    Returns:
        df_importances (pandas.DataFrame): Data frame of feature importances.
            Format: rows=iterations, cols=features+'random'.
          
    Raises:
        ValueError:
            * Raised if not `hasattr(estimator, 'fit')`
        RuntimeWarning:
            * Raised if not `size_sub <= len(df_features)`.
    
    Notes:
        * Feature importance is the normalized reduction in the loss score.
            See the `sklearn` documentation for your estimator and
            the estimator's 'feature_importances_' attribute.
    
    """
    # TODO: Replace show_progress and warnings.warn with logger.[debug,warn]
    #     https://github.com/stharrold/stharrold.github.io/issues/58
    # Check arguments.
    # Note: Copy df_features to avoid modifying input data.
    if not hasattr(estimator, 'fit'):
        raise ValueError(
            ("`estimator` must have the attribute 'fit'.\n" +
             "Required: hasattr(estimator, 'fit')"))
    if (size_sub is not None) and not (size_sub <= len(df_features)):
        warnings.warn(
            ("The number of records in the subset for calculating feature\n" +
             "importances is larger than the number of records in the data.\n" +
             "Suggested: size_sub <= len(df_features)\n" +
             "Given: {lhs} <= {rhs}").format(
                 lhs=size_sub, rhs=len(df_features)),
             RuntimeWarning)
    df_ftrs_rnd = df_features.copy()
    size_data = len(df_features)
    if size_sub is None:
        size_sub = min(int(1e4), size_data)
    dists = [
        'np.random.laplace(loc=0.0, scale=1.0, size=size_data)',
        'np.random.logistic(loc=0.0, scale=1.0, size=size_data)',
        'np.random.lognormal(mean=0.0, sigma=1.0, size=size_data)',
        'np.random.pareto(a=1.0, size=size_data)',
        'np.random.poisson(lam=1.0, size=size_data)',
        'np.random.rayleigh(scale=1.0, size=size_data)',
        'np.random.standard_cauchy(size=size_data)',
        'np.random.standard_exponential(size=size_data)',
        'np.random.standard_normal(size=size_data)',
        'np.random.uniform(low=-1.0, high=1.0, size=size_data)']
    # Include different of randomized features and evaluate their importance,
    # one at a time.
    ftrs_imps = collections.defaultdict(list)
    if show_progress:
        print("Progress:", end=' ')
    for (inum, dist) in enumerate(dists):
        df_ftrs_rnd['random'] = eval(dist)
        idxs_sub = np.random.choice(a=size_data, size=size_sub, replace=replace)
        if ds_weight is None:
            estimator.fit(
                X=df_ftrs_rnd.values[idxs_sub], y=ds_target.values[idxs_sub])            
        else:
            estimator.fit(
                X=df_ftrs_rnd.values[idxs_sub], y=ds_target.values[idxs_sub],
                sample_weight=ds_weight.values[idxs_sub])
        for (col, imp) in zip(
            df_ftrs_rnd.columns, estimator.feature_importances_):
            ftrs_imps[col].append(imp)
        if show_progress:
            print("{frac:.0%}".format(frac=(inum+1)/len(dists)), end=' ')
    if show_progress:
        print('\n')
    # Return the feature importances and plot the 20 most important features.
    df_importances = pd.DataFrame.from_dict(ftrs_imps)
    if show_plot:
        ds_ftrs_imps_mean = df_importances.mean().sort_values(ascending=False)
        tfmask = (ds_ftrs_imps_mean > df_importances['random'].mean())
        ftrs_plot = list(ds_ftrs_imps_mean[tfmask].index[:20])+['random']
        sns.barplot(
            data=df_importances[ftrs_plot], order=ftrs_plot, ci=95,
            orient='h', color=sns.color_palette()[0])
        plt.title(
            ("Feature column name vs top 20 importance scores\n" +
             "with 95% confidence interval and benchmark randomized scores"))
        plt.xlabel(
            ("Importance score\n" +
             "(normalized reduction of loss function)"))
        plt.ylabel("Feature column name")
        plt.show()
    return df_importances


def calc_score_pvalue(
    estimator,
    df_features:pd.DataFrame, ds_target:pd.Series, ds_weight:pd.Series=None,
    n_iter:int=20, frac_true:float=0.2, size_sub:int=None, frac_test:float=0.2,
    replace:bool=True, show_progress:bool=False, show_plot:bool=False) -> float:
    r"""Calculate the p-value of the scored predictions for weighted data sets.
    
    Args:
        estimator: Estimator model from `sklearn` with attributes `fit`
            and, after running `fit`, also `score`. See 'Raises', 'Notes'.
        df_features (pandas.DataFrame): Data frame of feature values.
            Format: rows=records, cols=features.
        ds_target (pandas.Series): Data series of target values.
            Format: rows=records, col=target.
        ds_weight (pandas.Series, optional, None): Data series of record
            weights. Format: rows=records, col=weight.
            Default: `None`, then all set to 1.
        n_iter (int, optional, 20): Number of iterations for calculating scores.
        frac_true (float, optional, 0.2): Proportion of `n_iter` for which the
            target values are not shuffled. Must be between 0 and 1.
            See 'Raises'.
        size_sub (int, optional, None): Number of records in subset for
            cross-validating scores. Only enough records need to be used to
            approximate the variance in the data.
            Default: `None`, then min(10K, all) records are used. See 'Raises'.
        frac_test (float, optional, 0.2): Proportion of `size_sub` for which to
            test the predicted target values and calculate each score.
        replace (bool, optional, True): Sample records with replacement
            (bootstrap sampling). Default: `True`.
            Use `replace=True` to reduce overfitting the scores.
        show_progress (bool, optional, False): Print status.
        show_plot (bool, optional, False): Show summary plot
            of score significance.
    
    Returns:
        pvalue
    
    Raises:
        ValueError:
            * Raised if not `hasattr(estimator, 'fit')`
            * Raised if not `0.0 < frac_true < 1.0`.
        RuntimeWarning:
            * Raised if not `size_sub <= len(df_features)`.
    
    See Also:
        sklearn.cross_validation.train_test_split
    
    Notes:
        * The significance test is calculated by sampling the differences
            between the score means then shuffling the labels for whether or
            not the target values were themselves shuffled.
    
    References:
        [^sklearn]: http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html
    """
    # TODO: Replace show_progress and warnings.warn with logger.[debug,warn]
    #     https://github.com/stharrold/stharrold.github.io/issues/58
    # Check arguments.
    if not hasattr(estimator, 'fit'):
        raise ValueError(
            ("`estimator` must have the attribute 'fit'.\n" +
             "Required: hasattr(estimator, 'fit')"))
    if not 0.0 < frac_true < 1.0:
        raise ValueError(
            ("`frac_true` must be between 0 and 1.\n" +
             "Required: 0.0 < frac_true < 1.0\n" +
             "Given: frac_true={frac_true}").format(frac_true=frac_true))
    if (size_sub is not None) and not (size_sub <= len(df_features)):
        warnings.warn(
            ("The number of records in the subset for calculating feature\n" +
             "importances is larger than the number of records in the data.\n" +
             "Suggested: size_sub <= len(df_features)\n" +
             "Given: {lhs} <= {rhs}").format(
                 lhs=size_sub, rhs=len(df_features)),
             RuntimeWarning)
    size_data = len(df_features)
    if size_sub is None:
        size_sub = min(int(1e4), size_data)
    # Score with/without shuffling the target values.
    inum_score_isshf = dict()
    imod = round(n_iter*frac_true)
    if show_progress:
        print("Progress:", end=' ')
    for inum in range(0, n_iter):
        inum_score_isshf[inum] = dict()
        idxs_sub = np.random.choice(a=size_data, size=size_sub, replace=replace)
        if inum % imod == 0:
            # Every 1 out of imod times, use true target values
            # and show progress.
            inum_score_isshf[inum]['is_shf'] = False
            trg_vals = ds_target.values[idxs_sub]
            if show_progress:
                print("{frac:.0%}".format(frac=(inum+1)/n_iter), end=' ')
        else:
            # Otherwise with randomly permuted target values.
            inum_score_isshf[inum]['is_shf'] = True
            trg_vals = np.random.permutation(ds_target.values[idxs_sub])
        if ds_weight is None:
            (ftrs_train, ftrs_test,
             trg_train, trg_test) = sk_cv.train_test_split(
                df_features.values[idxs_sub], trg_vals, test_size=frac_test)
            estimator.fit(X=ftrs_train, y=trg_train)
            inum_score_isshf[inum]['score'] = estimator.score(
                X=ftrs_test, y=trg_test)
        else:
            (ftrs_train, ftrs_test,
             trg_train, trg_test,
             pwt_train, pwt_test) = sk_cv.train_test_split(
                df_features.values[idxs_sub], trg_vals,
                ds_weight.values[idxs_sub], test_size=frac_test)
            estimator.fit(X=ftrs_train, y=trg_train, sample_weight=pwt_train)
            inum_score_isshf[inum]['score'] = estimator.score(
                X=ftrs_test, y=trg_test, sample_weight=pwt_test)
    if show_progress:
        print('\n')
    df_scores = pd.DataFrame.from_dict(data=inum_score_isshf, orient='index')
    # Plot the distributions of model scores with/without
    # shuffling the target values.
    if show_plot:
        sns.distplot(
            df_scores.loc[df_scores['is_shf'], 'score'],
            hist=True, kde=True, norm_hist=True, color=sns.color_palette()[0],
            label='scores with shuffled target values')
        sns.distplot(
            df_scores.loc[np.logical_not(df_scores['is_shf']), 'score'],
            hist=True, kde=True, norm_hist=True, color=sns.color_palette()[1],
            label='scores with actual target values')
        plt.title(
            "Probability density functions of model scores\n" +
            "by whether or not target values were permuted")
        plt.xlabel("Model score")
        plt.ylabel("Probability density")
        plt.legend(loc='upper left')
        plt.show()
        print("Average model score with shuffling: {score:.3f}".format(
                score=df_scores.loc[df_scores['is_shf'], 'score'].mean()))
        print("Average model score without shuffling: {score:.3f}".format(
                score=df_scores.loc[np.logical_not(df_scores['is_shf']), 'score'].mean()))
    # Calculate the distribution of differences in score means with/without
    # shuffling the target values.
    # Distribution of randomized differences in score means:
    rnd_mean_score_diffs = list()
    for _ in range(int(1e4)):
        tfmask_shf = np.random.permutation(df_scores['is_shf'].values)
        tfmask_notshf = np.logical_not(tfmask_shf)
        rnd_mean_score_diffs.append(
            df_scores.loc[tfmask_notshf, 'score'].mean()
            - df_scores.loc[tfmask_shf, 'score'].mean())
    # Distribution of actual differences in score means:
    tfmask_shf = df_scores['is_shf'].values
    tfmask_notshf = np.logical_not(tfmask_shf)
    mean_score_diff = (
        df_scores.loc[tfmask_notshf, 'score'].mean()
        - df_scores.loc[tfmask_shf, 'score'].mean())
    # Plot the distribution of differences in score means.
    if show_plot:
        sns.distplot(rnd_mean_score_diffs, hist=True, kde=True, norm_hist=True,
            color=sns.color_palette()[0],
            label=(
                'mean score differences assuming\n' +
                'no distinction between shuffled/unshuffled'))
        plt.axvline(mean_score_diff,
            color=sns.color_palette()[1], label='actual mean score difference')
        plt.title(
            "Differences between mean model scores\n" +
            "by whether or not target values were actually shuffled")
        plt.xlabel("Model score difference")
        plt.ylabel("Probability density")
        plt.legend(loc='upper left')
        plt.show()
    # Return the p-value and describe the statistical significance.
    pvalue = 100.0 - scipy.stats.percentileofscore(
        a=rnd_mean_score_diffs, score=mean_score_diff, kind='mean')
    if show_plot:
        print(
            ("Null hypothesis: There is no distinction in the differences\n" +
             "between the mean model scores whether or not the target\n" +
             "values have been shuffled.\n" +
             "Outcome: Assuming the null hypothesis, the probability of\n" +
             "obtaining a difference between the mean model scores at least\n" +
             "as great as {diff:.2f} is {pvalue:.1f}%.").format(
                 diff=mean_score_diff, pvalue=pvalue))
    return pvalue


def calc_silhouette_scores(
    df_features:pd.DataFrame, n_clusters_min:int=2, n_clusters_max:int=10,
    size_sub:int=None, n_scores:int=10,
    show_progress:bool=False, show_plot:bool=False) -> list:
    r"""Plot silhouette scores for determining number of clusters in k-means.
    
    Args:
        df_features (pandas.DataFrame): Data frame of feature values.
            Format: rows=records, cols=features.
            Note: `df_features` should aleady be scaled
            (e.g. sklearn.preprocessing.RobustScaler)
        n_clusters_min (int, optional, 2): Minimum number of clusters.
        n_clusters_max (int, optional, 10): Maximum number of clusters.
        size_sub (int, optional, None): Number of records in subset for
            calculating scores. See 'Notes', 'Raises'.
            Default: `None`, then min(1K, all) records are used.
        n_scores (int, optional, 10): Number of scores to calculate
            for each cluster. See 'Notes'.
        show_progress (bool, optional, False): Print status.
        show_plot (bool, optional, False): Show summary plot of scores.
        
    Returns:
        nclusters_scores (list): List of tuples (n_clusters, scores)
            where n_clusters is the number of clusters and
            scores are the calclated silhouette scores.
            Note: `sklearn.metrics.silhouette_score` may fail if cluster sizes
            are strongly imbalanced. In these cases,
            `len(scores) < n_scores` and the shape of `nclusters_scores` is
            irregular.
    Raises:
        ValueError:
            * Raised if not `2 <= n_clusters_min < n_clusters_max`.
        RuntimeWarning:
            * Raised if not `size_sub <= len(df_features)`.
    
    See Also:
        sklearn.cluster.MiniBatchKMeans,
        sklearn.metrics.silhouette_score
    
    Notes:
        * Silhouette scores are a measure comparing the relative size and
            proximity of clusters. Interpretation from [^sklearn]:
            "The best value is 1 and the worst value is -1. Values near 0
            indicate overlapping clusters. Negative values generally indicate
            that a sample has been assigned to the wrong cluster,
            as a different cluster is more similar."
        * For better score estimates, often it's more efficient to increase
            n_scores rather than size_sub since
            `sklearn.metrics.silhouette_score` creates a size_sub**2 matrix
            in RAM.
        * `sklearn.metrics.silhouette_score` may fail if cluster sizes
            are strongly imbalanced. In these cases,
            `len(scores) < n_scores` and the shape of `nclusters_scores` is
            irregular.
    
    References:
        [^sklearn] http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
    
    """
    # TODO: Replace show_progress and warnings.warn with logger.[debug,warn]
    #     https://github.com/stharrold/stharrold.github.io/issues/58
    # Check arguments.
    if not (2 <= n_clusters_min < n_clusters_max):
        raise ValueError(
            ("The number of clusters is not valid.\n" +
             "Required: 2 <= n_clusters_min < n_clusters_max\n" +
             "Given: 2 <= {nmin} < {nmax}").format(
                 nmin=n_clusters_min, nmax=n_clusters_max))
    if (size_sub is not None) and not (size_sub <= len(df_features)):
        warnings.warn(
            ("The number of records in the subset for calculating the\n" +
             "silhouette scores is larger than the number of records\n" +
             "in the data.\n" +
             "Suggested: size_sub <= len(df_features)\n" +
             "Given: {lhs} <= {rhs}").format(
                 lhs=size_sub, rhs=len(df_features)),
             RuntimeWarning)
    if size_sub is None:
        size_sub = min(int(1e3), len(df_features))
    # Estimate silhouette scores for each number of clusters.
    if show_progress:
        print("Progress:", end=' ')
    nclusters_scores = list()
    num_clusters = (n_clusters_max - n_clusters_min) + 1
    for n_clusters in range(n_clusters_min, n_clusters_max+1):
        transformer_kmeans = sk_cl.MiniBatchKMeans(n_clusters=n_clusters)
        labels_pred = transformer_kmeans.fit_predict(X=df_features)
        scores = list()
        n_fails = 0
        while len(scores) < n_scores:
            try:
                scores.append(
                    sk_met.silhouette_score(
                        X=df_features,
                        labels=labels_pred,
                        sample_size=size_sub))
            except ValueError:
                n_fails += 1
            if n_fails > 10*n_scores:
                warnings.warn(
                    ("`sklearn.silhouette_score` failed for given data with:\n" +
                     "n_clusters = {ncl}\n" +
                     "size_sub = {size}\n").format(
                         ncl=n_clusters, size=size_sub))
                break
        nclusters_scores.append((n_clusters, scores))
        if show_progress:
            print("{frac:.0%}".format(
                    frac=(n_clusters-n_clusters_min+1)/num_clusters),
                  end=' ')
    if show_progress:
        print('\n')
    # Plot silhouette scores vs number of clusters.
    if show_plot:
        nclusters_pctls = np.asarray(
            [np.append(tup[0], np.percentile(tup[1], q=[5,50,95]))
             for tup in nclusters_scores])
        plt.plot(
            nclusters_pctls[:, 0], nclusters_pctls[:, 2],
            marker='.', color=sns.color_palette()[0],
            label='50th pctl score')
        plt.fill_between(
            nclusters_pctls[:, 0],
            y1=nclusters_pctls[:, 1],
            y2=nclusters_pctls[:, 3],
            alpha=0.5, color=sns.color_palette()[0],
            label='5-95th pctls of scores')
        plt.title("Silhouette score vs number of clusters")
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette score")
        plt.legend(loc='lower left')
        plt.show()
    return nclusters_scores


class Container:
    r"""Empty class to contain dynamically allocated attributes.
    Use to minimize namespace clutter from variable names.
    Use for heirarchical data storage like a `dict`.
    
    Example:
        ```
        data = Container()
        data.features = features
        data.target = target
        ```

    """
    pass


def plot_actual_vs_predicted(
    y_true:np.ndarray, y_pred:np.ndarray, loglog:bool=False, xylims:tuple=None,
    path:str=None) -> None:
    r"""Plot actual vs predicted values.
    
    Args:
        y_true (numpy.ndarray): The true target values.
        y_pred (numpy.ndarray): The predicted target values.
        loglog (bool, optional, False): Log scale for both x and y axes.
        xylims (tuple, optional, None): Limits for x and y axes.
            Default: None, then set to (min, max) of `y_pred`.
        path (str, optional, None): Path to save figure.
    
    Returns:
        None
    
    """
    # TODO: Plot binned percentiles; Q-Q plot
    # TODO: Z1,Z2 gaussianity measures
    # Check input.
    # TODO: limit number of points to plot
    # TODO: Use hexbins for density.
    #   sns.jointplot(
    #       x=y_pred, y=y_true, kind='hex', stat_func=None,
    #       label='(predicted, actual)')
    plt.title("Actual vs predicted values")
    if loglog:
        plot_func = plt.loglog
        y_pred_extrema = (min(y_pred[y_pred > 0]), max(y_pred))
    else:
        plot_func = plt.plot
        y_pred_extrema = (min(y_pred), max(y_pred))
    if xylims is not None:
        y_pred_extrema = xylims
    plot_func(
        y_pred, y_true, color=sns.color_palette()[0],
        marker='.', linestyle='', alpha=0.05, label='(predicted, actual)')
    plot_func(
        y_pred_extrema, y_pred_extrema, color=sns.color_palette()[1],
        marker='', linestyle='-', linewidth=1, label='(predicted, predicted)')
    plt.xlabel("Predicted values")
    plt.xlim(y_pred_extrema)
    plt.ylabel("Actual values")
    plt.ylim(y_pred_extrema)
    plt.legend(loc='upper left', title='values')
    if path is not None:
        plt.savefig(path)
    plt.show()
    return None


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


def plot_stacked_timeseries_histogram(
    total_counts, itemized_counts=None, ax=None):
    r"""Create a time series histogram with stacked counts labeled by category.
    Convenience function for methods from
    `astroML.density_estimation.bayesian_blocks`.
    
    Args:
        total_counts (collections.Counter): Total counts by time.
            Example: total_counts.items() = [(1, 5), (2, 4), ...]
                where day 1 had 5 total counts, day 2 had 4 total counts...
        itemized_counts (optional, dict): `dict` of `collections.Counter`.
            If `None` (default), histogram is not stacked.
            Keys: `hashable` label for each type of event. To preserve
                key order, use `collections.OrderedDict`.
            Values: `collections.Counter` counts by time.
            Example: itemized_counts = dict(a=counter_a, b=counter_b)
                where counter_a.items() = [(1, 1), (2, 1), ...]
                and   counter_b.items() = [(1, 4), (2, 3), ...]
            Required: The `total_counts` must equal the sum of all
                `itemized_counts`
        ax (optional, matplotlib.Axes): Axes instance on which to add the plot.
            If `None` (default), an axes instance is created.
    
    Returns:
        ax (matplotlib.axes): Axes instance for the plot.

    Raises:
        AssertionError:
            If `total_counts` does not equal the sum of all `itemized_counts`.

    See Also:
        astroML.density_estimation.bayesian_blocks

    Notes:
        * This simple implementation assumes that the times are not regularly
            spaced and that the data are counts of events.
        * Example call with ax=`None`:
            ax = plot_stacked_timeseries_histogram(
                total_counts=total_counts,
                itemized_counts=itemized_counts,
                ax=None)
            ax.legend(loc='upper left')
            plt.show(ax)
        * Example call with ax defined:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax = plot_stacked_timeseries_histogram(
                total_counts=total_counts,
                itemized_counts=itemized_counts,
                ax=ax)
            ax.legend(loc='upper left')
            plt.show(ax)

    """
    # Check input.
    if itemized_counts is not None:
        summed_itemized_counts = collections.Counter()
        for key in itemized_counts.keys():
            summed_itemized_counts.update(itemized_counts[key])
        if not total_counts == summed_itemized_counts:
            raise AssertionError(
               "`total_counts` must equal the sum of all `itemized_counts`.")
    # Calculate histogram bins.
    (times, counts) = zip(*total_counts.items())
    bin_edges = astroML_dens.bayesian_blocks(
        t=times, x=counts, fitness='events')
    # Create plot.
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if itemized_counts is None:
        ax.hist(list(total_counts.elements()),
                bins=bin_edges, stacked=False, rwidth=1.0, label=None,
                color=sns.color_palette()[0])
    else:
        keys = itemized_counts.keys()
        ax.hist([list(itemized_counts[key].elements()) for key in keys],
                bins=bin_edges, stacked=True, rwidth=1.0, label=keys,
                color=sns.husl_palette(n_colors=len(keys)))
    return ax


def rolling_window(
    arr, window):
    r"""Efficient rolling window for statistics. From [1]_.

    Args:
        arr (numpy.ndarray) 
        window (int): Number of elements in window.

    Returns:
    arr_windowed (numpy.ndarray): `arr` where each element has been replaced
        by a `numpy.ndarray` of length `window`, centered on the element.
        Edge elements are dropped.

    Notes:
        * Example for 1D arrays: `numpy.std(rolling_window(arr1d, window), 1)`
        * Example for 2D arrays: `numpy.std(rolling_window(arr2d, window), -1)`
        * For additional examples, see [1]_, [2]_.

    References:
        .. [1] http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
        .. [2] http://stackoverflow.com/questions/6811183/
               rolling-window-for-1d-arrays-in-numpy
    """
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


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
    search = sk_gs.RandomizedSearchCV(
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
    train_pred_best = sk_cv.cross_val_predict(
        estimator=model_best, X=train_features, y=train_pred_true,
        cv=cv, n_jobs=n_jobs)
    print("Score from best model training predictions (R^2) = {score:.4f}".format(
            score=sk_met.r2_score(y_true=train_pred_true, y_pred=train_pred_best)))
    train_pred_default = sk_cv.cross_val_predict(
        estimator=model, X=train_features, y=train_pred_true, cv=cv, n_jobs=n_jobs)
    print("Score from default model training predictions (R^2) = {score:.4f}".format(
            score=sk_met.r2_score(y_true=train_pred_true, y_pred=train_pred_default)))
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
        sk_tree.export_graphviz(
            decision_tree=model_best.estimators_[0], out_file=path_dot,
            feature_names=test_features.columns)
        cmd = ['dot', '-Tsvg', path_dot, '-O']
        # Use pre-Python 3.5 subprocess API for backward compatibility.
        subprocess.check_call(args=cmd)
        if show_svg:
            display(SVG(filename=path_dot+'.svg'))
    return None

