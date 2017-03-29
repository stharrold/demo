#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Train model for API.

"""


# Import standard packages.
import inspect
import json
import logging
import os
import pickle
# Import installed packages.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.datasets as sk_ds
import sklearn.ensemble as sk_en
import sklearn.cross_validation as sk_cv
import sklearn.metrics as sk_me
# Import local packages.
from .. import utils


# Define module exports:
__all__ = ['train']


# Define state settings and globals.
# Note: For non-root-level loggers, use `getLogger(__name__)`
#     http://stackoverflow.com/questions/17336680/python-logging-with-multiple-modules-does-not-work
logger = logging.getLogger(__name__)
# Set the matplotlib backend to the Anti-Grain Geometry C++ library.
# Note: Use plt.switch_backend since matplotlib.use('agg') before importing pyplot fails.
plt.switch_backend('agg')
# Set matplotlib styles with seaborn
sns.set()


def train(
    path:str):
    r"""Predict the target(s) from the features.

    Args:
        path (str): Directory path to store the pickle files.
            Example: os.path.join(os.path.abspath(os.path.curdir), r'demo/app_api/pkl')

    Returns:
        None

    """
    # Define 'here' for logger and log arguments passed.
    here = inspect.stack()[0].function
    frame = inspect.currentframe()
    (args, *_, values) = inspect.getargvalues(frame)
    logger.info(here+": Argument values: {args_values}".format(
        args_values=[(arg, values[arg]) for arg in sorted(args)]))
    # Train the model, report, and serialize to disk.
    # TODO: Pickle the column order.
    iris = sk_ds.load_iris()
    X_train, X_test, y_train, y_test = sk_cv.train_test_split(iris.data, iris.target)
    model = sk_en.RandomForestClassifier(n_estimators=100, n_jobs=2)
    model.fit(X_train, y_train)
    logger.info(here+": Classification report:\n{rep}".format(
        rep=sk_me.classification_report(y_test, model.predict(X_test))))
    path_model = os.path.join(path, 'model.pkl')
    with open(path_model, mode='wb') as fobj:
        pickle.dump(obj=model, file=fobj)
    return None
