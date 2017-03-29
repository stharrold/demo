#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Predict from model for API.

"""


# Import standard packages.
import inspect
import json
import logging
# Import installed packages.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Import local packages.
from .. import utils


# Define module exports:
__all__ = ['predict']


# Define state settings and globals.
# Note: For non-root-level loggers, use `getLogger(__name__)`
#     http://stackoverflow.com/questions/17336680/python-logging-with-multiple-modules-does-not-work
logger = logging.getLogger(__name__)
# Set the matplotlib backend to the Anti-Grain Geometry C++ library.
# Note: Use plt.switch_backend since matplotlib.use('agg') before importing pyplot fails.
plt.switch_backend('agg')
# Set matplotlib styles with seaborn
sns.set()


def predict(
    features:str,
    model):
    r"""Predict the target(s) from the features.

    Args:
        features (str): Features as JSON from which to predict the target(s).
            Example: '{"sl": {"0":5.7, "1":6.9}, "sw": {"0":4.4, "1":3.1}, "pl": {"0":1.5, "1":4.9}, "pw": {"0":0.4, "1":1.5}}'
        model: Trained model from scikit-learn.

    Returns:
        preds (str): Predicted target(s).

    """
    # Define 'here' for logger and log arguments passed.
    here = inspect.stack()[0].function
    frame = inspect.currentframe()
    (args, *_, values) = inspect.getargvalues(frame)
    logger.info(here+": Argument values: {args_values}".format(
        args_values=[(arg, values[arg]) for arg in sorted(args)]))
    # Parse the JSON features, order the columns, and predict.
    # TODO: Pickle the column order.
    data = pd.DataFrame.from_dict(json.loads(features))
    cols = ['sl', 'sw', 'pl', 'pw']
    preds = pd.DataFrame(
        data=model.predict(data[cols].values),
        index=data.index).to_json()
    return preds
