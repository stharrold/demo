#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Application API.

"""


# Import standard packages.
import inspect
import logging
# Import installed packages.
import matplotlib.pyplot as plt
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


def predict(features:str):
    r"""Predict the target(s) from the features.

    Args:
        features (str): Features from which to predict the target(s).
            Example: TODO

    Returns:
        preds (str): Predicted target(s).

    """
    # Define 'here' for logger and log arguments passed.
    here = inspect.stack()[0].function
    frame = inspect.currentframe()
    (args, *_, values) = inspect.getargvalues(frame)
    logger.info(here+": Argument values: {args_values}".format(
        args_values=[(arg, values[arg]) for arg in sorted(args)]))
    # Log the code version from util.__version__.
    logger.info(here+": Version = {version}".format(version=utils.__version__))
    # Prepend the argument and return.
    # TODO
    preds = 'TODO'
    return app_ret
