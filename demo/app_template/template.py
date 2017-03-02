#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Application template.

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
__all__ = ['prepend_this']


# Define state settings and globals.
# Note: For non-root-level loggers, use `getLogger(__name__)`
#     http://stackoverflow.com/questions/17336680/python-logging-with-multiple-modules-does-not-work
logger = logging.getLogger(__name__)
# Set the matplotlib backend to the Anti-Grain Geometry C++ library.
# Note: Use plt.switch_backend since matplotlib.use('agg') before importing pyplot fails.
plt.switch_backend('agg')
# Set matplotlib styles with seaborn
sns.set()


def prepend_this(app_arg:str):
    r"""Prepend the application argument with 'Prepended '

    Args:
        app_arg (str): `str` to prepend.

    Returns:
        app_ret (str): Prepended `str`.

    Raises:
        ValueError: Raised if not `isinstance(app_arg, str)`

    """
    # Check arguments.
    if not isinstance(app_arg, str):
        raise ValueError(
            "`app_arg` must be type `str`. " +
            "Required: type(app_arg) == str"
            "Given: type(app_arg) == {typ}").format(
            typ=type(app_arg))
    # Define 'here' for logger and log arguments passed.
    here = inspect.stack()[0].function
    frame = inspect.currentframe()
    (args, *_, values) = inspect.getargvalues(frame)
    logger.info(here+": Argument values: {args_values}".format(
        args_values=[(arg, values[arg]) for arg in sorted(args)]))
    # Log the code version from util.__version__.
    logger.info(here+": Version = {version}".format(version=utils.__version__))
    # Prepend the argument and return.
    app_ret = 'Prepended '+app_arg
    return app_ret
