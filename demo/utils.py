#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Utilities for predictive analytics demo.

"""


# Import standard packages.
# For Python 2x backward compatibility
from __future__ import absolute_import, division, print_function
# Import installed packages.


def shuffle_dataframe(df):
    r"""Shuffle the index and columns of a dataframe to remove any pre-existing structure.
    
    Args:
        df (pandas.DataFrame): The data to shuffle.
    
    Returns:
        df_shuffled (pandas.DataFrame): A deep copy of the shuffled dataframe.
        
    See Also:
        numpy.random.shuffle
    
    Notes:
        * For reproducibility, seed the random state using `numpy.random.RandomState`.
    
    """
    # Check input.
    if not isinstance(df, pd.DataFrame):
        raise ValueError("`df` must be an instance of `pandas.DataFrame`.")
    # Shuffle index and columns in-place.
    index = df.index.values.copy()
    np.random.shuffle(index)
    columns = df.columns.values.copy()
    np.random.shuffle(columns)
    df_shuffled = df.loc[index, columns].copy()
    return df_shuffled
