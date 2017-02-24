# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# r"""Archived utilities for predictive analytics demo.

# """


# # Import standard packages.
# # Import __future__ for Python 2x backward compatibility.
# from __future__ import absolute_import, division, print_function
# # Import installed packages.
# import numpy as np
# import pandas as pd


# def shuffle_dataframe(df, seed_row=None, seed_col=None):
#     r"""Shuffle the rows and columns of a dataframe to remove any pre-existing
#     structure.
    
#     Args:
#         df (pandas.DataFrame): The data to shuffle.
#         seed_row (int, optional, default=None):
#         seed_col (int, optional, defualt=None):
#             Seeds for `numpy.random.seed` before shuffling the
#             row/column. Set to an `int` for reproducibility.
    
#     Returns:
#         df_shuffled (pandas.DataFrame): A deep copy of the shuffled dataframe.
        
#     See Also:
#         pandas.DataFrame, numpy.random.shuffle, numpy.random.seed
    
#     Notes:
#         * Made obsolete by `pandas.DataFrame.sample`.
    
#     """
#     # Check input.
#     if not isinstance(df, pd.DataFrame):
#         raise ValueError("`df` must be an instance of `pandas.DataFrame`.")
#     if (seed_row is not None) and (not isinstance(seed_row, int)):
#         raise ValueError("`seed_row` must be an `int` or `None`.")
#     if (seed_col is not None) and (not isinstance(seed_col, int)):
#         raise ValueError("`seed_col` must be an `int` or `None`.")
#     # Shuffle index and columns in-place.
#     index = df.index.values.copy()
#     if seed_row is not None:
#         np.random.seed(seed=seed_row)
#     np.random.shuffle(index)
#     columns = df.columns.values.copy()
#     if seed_col is not None:
#         np.random.seed(seed=seed_col)
#     np.random.shuffle(columns)
#     df_shuffled = df.loc[index, columns].copy()
#     return df_shuffled
