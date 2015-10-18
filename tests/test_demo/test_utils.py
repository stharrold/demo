#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Pytests for demo/utils.py

"""


# Import standard packages.
# Import __future__ for Python 2x backward compatibility.
from __future__ import absolute_import, division, print_function
import sys
import pdb
sys.path.insert(0, '.') # Test the code in this repository.
# Import installed packages.
import numpy as np
import pandas as pd
# Import local packages.
import demo


def test_shuffle_dataframe(
    df=pd.DataFrame(
        data=[('A', 'a', 1), ('B', 'b', 2), ('C', 'c', 3)],
        index=['r0', 'r1', 'r2'], columns=['c0', 'c1', 'c2']),
    seed_row=0, seed_col=0,
    ref_df_shuffled=pd.DataFrame(
        data=[(3, 'c', 'C'), (1, 'a', 'A'), (2, 'b', 'B')],
        index=['r2', 'r0', 'r1'], columns=['c2', 'c1', 'c0'])):
    r"""Pytest for demo/utils.py:
    shuffle_dataframe
    
    """
    test_df_shuffled = demo.utils.shuffle_dataframe(
        df=df, seed_row=seed_row, seed_col=seed_col)
    pdb.set_trace()    
    assert ref_df_shuffled.equals(test_df_shuffled)
    return None
