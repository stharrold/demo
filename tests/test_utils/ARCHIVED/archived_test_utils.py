    # #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# r"""Archived pytests for demo/archived_utils.py

# """


# # Import standard packages.
# # Import __future__ for Python 2x backward compatibility.
# from __future__ import absolute_import, division, print_function
# import sys
# sys.path.insert(0, '.') # Test the code in this repository.
# # Import installed packages.
# import numpy as np
# import pandas as pd
# # Import local packages.
# import demo


# def test_shuffle_dataframe(
#     df=pd.DataFrame(
#         data=[('A', 'a', 0), ('B', 'b', 1), ('C', 'c', 2)],
#         index=['r0', 'r1', 'r2'], columns=['c0', 'c1', 'c2']),
#     seed_row=0, seed_col=0,
#     ref_df_shuffled=pd.DataFrame(
#         data=[(2, 'c', 'C'), (1, 'b', 'B'), (0, 'a', 'A')],
#         index=['r2', 'r1', 'r0'], columns=['c2', 'c1', 'c0'])):
#     r"""Pytest for demo/utils.py:
#     shuffle_dataframe
    
#     """
#     test_df_shuffled = demo.utils.shuffle_dataframe(
#         df=df, seed_row=seed_row, seed_col=seed_col)
#     assert ref_df_shuffled.equals(test_df_shuffled)
#     return None
    
    
# # TODO
# # def robust_cohen_d(arr1, arr2):
# #     r"""Calculate Cohen d statistic for two arrays using rank-based statistics.
# #     The Cohen d statistic is a measure of the size of an effect.
# #     Rank-based statistics are robust (more tolerant of) outliers.
    
# #     Args:
# #         arr1 (numpy.ndarray):
# #         arr2 (numpy.ndarray):
# #             Arrays for the two data sets.
    
# #     Returns:
# #         cohen_d (float)
        
# #     Notes:
# #         * cohen_d = (mean(arr1) - mean(arr2)) / from Ch 2 [1]_.
        
# #     References:
# #     .. [1] 
    
# #     """
