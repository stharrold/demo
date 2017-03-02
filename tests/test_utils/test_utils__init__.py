#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Pytests for demo/utils/__init__.py

"""


# Import standard packages.
import os
import sys
# Import installed packages.
# Import local packages.
sys.path.insert(0, os.path.curdir)
import demo


def test__all__(
    ref_all = [
        '__version__',
        'utils']
    ) -> None:
    r"""Pytest for __all__

    Notes:
        * Check that expected modules are exported.

    """
    test_all = demo.utils.__all__
    assert ref_all == test_all
    for attr in ref_all:
        assert hasattr(demo.utils, attr)
    return None
