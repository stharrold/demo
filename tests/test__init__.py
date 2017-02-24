#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Pytests for demo/__init__.py

"""


# Import standard packages.
import os
import sys
# Import installed packages.
# Import local packages.
sys.path.insert(0, os.path.curdir)
import demo


def test__all__(
    ref_all=[
        'app_template',
        'utils']
    ) -> None:
    r"""Pytest for __all__

    Notes:
        * Check that expected modules are exported.

    """
    test_all = demo.__all__
    assert ref_all == test_all
    for attr in ref_all:
        assert hasattr(demo, attr)
    return None


def test__version__(
    version:str=demo.__version__
    ) -> None:
    r"""Pytest for __version__.

    Notes:
        * Version numbers follow Semantic Versioning v2.0.0.[^semver]
        * Test that demo.__version__ is a string with format 'N.N.N[-X]',
            where N are integer numbers and X is an optional label string.
        * Git tags of versions have format 'vN.N.N[-X]'.

    References:
        [^semver]: http://semver.org/spec/v2.0.0.html

    """
    assert isinstance(version, str)
    assert len(version.split('.')) == 3
    (major, minor, patch) = version.split('.')
    assert isinstance(int(major), int)
    assert isinstance(int(minor), int)
    assert (len(patch.split('-')) == 1) or (len(patch.split('-')) == 2)
    if len(patch.split('-')) == 1:
        assert isinstance(int(patch), int)
    else:
        assert len(patch.split('-')) == 2
        (patch_num, _) = patch.split('-')
        assert isinstance(int(patch_num), int)
    return None
