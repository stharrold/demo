#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Pytests for demo/app_template/main.py.

"""


# Import standard packages.
import io
import os
import subprocess
import sys
import tempfile
# Import installed packages.
# Import local packages as top-level script.
sys.path.insert(0, os.path.curdir)
import demo


def test__all__(
    ref_all=['main']
    ) -> None:
    r"""Pytest for __all__

    Notes:
        * Check that expected objects are exported.

    """
    test_all = demo.app_template.main.__all__
    assert ref_all == test_all
    for attr in ref_all:
        assert hasattr(demo.app_template, attr)
    return None


def test_main(
    app_arg:str='test argument',
    logging_level:str='INFO'
    ) -> None:
    r"""Pytest for main.

    Notes:
        * Execute function, capture stdout, and
            check that a non-empty log file exists.

    """
    # Create a string buffer to capture print statements and
    # reassign to the `sys.stdout` pointer. Capture the output then
    # reassign `sys.stdout` to original stream.
    test_stdout = io.StringIO()
    sys.stdout = test_stdout
    demo.app_template.main.main(
        app_arg=app_arg,
        logging_level=logging_level)
    sys.stdout = sys.__stdout__
    (_, path_log) = test_stdout.getvalue().split('\n')[:2]
    with open(path_log) as fobj:
        for line in fobj:
            assert ',"CRITICAL",' not in line
    return None


def test__main__(
    app_arg:str='test argument',
    logging_level:str='INFO'
    ) -> None:
    r"""Pytest for __main__

    Notes:
        * Execute function as top-level script, capture stdout,
            and check that a non-empty log file exists.

    """
    # Create a file handle and pass to subprocess to capture stdout.
    # Remove local stdout file.
    (_, path_stdout) = tempfile.mkstemp(
        prefix='stdout_', suffix='.txt', text=True)
    cmd = [
        'python', 'demo/app_template/main.py',
        '--app_arg', app_arg,
        '--logging_level', logging_level]
    with open(path_stdout, 'w') as fobj:
        subprocess.run(args=cmd, check=True, stdout=fobj, stderr=fobj)
    with open(path_stdout) as fobj:
        (_, path_log) = fobj.read().split('\n')[:2]
    with open(path_log) as fobj:
        for line in fobj:
            assert ',"CRITICAL",' not in line
    os.remove(path=path_stdout)
    return None
