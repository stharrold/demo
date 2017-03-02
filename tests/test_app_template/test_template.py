#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Pytests for demo/app_template/template.py.

"""


# Import standard packages.
import os
import sys
# Import installed packages.
# Import local packages.
sys.path.insert(0, os.path.curdir)
import demo


def test_prepend_this(
	app_arg:str='my argument',
	ref_app_ret:str='Prepended my argument'
	) -> None:
	r"""Pytest for prepend_this

	"""
	test_app_ret = demo.app_template.template.prepend_this(app_arg=app_arg)
	assert ref_app_ret == test_app_ret
	return None
