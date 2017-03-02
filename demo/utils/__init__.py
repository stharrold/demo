#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""General-purpose utilities for demo applications.

"""


# Import standard packages.
# Import installed packages.
# Import local packages.
from . import utils
# Note: `demo.__init__.__version__` can only be imported by one
#     subpackage of `demo`, otherwise an `ImportError` is raised.
#     If your module needs to import `__version__` into its namespace,
#     access with `import utils` then `utils.__version__`.
from demo.__init__ import __version__


# Define package exports.
__all__ = [
	'__version__',
    'utils']
