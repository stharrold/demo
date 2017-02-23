#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Application demos.

"""


# Import standard packages.
# Import installed packages.
# Import local packages.
from . import app_template
from . import utils


# Define package exports
__all__ = [
    'app_template',
    'utils']


# Follow Semantic Versioning v2.0.0 (http://semver.org/spec/v2.0.0.html)
# Note: `demo.__init__.__version__` can only be imported by one
#     subpackage of `demo`, otherwise an `ImportError` is raised.
#     If your module needs to import `__version__` into its namespace,
#     access with `import utils` then `utils.__version__`.
__version__ = '0.0.0'
