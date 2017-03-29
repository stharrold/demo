#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Demo application with an API.

"""


# Import standard packages.
# Import installed packages.
# Import local packages.
from . import api
from . import main
from . import predict
from . import train


# Define package index.
__all__ = [
    'api',
    'main',
    'predict',
    'train']
