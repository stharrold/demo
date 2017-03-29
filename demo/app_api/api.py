#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""App for API.

"""


# Import standard packages.
import inspect
import json
import logging
import os
import pickle
import sys
import tempfile
import time
# Import installed packages.
import flask
import pandas as pd
# Import local packages as top-level script.
sys.path.insert(0, os.path.curdir)
import demo


# Define module exports.
__all__ = ['predict']


# Define state settings and globals.
# Note: For non-root-level loggers, use `getLogger(__name__)`
#     http://stackoverflow.com/questions/17336680/python-logging-with-multiple-modules-does-not-work
logger = logging.getLogger(__name__)
# Configure app following https://github.com/fromzeroedu/flask_blog/blob/master/__init__.py
app = flask.Flask(__name__)
app.config.from_object('settings')
# Load pickled model.
# TODO: Use cPickle for speed.
# TODO: Don't hardcode location of pickle files in both main.py and api.py
path_pkl = os.path.join(os.path.abspath(os.path.curdir), r'demo/app_api/pkl')
path_model = os.path.join(path_pkl, 'model.pkl')
with open(path_model, mode='rb') as fobj:
    model = pickle.load(file=fobj)



@app.route('/api', methods=['POST'])
def predict():
    r"""Endpoint for making predictions.

    Args:
        None

    Request:
        features (str): Features as JSON from which to predict the target(s).
            Example: '{"sl": {"0":5.7, "1":6.9}, "sw": {"0":4.4, "1":3.1}, "pl": {"0":1.5, "1":4.9}, "pw": {"0":0.4, "1":1.5}}'

    Returns:
        res (str): Predicted target(s) as JSON response.

    See Also:
        demo.predict.predict

    """
    features = flask.request.get_json(force=True)
    preds = demo.predict.predict(
        features=features,
        model=model)
    res = flask.jsonify(results=preds)
    return res
