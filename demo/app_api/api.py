#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Top-level script for app_api.

"""


# Import standard packages.
import argparse
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
__all__ = ['make_predict']


# Define globals.
path_app = os.path.join(os.path.abspath(os.path.curdir), r'demo/app_api')
path_model = os.path.join(path_app, 'model.pkl')
with open(path_model, mode='rb') as fobj:
    model = pickle.load(file=fobj)
# TODO: path_cols
# TODO: main --train calls train.
# TODO: main --predict calls predict. args: features
# TODO: main --api serves api. args: port, debug.
app = flask.Flask(__name__)


@app.route('/api', methods=['POST'])
def make_predict():
    data = pd.DataFrame.from_dict(
        flask.request.get_json(force=True))
    cols = ['sl', 'sw', 'pl', 'pw']
    pred = pd.DataFrame(
        data=model.predict(data[cols].values),
        index=data.index)
    return flask.jsonify(results=pred.to_json())


if __name__ == '__main__':
    app.run(port=9000, debug=True)