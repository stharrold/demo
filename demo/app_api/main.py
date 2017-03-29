#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Top-level script for app_api.

"""


# Import standard packages.
import argparse
import inspect
import logging
import os
import pickle
import sys
import tempfile
import time
# Import installed packages.
# Import local packages as top-level script.
sys.path.insert(0, os.path.curdir)
import demo


# Define module exports.
__all__ = ['main']


# Define state settings and globals.
# Note: For root-level logger, use `getLogger()`, not `getLogger(__name__)`
# http://stackoverflow.com/questions/17336680/python-logging-with-multiple-modules-does-not-work
logger = logging.getLogger()


def main(
    args
    ) -> None:
    r"""Top-level script for template application.

    Args:
        args (argparse.Namespace): Namespace of arguments.

    Returns:
        None

    """
    # Define log with metadata.
    timestamp = time.strftime(r'%Y%m%dT%H%M%SZ', time.gmtime())
    metadata = {
        'app': 'api',
        'timestamp': timestamp}
    basename = (
        'app-'+metadata['app']+'/'+
        metadata['timestamp'])
    # Set logging level, format, and handlers.
    logger.setLevel(level=args.logging_level)
    fmt = '"%(asctime)s","%(name)s","%(levelname)s","%(message)s"'
    formatter = logging.Formatter(fmt=fmt)
    formatter.converter = time.gmtime
    path_log = os.path.join(
        tempfile.gettempdir(),
        basename.replace('/', '_')+'.log')
    handler_file = logging.FileHandler(filename=path_log, mode='a')
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    print("Log path:\n{path}".format(path=path_log))
    # Define 'here' for logger, begin log, and log arguments passed.
    here = inspect.stack()[0].function
    logger.info(here+": BEGIN_LOGGING")
    logger.info(here+": Log format: {fmt}".format(fmt=fmt.replace('\"', '\'')))
    logger.info(here+": Log date format: ISO 8601, UTC")
    frame = inspect.currentframe()
    (iargs, *_, ivalues) = inspect.getargvalues(frame)
    logger.info(here+": Argument values: {iargs_ivalues}".format(
        iargs_ivalues=[(arg, ivalues[arg]) for arg in sorted(iargs)]))
    logger.info(here+": Version = {version}".format(version=demo.__version__))
    # Execute application.
    logger.info(here+": Executing application.")
    path_pkl = os.path.join(os.path.abspath(os.path.curdir), r'demo/app_api/pkl')
    try:
        if args.cmd == 'train':
            logger.info(here+": Training.")
            demo.app_api.train.train(path=path_pkl)
        elif args.cmd == 'predict':
            # Load pickled model.
            # TODO: Use cPickle for speed.
            path_model = os.path.join(path_pkl, 'model.pkl')
            with open(path_model, mode='rb') as fobj:
                model = pickle.load(file=fobj)
            logger.info(here+": Predicting.")
            res = demo.app_api.predict.predict(features=args.features, model=model)
            logger.info((here+": Predictions:\n{res}").format(res=res))
        else:
            # Load pickled model.
            # TODO: Use cPickle for speed.
            path_model = os.path.join(path_pkl, 'model.pkl')
            with open(path_model, mode='rb') as fobj:
                model = pickle.load(file=fobj)
            logger.info(here+": Serving API.")
            assert args.cmd == 'api'
            # TDOO: Use manager.run instead https://github.com/fromzeroedu/flask_blog/blob/master/manage.py
            app.run(port=args.port, debug=app.debug)
    except:
        logger.critical(here+": Failed executing application.", exc_info=True)
    # Close log file.
    logger.info(here+": END_LOGGING")
    logger.removeHandler(handler_file)
    return None


if __name__ == '__main__':
    # Note:
    #     * Two-word arguments are separated with an underscore so that the
    #       arguments become valid Python variables when parsed.
    #     * Arguments are checked within the `main` function.
    # TODO: Use a Web Server Gateway Interface for deployment
    #     http://flask.pocoo.org/docs/0.12/deploying/#deployment
    # Define defaults.
    defaults = {
        'logging_level': 'INFO',
        'api': {
            'port': 9000,
            'debug': False}}
    # Parse input arguments and check.
    parser = argparse.ArgumentParser(
        description="Predictive application with an API.")
    parser.add_argument(
        "--logging_level",
        required=False,
        type=str,
        default=defaults['logging_level'],
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help=("Verbosity of logging level from 'DEBUG' (most) to " +
              "'CRITICAL' (least). Default: {default}").format(
              default=defaults['logging_level']))
    subparsers = parser.add_subparsers(dest='cmd')
    parser_train = subparsers.add_parser(
        'train',
        help="Train (or retrain) the model.")
    parser_predict = subparsers.add_parser(
        'predict',
        help="Predict target(s) from features.")
    parser_predict.add_argument(
        "--features",
        required=True,
        type=str,
        help=("Features as JSON from which to predict the target(s)." +
              "Example: {ftrs}".format(
                ftrs='{"sl": {"0":5.7, "1":6.9}, "sw": {"0":4.4, "1":3.1}, "pl": {"0":1.5, "1":4.9}, "pw": {"0":0.4, "1":1.5}}')))
    parser_api = subparsers.add_parser(
        'api',
        help="Serve the predictive model with a RESTful API.")
    parser_api.add_argument(
        '--port',
        required=False,
        type=int,
        default=defaults['api']['port'],
        help="Port on which to serve the API. Default: {default}".format(
            default=defaults['api']['port']))
    parser_api.add_argument(
        '--debug',
        required=False,
        type=bool,
        default=defaults['api']['debug'],
        help="Flag to reload code and show debugger. Default: {default}".format(
            default=defaults['api']['debug']))
    args = parser.parse_args()
    main(args)
