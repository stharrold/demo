#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Top-level script for app_template.

"""


# Import standard packages.
import argparse
import inspect
import logging
import os
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
    app_arg:str,
    logging_level:str='INFO'
    ) -> None:
    r"""Top-level script for template application.

    Args:
        app_arg (str): Argument to pass to the template application.
            Example: arg='1234'
        logging_level (str, optional, 'INFO'): Verbosity of logging level
            from 'DEBUG' (most) to 'CRITICAL' (least).

    Returns:
        None

    """
    # Define log with metadata.
    timestamp = time.strftime(r'%Y%m%dT%H%M%SZ', time.gmtime())
    metadata = {
        'app': 'template',
        'timestamp': timestamp}
    basename = (
        'app-'+metadata['app']+'/'+
        metadata['timestamp'])
    # Set logging level, format, and handlers.
    logger.setLevel(level=logging_level)
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
    (args, *_, values) = inspect.getargvalues(frame)
    logger.info(here+": Argument values: {args_values}".format(
        args_values=[(arg, values[arg]) for arg in sorted(args)]))
    # Execute application.
    logger.info(here+": Executing application.")
    try:
        app_ret = demo.app_template.template.prepend_this(app_arg=app_arg)
        logger.info((here+": Returned value: {app_ret}").format(app_ret=app_ret))
    except:
        logger.critical(
            here+": Failed executing application.",
            exc_info=True,
            stack_info=True)
    # Close log file.
    logger.info(here+": END_LOGGING")
    logger.removeHandler(handler_file)
    return None


if __name__ == '__main__':
    # Note:
    #     * Two-word arguments are separated with an underscore so that the
    #       arguments become valid Python variables when parsed.
    #     * Arguments are checked within the `main` function.
    # Define defaults.
    defaults = {'logging_level': 'INFO'}
    # Parse input arguments and check.
    parser = argparse.ArgumentParser(
        description="Template for demo applications.")
    parser.add_argument(
        "--app_arg",
        required=True,
        type=str,
        help=("Argument to pass to the template application. " +
              "Example: '1234'"))
    parser.add_argument(
        "--logging_level",
        required=False,
        type=str,
        default=defaults['logging_level'],
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help=("Verbosity of logging level from 'DEBUG' (most) to " +
              "'CRITICAL' (least). Default: {default}").format(
              default=defaults['logging_level']))
    args = parser.parse_args()
    main(
        app_arg=args.app_arg,
        logging_level=args.logging_level)
