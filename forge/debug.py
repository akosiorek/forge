from __future__ import print_function
import functools
import sys
import traceback
import pdb


def debug_on(*exceptions):
    """Adapted from https://stackoverflow.com/questions/18960242/is-it-possible-to-automatically-break-into-the-debugger-when-a-exception-is-thro/18962528
    """

    if not exceptions:
        exceptions = (AssertionError,)

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)

            except exceptions as err:
                last_traceback = sys.exc_info()[2]
                traceback.print_tb(last_traceback)
                print(err)
                pdb.post_mortem(last_traceback)

        return wrapper

    return decorator