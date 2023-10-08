import enum
import logging
import typing as t


class VerbosityLevel(enum.IntEnum):
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG


def set_verbosity(obj: t.Any, verbosity: t.Optional[str] = None) -> None:
    if verbosity is None:
        return

    try:
        obj.logger.setLevel(VerbosityLevel[verbosity.upper()].value)
    except KeyError:
        obj.logger.error(f"Unknown verbosity level {verbosity}")
