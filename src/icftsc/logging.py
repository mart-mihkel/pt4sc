import logging
import threading

import accelerate
import datasets
import httpx
import numpy
import torch
import transformers
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import Traceback, install

_console = Console()
_suppress = [transformers, datasets, torch, accelerate, httpx, numpy]
_handler = RichHandler(
    show_path=False,
    console=_console,
    rich_tracebacks=True,
    tracebacks_show_locals=True,
    tracebacks_suppress=_suppress,
)

install(
    console=_console,
    show_locals=True,
    suppress=_suppress,
)

threading.excepthook = lambda args: _console.print(
    Traceback.from_exception(
        args.exc_type,
        args.exc_value,
        args.exc_traceback,
        show_locals=True,
        suppress=_suppress,
    )
)

logging.basicConfig(format="%(message)s", handlers=[_handler])

logger = logging.getLogger("icftsc")
