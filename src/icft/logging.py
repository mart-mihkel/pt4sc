import logging
import threading

import datasets
import httpx
import torch
import transformers
import accelerate
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import Traceback, install

_console = Console(force_terminal=True)
_suppress = [transformers, datasets, torch, accelerate, httpx]
_handler = RichHandler(
    console=_console,
    markup=True,
    show_time=False,
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

logger = logging.getLogger("icft")
