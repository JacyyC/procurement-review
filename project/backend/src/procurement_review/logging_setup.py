"""Narrow standard-logging wiring for the service runtime (V1-06, AC-08).

Logs are a diagnostic surface, not a business record: Trace, evidence references
and the approval audit keep carrying the facts.  Events only ever carry stable IDs
(run/case/request IDs, generation numbers) and status codes — never credential
values, DSNs, idempotency keys or business payloads.

The level comes from ``REVIEW_LOG_LEVEL`` (default ``INFO``); the format is a
deterministic UTC timestamp/level/logger/message line so container logs can be
diffed.  Wiring is idempotent: repeated ``create_app`` calls attach exactly one
handler to the package logger.  The package logger does not propagate, so one
event yields exactly one line (see ``configure_logging``); tests capture records
by attaching ``caplog.handler`` to it.
"""

from __future__ import annotations

import logging
import os
import time

REVIEW_LOG_LEVEL_ENV = "REVIEW_LOG_LEVEL"
PACKAGE_LOGGER_NAME = "procurement_review"
_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%SZ"


class _RuntimeHandler(logging.StreamHandler):
    """Marker subclass so repeated wiring can recognise its own handler."""


def configure_logging() -> None:
    """Attach one stderr handler to the package logger, at most once."""

    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    level_name = (os.environ.get(REVIEW_LOG_LEVEL_ENV) or "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    # The MCP server dependency calls ``logging.basicConfig`` on the first Run
    # execution and installs a bare ``%(message)s`` root handler.  Propagating
    # would print every event a second time without the deterministic format.
    logger.propagate = False
    if any(isinstance(handler, _RuntimeHandler) for handler in logger.handlers):
        return
    handler = _RuntimeHandler()
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    logger.addHandler(handler)


__all__ = ["PACKAGE_LOGGER_NAME", "REVIEW_LOG_LEVEL_ENV", "configure_logging"]
