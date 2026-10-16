"""Resource-split APIRouter modules for the Review API (V1-06, AC-07).

Each module builds the router for one resource boundary; ``api.py`` only
assembles them.  Endpoint function names, paths, methods, status codes and
response models are the frozen contract — they moved, they did not change.
"""

from __future__ import annotations

from .actions import build_actions_router
from .cases import build_cases_router
from .meta import build_meta_router, build_retired_router
from .reviews import build_review_records_router, build_reviews_router
from .session import build_session_router

__all__ = [
    "build_actions_router",
    "build_cases_router",
    "build_meta_router",
    "build_retired_router",
    "build_review_records_router",
    "build_reviews_router",
    "build_session_router",
]
