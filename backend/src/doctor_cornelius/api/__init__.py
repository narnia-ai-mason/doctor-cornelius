"""Doctor Cornelius API module.

This module provides the FastAPI application and API routers for
the Doctor Cornelius temporal knowledge graph service.
"""

from doctor_cornelius.api.router import create_api_router, create_app, get_app

__all__ = [
    "get_app",
    "create_app",
    "create_api_router",
]
