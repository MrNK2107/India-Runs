from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class InputValidationMiddleware(BaseHTTPMiddleware):
    MAX_BODY_SIZE = 10 * 1024 * 1024

    async def dispatch(self, request: Request, call_next) -> Response:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.MAX_BODY_SIZE:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body too large (max 10MB)"},
            )
        return await call_next(request)
