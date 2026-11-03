from __future__ import annotations

import math
from typing import Any

from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse


def _json_safe(obj: Any) -> Any:
    """Replace NaN/Infinity (which strict JSON can't encode) with their string form."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(v) for v in obj]
    return obj


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """FastAPI's default 422 handler echoes the offending input; when that input is
    NaN/Infinity (Python's json accepts them) rendering the 422 failed and became a 500."""
    errors = exc.errors() if isinstance(exc, RequestValidationError) else []
    return JSONResponse(status_code=422, content={"detail": _json_safe(jsonable_encoder(errors))})
