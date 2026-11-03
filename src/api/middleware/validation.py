from __future__ import annotations

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_TOO_LARGE = "Request body too large (max 10MB)"


class InputValidationMiddleware:
    """Reject malformed Content-Length headers and bodies over MAX_BODY_SIZE.

    Pure ASGI (not BaseHTTPMiddleware) so the limit also covers chunked requests, which
    carry no Content-Length and used to bypass the check entirely.
    """

    MAX_BODY_SIZE = 10 * 1024 * 1024

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = self.MAX_BODY_SIZE
        content_length = Headers(scope=scope).get("content-length")
        if content_length:
            try:
                declared = int(content_length)
            except ValueError:
                declared = -1
            if declared < 0:
                # A malformed header used to raise and surface as a 500.
                await JSONResponse(
                    status_code=400, content={"detail": "Invalid Content-Length header"}
                )(scope, receive, send)
                return
            if declared > limit:
                await JSONResponse(status_code=413, content={"detail": _TOO_LARGE})(
                    scope, receive, send
                )
                return

            await self.app(scope, receive, send)
            return

        # No Content-Length (chunked): read the body here, bounded by the limit, then
        # replay it. (Raising from a wrapped receive doesn't work: FastAPI turns it into
        # a generic 400.) Requests that declare a length are cut off at that length by
        # the HTTP server, so they are covered by the check above.
        chunks: list[bytes] = []
        received = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":  # client disconnected
                return
            body = message.get("body", b"")
            received += len(body)
            if received > limit:
                await JSONResponse(status_code=413, content={"detail": _TOO_LARGE})(
                    scope, receive, send
                )
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)
