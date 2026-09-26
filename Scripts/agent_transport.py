"""Transport limits apply before the ASGI router sees complete HTTP headers."""
from uvicorn.protocols.http.h11_impl import H11Protocol

MAX_CONNECTIONS = 128
HEADER_TIMEOUT = 10


class AgentHTTPProtocol(H11Protocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.header_timer = None
        self.rejected = False

    def connection_made(self, transport):
        super().connection_made(transport)
        if len(self.connections) > MAX_CONNECTIONS:
            self.rejected = True
            transport.close()
        else:
            self._wait_for_headers()

    def _cancel_header_timer(self):
        if self.header_timer is not None:
            self.header_timer.cancel()
            self.header_timer = None

    def _wait_for_headers(self):
        self._cancel_header_timer()
        self.header_timer = self.loop.call_later(HEADER_TIMEOUT, self.transport.close)

    def handle_events(self):
        if self.rejected:
            return
        previous = self.cycle
        super().handle_events()
        if self.cycle is not previous:  # Complete headers created a request.
            self._cancel_header_timer()

    def on_response_complete(self):
        if not self.transport.is_closing():
            self._wait_for_headers()
        # May immediately parse a pipelined request and cancel the new timer.
        super().on_response_complete()

    def connection_lost(self, exc):
        self._cancel_header_timer()
        super().connection_lost(exc)


def install_transport_limits():
    """The pinned oMLX CLI constructs uvicorn.Config internally in both modes."""
    import uvicorn
    original = uvicorn.Config
    class AgentConfig(original):
        def __init__(self, *args, **kwargs):
            kwargs.update(http=AgentHTTPProtocol, ws='none', limit_concurrency=MAX_CONNECTIONS,
                          h11_max_incomplete_event_size=16 * 1024, timeout_keep_alive=5)
            super().__init__(*args, **kwargs)
    uvicorn.Config = AgentConfig
