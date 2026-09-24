"""Small, model-free boundary for the local agent API."""
import asyncio
import json
import math
from urllib.parse import urlsplit

MAX_BODY = 16 * 1024 * 1024
BODY_TIMEOUT = 30


class RouteError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def validate_http(scope):
    headers = {}
    for name, value in scope.get('headers', []):
        name = name.lower()
        if name in (b'host', b'origin', b'content-type', b'content-length', b'sec-fetch-site'):
            if name in headers:
                raise RouteError('Duplicate request header')
            headers[name] = value.decode('latin-1')
    host = headers.get(b'host', '')
    try:
        authority = urlsplit('http://' + host)
        if (authority.hostname not in ('localhost', '127.0.0.1', '::1') or authority.username
                or authority.password or authority.path or authority.query or authority.fragment):
            raise ValueError('Host')
        port = authority.port or 80
        origin = headers.get(b'origin')
        if origin is not None:
            browser = urlsplit(origin)
            if (browser.scheme != 'http' or browser.netloc.lower() != authority.netloc.lower()
                    or browser.path or browser.query or browser.fragment):
                raise ValueError('Origin')
        if headers.get(b'sec-fetch-site') == 'cross-site':
            raise ValueError('Fetch site')
        if scope.get('server') and port != scope['server'][1]:
            raise ValueError('Port')
    except ValueError:
        raise RouteError('Only direct loopback requests and same-origin browser requests are allowed', 403)
    if scope['method'] == 'POST':
        media = headers.get(b'content-type', '').split(';', 1)[0].strip().lower()
        if media != 'application/json':
            raise RouteError('Content-Type must be application/json', 415)
        length = headers.get(b'content-length')
        if length is not None:
            if not length.isascii() or not length.isdecimal():
                raise RouteError('Invalid Content-Length')
            if len(length) > 8 or int(length) > MAX_BODY:
                raise RouteError('Request exceeds 16 MiB', 413)


def decode_json(raw):
    def reject_constant(value):
        raise ValueError('Non-finite JSON number')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    try:
        result = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
        pending = [(result, 0)]
        visited = 0
        while pending:
            value, depth = pending.pop()
            visited += 1
            if visited + len(pending) > 100000:
                raise ValueError('JSON has too many values')
            if depth > 64:
                raise ValueError('JSON nesting exceeds 64 levels')
            if isinstance(value, str):
                value.encode('utf-8')  # Reject lone escaped surrogates.
            elif isinstance(value, float) and not math.isfinite(value):
                raise ValueError('Non-finite JSON number')
            elif isinstance(value, dict):
                pending.extend((item, depth + 1) for pair in value.items() for item in pair)
            elif isinstance(value, list):
                pending.extend((item, depth + 1) for item in value)
        return result
    except (ValueError, UnicodeError, RecursionError):
        raise RouteError('Invalid JSON: use unique keys, finite numbers, valid Unicode, at most 100000 values and 64 nesting levels')


async def read_body(receive):
    body = bytearray()
    try:
        async with asyncio.timeout(BODY_TIMEOUT):
            while True:
                message = await receive()
                if message['type'] == 'http.disconnect':
                    return None
                chunk = message.get('body', b'')
                if len(body) + len(chunk) > MAX_BODY:
                    raise RouteError('Request exceeds 16 MiB', 413)
                body.extend(chunk)
                if not message.get('more_body', False):
                    return body
    except TimeoutError:
        raise RouteError('Request body upload timed out', 408)
