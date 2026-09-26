"""Bound inline image work before upstream token counting can decode pixels."""
import base64
import binascii
from io import BytesIO
import warnings

from agent_http import RouteError

MAX_IMAGES = 8
MAX_ENCODED_BYTES = 8 * 1024**2
MAX_PIXELS = 16 * 1024**2  # aggregate; reserve <=256 MiB at 16 bytes/pixel
MAX_DIMENSION = 8192
FORMATS = {'data:image/png;base64': 'PNG', 'data:image/jpeg;base64': 'JPEG',
           'data:image/webp;base64': 'WEBP'}


def validate_images(messages):
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            raise RouteError('Each message must be an object')
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                raise RouteError('Content parts must be objects')
            kind = part.get('type')
            if kind == 'text':
                if not isinstance(part.get('text'), str):
                    raise RouteError('Text content must be a string')
                continue
            if kind not in ('image_url', 'input_image'):
                raise RouteError('Only text and inline images are supported by this API')
            if message.get('role') != 'user':
                raise RouteError('Images must be supplied in user messages')
            parts.append(part)
            if len(parts) > MAX_IMAGES:
                raise RouteError(f'At most {MAX_IMAGES} images per request', 413)
    if not parts:
        return
    from PIL import Image
    encoded_bytes = pixels = 0
    for part in parts:
        image = part.get('image_url', part.get('input_image'))
        url = image.get('url') if isinstance(image, dict) else image
        if not isinstance(url, str):
            raise RouteError('Image content requires an inline data URL')
        header, separator, payload = url.partition(',')
        if not separator or header not in FORMATS:
            raise RouteError('Images must be inline PNG, JPEG or WebP base64 data URLs')
        # Check size before decoding. Base64 is ASCII; whitespace is rejected.
        if len(payload) > 4 * ((MAX_ENCODED_BYTES - encoded_bytes + 2) // 3):
            raise RouteError('Encoded images exceed the 8 MiB request budget', 413)
        try:
            raw = base64.b64decode(payload, validate=True)
            encoded_bytes += len(raw)
            if encoded_bytes > MAX_ENCODED_BYTES:
                raise RouteError('Encoded images exceed the 8 MiB request budget', 413)
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(BytesIO(raw), formats=list(FORMATS.values())) as opened:
                    width, height = opened.size
                    if (opened.format != FORMATS[header] or getattr(opened, 'n_frames', 1) != 1
                            or opened.mode not in ('1', 'L', 'LA', 'P', 'RGB', 'RGBA')):
                        raise RouteError('Unsupported image format, depth or animation')
                    pixels += width * height
                    if not 0 < width <= MAX_DIMENSION or not 0 < height <= MAX_DIMENSION or pixels > MAX_PIXELS:
                        raise RouteError('Images exceed the dimension or aggregate pixel budget', 413)
            del raw
        except RouteError:
            raise
        except (ValueError, OSError, binascii.Error, Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise RouteError('Invalid or oversized image') from None
        # Normalize the accepted alias so the upstream schema cannot drop it.
        normalized = {'url': url}
        if isinstance(image, dict) and 'detail' in image:
            if image['detail'] not in ('auto', 'low', 'high'):
                raise RouteError('Invalid image detail')
            normalized['detail'] = image['detail']
        part.clear()
        part.update(type='image_url', image_url=normalized)
