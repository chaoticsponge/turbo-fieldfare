"""Local file opens that reject symlink and special-file substitutions."""
from contextlib import contextmanager
import os
import stat


@contextmanager
def open_regular(path, flags=os.O_RDONLY, mode='rb'):
    fd = os.open(path, (flags & ~os.O_TRUNC) | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('Expected a regular file')
        if flags & (os.O_WRONLY | os.O_RDWR):
            if info.st_nlink != 1:
                raise ValueError('Refusing to write a multiply-linked file')
            if flags & os.O_TRUNC:
                os.ftruncate(fd, 0)
        handle = os.fdopen(fd, mode)
    except BaseException:
        os.close(fd)
        raise
    with handle:
        yield handle


def private_directory(path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fchmod(fd, 0o700)
    finally:
        os.close(fd)
