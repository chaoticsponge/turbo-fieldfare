#!/usr/bin/env python3
"""List or explicitly install pinned specialist packages, without loading models."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request
import urllib.parse

from agent_files import open_regular
from agent_models import ROOT, ROLES, catalog, model_path


def digest(path):
    h = hashlib.sha256()
    with open_regular(path) as handle:
        while data := handle.read(1024 * 1024):
            h.update(data)
    return h.hexdigest()


def matches(path, entry):
    return (not path.is_symlink() and path.is_file()
            and path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256'])


class HTTPSRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != 'https':
            raise ValueError('Model downloads cannot redirect away from HTTPS')
        return super().redirect_request(request, fp, code, message, headers, newurl)


def fetch_file(directory, entry, url, opener=None):
    if urllib.parse.urlsplit(url).scheme != 'https':
        raise ValueError('Model downloads require HTTPS')
    opener = opener or urllib.request.build_opener(HTTPSRedirects()).open
    name = entry['name']
    if Path(name).name != name or '..' in name:
        raise ValueError('Invalid catalog path')
    target = directory / name
    partial = directory / (name + '.part')
    if target.is_symlink() or partial.is_symlink():
        raise ValueError('Refusing a symlink payload')
    if target.exists():
        if matches(target, entry):
            return
        raise ValueError(f'Existing file failed verification: {target}')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > entry['bytes']:
        raise ValueError(f'Oversized partial file: {partial}')
    if offset < entry['bytes']:
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        with opener(urllib.request.Request(url, headers=headers), timeout=120) as response:
            append = offset > 0 and response.status == 206
            if append and response.headers.get('Content-Range') != f"bytes {offset}-{entry['bytes'] - 1}/{entry['bytes']}":
                raise ValueError('Unexpected download range; partial file preserved')
            if response.status not in (200, 206) or (not offset and response.status != 200):
                raise ValueError('Unexpected download response')
            written = offset if append else 0
            with open_regular(partial, os.O_CREAT | os.O_WRONLY | (os.O_APPEND if append else os.O_TRUNC),
                              'ab' if append else 'wb') as handle:
                while data := response.read(1024 * 1024):
                    written += len(data)
                    if written > entry['bytes']:
                        raise ValueError('Download exceeds pinned file size')
                    handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
    if not matches(partial, entry):
        raise ValueError(f'Incomplete or corrupt download: {partial}; partial file preserved')
    partial.replace(target)


def install(role, root, worker_precision='8bit'):
    manifest = catalog(role, worker_precision)
    directory = model_path(role, root, worker_precision)
    if directory.is_symlink():
        raise ValueError('Installation directory cannot be a symlink')
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / 'qwen-install.json'
    if receipt.is_symlink():
        raise ValueError('Installation receipt cannot be a symlink')
    if receipt.exists() and json.loads(receipt.read_text()) != manifest:
        raise ValueError('Existing receipt belongs to another revision')
    lock_path = directory / '.install.lock'
    if lock_path.is_symlink():
        raise ValueError('Installation lock cannot be a symlink')
    with open_regular(lock_path, os.O_CREAT | os.O_RDWR, 'r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        required = 0
        for entry in manifest['files']:
            if not (directory / entry['name']).exists():
                partial = directory / (entry['name'] + '.part')
                if partial.is_symlink():
                    raise ValueError('Partial file cannot be a symlink')
                required += max(0, entry['bytes'] - (partial.stat().st_size if partial.exists() else 0))
        if shutil.disk_usage(directory).free < required + 2 * 1024**3:
            raise ValueError(f'Need {required / 1024**3:.1f} GiB plus 2 GiB free disk')
        for entry in manifest['files']:
            print(f"{role}: verifying/downloading {entry['name']}", flush=True)
            url = f"https://huggingface.co/{manifest['repoID']}/resolve/{manifest['revision']}/{entry['name']}"
            fetch_file(directory, entry, url)
        temporary = directory / 'qwen-install.json.part'
        if temporary.is_symlink():
            raise ValueError('Receipt staging file cannot be a symlink')
        with open_regular(temporary, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 'w') as handle:
            handle.write(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(receipt)
    print(f'{role}: verified installation at {directory}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['list', 'install'])
    parser.add_argument('roles', nargs='*', choices=[*ROLES, 'all'])
    parser.add_argument('--root', type=Path, default=ROOT / 'scratch')
    parser.add_argument('--worker-precision', choices=['3bit', '8bit'], default='8bit')
    args = parser.parse_args()
    roles = list(ROLES) if not args.roles or 'all' in args.roles else list(dict.fromkeys(args.roles))
    if args.action == 'install' and not args.roles:
        parser.error('Choose a role or all explicitly to download weights')
    try:
        for role in roles:
            if args.action == 'install':
                install(role, args.root.expanduser().absolute(), args.worker_precision)
            else:
                data = catalog(role, args.worker_precision)
                print(f"{role:8} {data['repoID']}  {sum(f['bytes'] for f in data['files']) / 1024**3:.2f} GiB  {model_path(role, args.root, args.worker_precision)}")
        return 0
    except (OSError, ValueError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
