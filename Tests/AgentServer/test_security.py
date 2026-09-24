import asyncio
import json
import os
from pathlib import Path
import stat
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from agent_http import RouteError, validate_http, decode_json, read_body, MAX_BODY
from agent_files import open_regular, private_directory
from agent_models import route_config
from agent_routing import AgentRouter
from test_specialist_packages import installer


def scope(headers=()):
    return {'type':'http','method':'POST','path':'/v1/chat/completions',
            'server':('127.0.0.1',8080),
            'headers':[(b'host',b'127.0.0.1:8080'),(b'content-type',b'application/json'),*headers]}


class BoundaryTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ruby') and shutil.which('git'), 'Requires Ruby and Git')
    def test_symlink_checker_treats_filenames_as_data(self):
        source = Path(__file__).resolve().parents[2] / 'Scripts/check_tracked_symlinks.rb'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'repo with spaces'
            (root / 'Scripts').mkdir(parents=True)
            shutil.copyfile(source, root / 'Scripts/check_tracked_symlinks.rb')
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            name = '$(touch PWNED)'
            (root / name).symlink_to('missing')
            subprocess.run(['git', '-C', str(root), 'add', '--', name], check=True)
            result = subprocess.run(['ruby', str(root / 'Scripts/check_tracked_symlinks.rb')],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn('tracked symlink(s) found', result.stderr)
            self.assertIn(name, result.stderr)
            self.assertFalse((root / 'PWNED').exists())

    def test_loopback_host_and_browser_origin(self):
        validate_http(scope())
        validate_http(scope([(b'origin',b'http://127.0.0.1:8080')]))
        for headers in [[(b'origin',b'https://evil.example')],[(b'origin',b'null')],
                        [(b'sec-fetch-site',b'cross-site')],[(b'host',b'localhost:8080')]]:
            with self.assertRaises(RouteError):validate_http(scope(headers))
        for host in (b'evil.example',b'localhost:9999',b'127.0.0.1:8080@evil.example',b'localhost:invalid'):
            request=scope();request['headers'][0]=(b'host',host)
            with self.assertRaises(RouteError):validate_http(request)

    def test_json_type_and_oversized_declared_body_rejected_before_reading(self):
        request=scope();request['headers'][1]=(b'content-type',b'text/plain')
        with self.assertRaises(RouteError) as error:validate_http(request)
        self.assertEqual(error.exception.status,415)
        for value in (str(MAX_BODY+1).encode(),b'9'*5000):
            with self.assertRaises(RouteError) as error:validate_http(scope([(b'content-length',value)]))
            self.assertEqual(error.exception.status,413)

    def test_invalid_json_cannot_escape_as_internal_error(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}',b'{"x":1e999}', b'{"x":"\\ud800"}',
                    b'['*1100+b'0'+b']'*1100, b'['+b'0,'*100001+b'0]'):
            with self.assertRaises(RouteError):decode_json(raw)
        self.assertEqual(decode_json('{"text":"工具"}'.encode()),{'text':'工具'})

    def test_private_state_and_no_follow_file_opens(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);directory=root/'state';directory.mkdir(mode=0o755)
            private_directory(directory)
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode),0o700)
            target=root/'target';target.write_text('unchanged')
            link=root/'link';link.symlink_to(target)
            with self.assertRaises(OSError):
                with open_regular(link,os.O_WRONLY|os.O_TRUNC,'w'):pass
            hard=root/'hard';os.link(target,hard)
            with self.assertRaises(ValueError):
                with open_regular(hard,os.O_WRONLY|os.O_TRUNC,'w'):pass
            self.assertEqual(target.read_text(),'unchanged')
            fifo=root/'fifo';os.mkfifo(fifo)
            with self.assertRaises(ValueError):
                with open_regular(fifo):pass
            dirlink=root/'dirlink';dirlink.symlink_to(directory)
            with self.assertRaises(OSError):private_directory(dirlink)

    def test_download_redirects_cannot_downgrade_https(self):
        import urllib.request
        request=urllib.request.Request('https://example.test/model')
        redirect=installer.HTTPSRedirects()
        with self.assertRaises(ValueError):
            redirect.redirect_request(request,None,302,'Moved',{},'http://example.test/model')
        self.assertEqual(redirect.redirect_request(request,None,302,'Moved',{},
            'https://cdn.example.test/model').full_url,'https://cdn.example.test/model')


class UploadTests(unittest.IsolatedAsyncioTestCase):
    async def test_chunked_limit_and_timeout(self):
        async def chunk():return {'type':'http.request','body':b'x'*9,'more_body':True}
        with patch('agent_http.MAX_BODY',16):
            with self.assertRaises(RouteError) as error:await read_body(chunk)
            self.assertEqual(error.exception.status,413)
        async def slow():await asyncio.Event().wait()
        with patch('agent_http.BODY_TIMEOUT',.01):
            with self.assertRaises(RouteError) as error:await read_body(slow)
            self.assertEqual(error.exception.status,408)

    async def test_ingress_is_bounded_before_body_reads_and_cancellation_releases_it(self):
        async def backend(*args):self.fail('No request finished uploading')
        async def switch(*args):self.fail('No model should be admitted')
        router=AgentRouter(backend,route_config(['worker']),switch);router.ingress_limit=1
        started=asyncio.Event()
        async def blocked():started.set();await asyncio.Event().wait()
        async def send(message):pass
        task=asyncio.create_task(router(scope(),blocked,send))
        try:
            await asyncio.wait_for(started.wait(),1)
            sent=[]
            async def unexpected():self.fail('Rejected upload body was read')
            async def capture(message):sent.append(message)
            await router(scope(),unexpected,capture)
            self.assertEqual(sent[0]['status'],429)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(router.ingress,0)
