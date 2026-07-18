from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent

class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        request = unquote(urlparse(path).path).lstrip('/')
        root = ROOT / ('public' if request.startswith(('models/', 'fixtures/')) else 'dist')
        target = (root / request).resolve()
        return str(target if root in (target, *target.parents) else root / '__invalid__')

if __name__ == '__main__':
    ThreadingHTTPServer(('localhost', 4174), Handler).serve_forever()
