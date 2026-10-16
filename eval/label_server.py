"""Local server for the rollout labelling page.

    python label_server.py        # then open http://localhost:8765

Serves index.html and manifest.json from this folder and saves labels to labels.json
(atomic write on every change). Videos stream straight from huggingface.co.
"""
import http.server
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(HERE, "labels.json")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HERE, **kwargs)

    def do_GET(self):
        if self.path == "/labels":
            data = open(LABELS, "rb").read() if os.path.exists(LABELS) else b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/labels":
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers["Content-Length"]))
        labels = json.loads(body)  # refuse to save anything that is not valid JSON
        tmp = LABELS + ".tmp"
        with open(tmp, "w") as f:
            json.dump(labels, f, indent=1, ensure_ascii=False, sort_keys=True)
        os.replace(tmp, LABELS)
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt, *args):
        if "POST" not in fmt % args:
            super().log_message(fmt, *args)


print(f"http://localhost:{PORT}  (labels -> {LABELS})")
http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
