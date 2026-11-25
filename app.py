"""ARGUS-X launcher: starts the dashboard server and opens it.

    python app.py                 # http://127.0.0.1:8000
    python app.py --port 9000 --no-browser
"""

from __future__ import annotations

import argparse
import threading
import webbrowser

from server import serve


def main() -> None:
    parser = argparse.ArgumentParser(description="ARGUS-X dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
