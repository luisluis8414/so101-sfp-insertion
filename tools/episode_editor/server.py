"""Local episode editor for LeRobot v3.0 datasets.

Serves a browser UI to rewatch episodes (all cameras + joint curves), rate them 1-5,
trim start/end, mark them for deletion and edit their task. Edits are saved to a JSON
sidecar next to the dataset; the dataset itself is never modified. Apply the edits
with export_edits.py.

Run:
  python server.py DATASET_DIR        (needs pandas and pyarrow)
"""

import argparse
import json
import mimetypes
import os
import re
import tempfile
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent


def load_dataset(root: Path) -> dict:
    info = json.loads((root / "meta/info.json").read_text())
    fps = info["fps"]
    cams = [k for k, v in info["features"].items() if v["dtype"] == "video"]

    eps = pd.concat(pd.read_parquet(p) for p in sorted((root / "meta/episodes").rglob("*.parquet")))
    eps = eps.sort_values("episode_index")
    video_tpl = info["video_path"]
    episodes = []
    for _, r in eps.iterrows():
        videos = {}
        for cam in cams:
            rel = video_tpl.format(
                video_key=cam,
                chunk_index=int(r[f"videos/{cam}/chunk_index"]),
                file_index=int(r[f"videos/{cam}/file_index"]),
            )
            videos[cam] = {
                "url": "/dataset/" + rel,
                "from": float(r[f"videos/{cam}/from_timestamp"]),
                "to": float(r[f"videos/{cam}/to_timestamp"]),
            }
        task_list = list(r["tasks"]) if "tasks" in r and r["tasks"] is not None else []
        episodes.append(
            {
                "index": int(r["episode_index"]),
                "length": int(r["length"]),
                "task": task_list[0] if task_list else "",
                "videos": videos,
            }
        )

    data = pd.concat(pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet")))
    names = info["features"]["observation.state"].get("names")
    action_names = info["features"]["action"].get("names")
    series = {}
    for ep, g in data.groupby("episode_index"):
        g = g.sort_values("frame_index")
        state = list(zip(*g["observation.state"].tolist()))
        action = list(zip(*g["action"].tolist()))
        names = names or [f"state{i}" for i in range(len(state))]
        action_names = action_names or [f"action{i}" for i in range(len(action))]
        series[int(ep)] = {
            "names": names,
            "action_names": action_names,
            "state": [[round(float(x), 3) for x in col] for col in state],
            "action": [[round(float(x), 3) for x in col] for col in action],
        }
    return {"fps": fps, "cameras": cams, "episodes": episodes, "series": series, "name": root.name}


class Editor:
    def __init__(self, root: Path, edits_path: Path):
        self.root = root
        self.edits_path = edits_path
        self.lock = threading.Lock()
        self.ds = load_dataset(root)
        if edits_path.exists():
            self.edits = json.loads(edits_path.read_text())
        else:
            self.edits = {"source": str(root), "episodes": {}}

    def save_edits(self, edits: dict) -> None:
        edits["source"] = str(self.root)
        with self.lock:
            self.edits = edits
            fd, tmp = tempfile.mkstemp(dir=self.edits_path.parent, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(edits, f, indent=1, ensure_ascii=False)
            os.replace(tmp, self.edits_path)


def make_handler(editor: Editor):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _json(self, obj, status=HTTPStatus.OK):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._file(HERE / "index.html")
            if path == "/api/meta":
                ds = editor.ds
                return self._json(
                    {
                        "name": ds["name"],
                        "fps": ds["fps"],
                        "cameras": ds["cameras"],
                        "episodes": ds["episodes"],
                        "edits": editor.edits,
                        "edits_path": str(editor.edits_path),
                    }
                )
            m = re.fullmatch(r"/api/series/(\d+)", path)
            if m:
                return self._json(editor.ds["series"][int(m.group(1))])
            if path.startswith("/dataset/"):
                target = (editor.root / path[len("/dataset/"):]).resolve()
                if editor.root.resolve() not in target.parents or not target.is_file():
                    return self.send_error(HTTPStatus.NOT_FOUND)
                return self._file(target)
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if self.path != "/api/edits":
                return self.send_error(HTTPStatus.NOT_FOUND)
            length = int(self.headers.get("Content-Length", 0))
            editor.save_edits(json.loads(self.rfile.read(length)))
            self._json({"ok": True})

        def _file(self, p: Path):
            # Byte-range support so the browser can seek inside the long video files.
            size = p.stat().st_size
            ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            if rng:
                m = re.match(r"bytes=(\d*)-(\d*)", rng)
                if m:
                    if m.group(1):
                        start = int(m.group(1))
                        end = int(m.group(2)) if m.group(2) else size - 1
                    else:
                        start = size - int(m.group(2))
                end = min(end, size - 1)
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            try:
                with open(p, "rb") as f:
                    f.seek(start)
                    remaining = end - start + 1
                    while remaining > 0:
                        chunk = f.read(min(1 << 20, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="dataset root (contains meta/, data/, videos/)")
    ap.add_argument("--edits", type=Path, default=None, help="edits JSON (default: <root>.edits.json)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    edits_path = (args.edits or root.with_name(root.name + ".edits.json")).expanduser().resolve()
    mimetypes.add_type("video/mp4", ".mp4")

    print(f"Loading {root} ...", flush=True)
    editor = Editor(root, edits_path)
    print(f"{len(editor.ds['episodes'])} episodes, cameras: {', '.join(editor.ds['cameras'])}")
    print(f"Edits are saved to {edits_path}")
    url = f"http://127.0.0.1:{args.port}/"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(editor))
    print(f"Editor running at {url}  (Ctrl-C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
