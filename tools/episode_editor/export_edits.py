"""Write a new LeRobot dataset from a source dataset plus the editor's edits JSON.

Drops episodes marked for deletion (and, optionally, those rated below --min-rating),
trims each kept episode to its [start, end) frame range, applies edited tasks, and
re-encodes the videos through LeRobot's own writer, so the result is a regular
LeRobot v3.0 dataset with fresh stats. Ratings, notes and the source episode index
are stored in meta/episode_labels.json of the new dataset. The source is never modified.

Needs a LeRobot environment:
  python export_edits.py SRC_ROOT EDITS_JSON --repo-id local/NAME
"""

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import DEFAULT_FEATURES


def decode_segment(path: Path, start_s: float, n_frames: int, h: int, w: int, fps: int) -> np.ndarray:
    # Start half a frame early so float rounding can never skip the first wanted frame.
    cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{max(0.0, start_s - 0.5 / fps):.6f}", "-i", str(path),
        "-frames:v", str(n_frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    if len(frames) < n_frames:
        if n_frames - len(frames) > 2:
            raise RuntimeError(f"{path}: expected {n_frames} frames from {start_s:.3f}s, got {len(frames)}")
        frames = np.concatenate([frames, np.repeat(frames[-1:], n_frames - len(frames), axis=0)])
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("edits", type=Path)
    ap.add_argument("--repo-id", required=True, help="e.g. local/my_dataset_clean")
    ap.add_argument("--root", type=Path, default=None, help="output dir (default: HF_LEROBOT_HOME/repo-id)")
    ap.add_argument("--min-rating", type=int, default=None, help="also drop episodes rated below this")
    ap.add_argument("--drop-unrated", action="store_true", help="also drop episodes without a rating")
    ap.add_argument("--vcodec", default="libsvtav1", help="video codec for the new dataset (e.g. libsvtav1, h264_nvenc, auto)")
    ap.add_argument("--drop-feature", action="append", default=[], help="leave out this feature (e.g. intervention, to merge DAgger corrections with demos)")
    ap.add_argument("--no-streaming", action="store_true", help="write PNGs and encode per episode (slower, old behaviour)")
    args = ap.parse_args()

    src = args.src.expanduser().resolve()
    edits = json.loads(args.edits.expanduser().read_text())["episodes"]
    info = json.loads((src / "meta/info.json").read_text())
    fps = info["fps"]
    cams = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    features = {
        k: {kk: vv for kk, vv in v.items() if kk != "info"}
        for k, v in info["features"].items()
        if k not in DEFAULT_FEATURES and k not in args.drop_feature
    }
    for k in features:
        features[k]["shape"] = tuple(features[k]["shape"])
    plain_keys = [k for k in features if k not in cams]

    eps = pd.concat(pd.read_parquet(p) for p in sorted((src / "meta/episodes").rglob("*.parquet")))
    eps = eps.sort_values("episode_index").set_index("episode_index")
    data = pd.concat(pd.read_parquet(p) for p in sorted((src / "data").rglob("*.parquet")))

    plan = []
    for ep_idx, r in eps.iterrows():
        e = edits.get(str(ep_idx), {})
        length = int(r["length"])
        start, end = int(e.get("start", 0)), int(e.get("end", length))
        rating = e.get("rating")
        reason = None
        if e.get("delete"):
            reason = "marked for deletion"
        elif args.drop_unrated and rating is None:
            reason = "unrated"
        elif args.min_rating is not None and rating is not None and rating < args.min_rating:
            reason = f"rating {rating} < {args.min_rating}"
        elif not 0 <= start < end <= length:
            reason = f"invalid trim {start}-{end}"
        task = e.get("task") or list(r["tasks"])[0]
        plan.append((int(ep_idx), start, end, rating, task, e.get("notes", ""), reason))

    kept = [p for p in plan if p[6] is None]
    for p in plan:
        if p[6]:
            print(f"skip episode {p[0]}: {p[6]}")
    print(f"exporting {len(kept)} of {len(plan)} episodes, {sum(p[2] - p[1] for p in kept)} frames")

    out = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        features=features,
        root=args.root,
        robot_type=info.get("robot_type"),
        use_videos=True,
        image_writer_threads=0 if not args.no_streaming else 4 * len(cams),
        streaming_encoding=not args.no_streaming,
        rgb_encoder=RGBEncoderConfig(vcodec=args.vcodec),
    )
    pool = ThreadPoolExecutor(max_workers=len(cams))
    t_start = time.perf_counter()
    labels = []
    for new_idx, (ep_idx, start, end, rating, task, notes, _) in enumerate(kept):
        r = eps.loc[ep_idx]
        rows = data[data["episode_index"] == ep_idx].sort_values("frame_index").iloc[start:end]
        n = len(rows)
        def decode(cam):
            h, w = info["features"][cam]["shape"][:2]
            path = src / info["video_path"].format(
                video_key=cam,
                chunk_index=int(r[f"videos/{cam}/chunk_index"]),
                file_index=int(r[f"videos/{cam}/file_index"]),
            )
            return decode_segment(path, float(r[f"videos/{cam}/from_timestamp"]) + start / fps, n, h, w, fps)

        images = dict(zip(cams, pool.map(decode, cams)))
        for i, (_, row) in enumerate(rows.iterrows()):
            frame = {k: np.asarray(row[k], dtype=np.float32) for k in plain_keys}
            frame.update({cam: images[cam][i] for cam in cams})
            frame["task"] = task
            out.add_frame(frame)
        out.save_episode()
        labels.append({"episode_index": new_idx, "source_episode_index": ep_idx, "trim_start": start,
                       "trim_end": end, "rating": rating, "notes": notes})
        el = time.perf_counter() - t_start
        print(f"[{new_idx + 1}/{len(kept)}] source episode {ep_idx}: frames {start}-{end} ({n}), rating {rating} | {el:.0f}s elapsed", flush=True)
    out.finalize()

    (out.root / "meta/episode_labels.json").write_text(
        json.dumps({"source": str(src), "episodes": labels}, indent=1, ensure_ascii=False)
    )
    print(f"done: {out.root}")


if __name__ == "__main__":
    main()
