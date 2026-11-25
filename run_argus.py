"""ARGUS-X command line interface.

Examples
--------
    python run_argus.py --video test.mp4 --max-frames 240
    python run_argus.py --video test.mp4 --out runs/argus_x.mp4 --json runs/report.json
    python run_argus.py --benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from core.benchmark import run_benchmark
from core.config import ARGUSConfig
from core.pipeline import run_video, save_report


def _progress(payload: dict) -> None:
    total = payload.get("total") or 1
    processed = payload.get("processed", 0)
    bar_width = 28
    filled = int(bar_width * processed / max(total, 1))
    bar = "#" * filled + "-" * (bar_width - filled)
    sys.stdout.write(
        f"\r[{bar}] {processed}/{total}  ids {payload.get('identities', 0):<3} "
        f"active {payload.get('active', 0):<3} reacq {payload.get('recoveries', 0):<2}"
    )
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="ARGUS-X long-horizon identity tracker")
    parser.add_argument("--video", default="test.mp4", help="input video path")
    parser.add_argument("--out", default=None, help="annotated output video (.mp4)")
    parser.add_argument("--json", default=None, help="write the full tracking report here")
    parser.add_argument("--frames-dir", default=None, help="also dump annotated JPEG frames")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    parser.add_argument("--width", type=int, default=1280, help="processing width")
    parser.add_argument("--conf", type=float, default=0.20, help="detector confidence floor")
    parser.add_argument("--reid-threshold", type=float, default=0.70, help="appearance gate for recovery")
    parser.add_argument("--no-camera", action="store_true", help="disable camera motion compensation")
    parser.add_argument("--classes", default="0", help="comma separated COCO class ids, empty = all")
    parser.add_argument("--benchmark", action="store_true", help="run the labelled synthetic benchmark")
    args = parser.parse_args()

    if args.benchmark:
        print("Running synthetic long-occlusion benchmark ...")
        result = run_benchmark()
        print(json.dumps(result, indent=2))
        return 0

    if not os.path.isfile(args.video):
        print(f"error: video not found: {args.video}", file=sys.stderr)
        return 2

    classes = tuple(int(c) for c in args.classes.split(",") if c.strip()) if args.classes else None
    config = ARGUSConfig(
        output_width=args.width,
        conf_thresh=args.conf,
        reid_threshold=args.reid_threshold,
        camera_compensation=not args.no_camera,
        classes=classes,
    )

    out_video = args.out or os.path.join("runs", "argus_x_tracked.mp4")
    report = run_video(
        args.video,
        config=config,
        out_video=out_video,
        frame_dir=args.frames_dir,
        max_frames=args.max_frames,
        progress=_progress,
    )
    print()

    summary = report["summary"]
    runtime = report["runtime"]

    print("\n============================== ARGUS-X ==============================")
    print(f" video            : {report['video']['path']}  "
          f"{report['video']['width']}x{report['video']['height']} @ {report['video']['source_fps']} fps")
    print(f" frames           : {report['video']['frames_processed']} "
          f"({runtime['processing_fps']} fps processing, {runtime['seconds']} s)")
    print(f" identities       : {summary['identities']} created, "
          f"{summary['recovered_identities']} re-acquired")
    print(f" verified reacq.  : {summary['recoveries']}   "
          f"rejected: {summary['rejected_recoveries']}")
    print(f" longest occlusion: {summary['max_occlusion_frames']} frames")
    print(f" avg track length : {summary['avg_track_length']} observations")
    print(f" annotated video  : {out_video}")
    if args.json:
        save_report(report, args.json)
        print(f" report           : {args.json}")
    print("=====================================================================\n")

    if report["events"]:
        print("Identity timeline (last events):")
        for event in report["events"][-12:]:
            print(f"  f{event['frame']:>4}  {event['type']:<18} {event['id']:<8} "
                  f"{event['confidence']:.3f}  {event['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
